"""Retrieve PANDA insights and evaluate essays with a local judge."""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .PandaModel import PandaModel
from ..criteria import format_rubric, format_score_schema
from ..paths import PATHS
from ..data.preparation import text_hash
from ..data.runtime import Submission, submission_text


class PandaRetrievalEvaluator(PandaModel):
    def load_transormer_and_tokenizer(self, transformer_url: str = "BAAI/bge-base-en-v1.5", revision: str | None = None):
        """
        Load embedding model for retrieval (SentenceTransformer).
        """
        self.embedder = SentenceTransformer(transformer_url, revision=revision)
        self.embedding_profile = {"model_id": transformer_url, "revision": revision}

    def load_panda_insights(self, path: str):
        """
        Load PANDA insights from CSV file.
        """
        self.panda_insights = pd.read_csv(path)

    def build_retrieval_index(self, essays: np.ndarray | pd.DataFrame):
        """
        Build embedding index for retrieval.
        """
        self.corpus_frame = None
        if isinstance(essays, pd.DataFrame):
            frame = essays.copy()
            if not frame['status'].eq('accepted').all() or not frame['split'].eq('train').all():
                raise ValueError("Reference corpus must contain only accepted train rows")
            if frame['essay_id'].duplicated().any() or frame['essay_hash'].duplicated().any():
                raise ValueError("Reference IDs and essay hashes must be unique")
            if 'essay_id' not in self.panda_insights or self.panda_insights['essay_id'].duplicated().any():
                raise ValueError("Reference insights require unique essay IDs")
            if set(frame.essay_id) != set(self.panda_insights.essay_id):
                raise ValueError("Reference corpus and insight essay IDs do not match")
            manifest = frame.attrs.get('manifest_sha256')
            if not manifest or 'dataset_manifest' not in self.panda_insights or not self.panda_insights.dataset_manifest.eq(manifest).all():
                raise ValueError("Reference insights belong to a different dataset release")
            self.panda_insights = self.panda_insights.set_index('essay_id', drop=False).loc[frame.essay_id].reset_index(drop=True)
            self.corpus_frame = frame.reset_index(drop=True)
            self.retrieval_metadata = {"dataset_manifest": manifest, "essay_ids": frame.essay_id.tolist(),
                                       "embedding_profile": getattr(self, 'embedding_profile', None)}
            essay_texts = frame.essay.to_numpy()
        else:
            essay_texts = essays
        self.essay_corpus = essay_texts
        self.essay_embeddings = self.embedder.encode(
            essay_texts,
            batch_size=64,
            show_progress_bar=True
        )

    def find_similar_essays(self, train_essay: Submission, num_similar: int = 1):
        """
        Find top-k similar essays based on cosine similarity.
        """
        assert self.essay_embeddings is not None, "Embeddings not initialized"
        
        submission = train_essay
        query_essay = submission['essay'] if isinstance(submission, dict) else submission
        query_emb = self.embedder.encode([query_essay])
        similarities = cosine_similarity(query_emb, self.essay_embeddings)[0]
        
        candidates = np.argsort(similarities)[::-1]
        if self.corpus_frame is not None:
            query_id = submission.get('essay_id') if isinstance(submission, dict) else None
            candidates = [i for i in candidates if self.corpus_frame.iloc[i].essay_id != query_id
                          and self.corpus_frame.iloc[i].essay_hash != text_hash(query_essay)]
        top_idx = candidates[:num_similar]
        
        similar_essays = [self.essay_corpus[i] for i in top_idx]
        similar_insights = [self.panda_insights.iloc[i] for i in top_idx]
        
        return similar_essays, similar_insights

    def save_results_json(self, new_entry, save_path):
        """
        Save results incrementally to JSON (append/update).
        """
        # если файл уже есть — читаем
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        else:
            data = []
        data.append(new_entry)
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def build_evaluation_messages(self, essay: Submission, similar_insights) -> list[dict[str, str]]:
        """Build the legacy retrieval prompt without models or index access."""
        essay = submission_text(essay)
        insights_text = "\n\n".join([
            f"Insight {i+1}:\n{ins}"
            for i, ins in enumerate(similar_insights)
        ])
        
        prompt = f"""
        ### Instruction:
        You are an IELTS writing evaluator.
        Below are insights from similar essays and their evaluations:

        {insights_text}

        Now evaluate the following essay using given insights and return ONLY JSON, no text explanation.
        Scores range from 0 to 9 with step 0.5.

        Return format:
        {format_score_schema()}

        Essay:
        {essay}

        ### Answer:
        """
        
        return [
            {
                "role": "system",
                "content": (
                    "You are an IELTS examiner. Return ONLY valid JSON.\n"
                    "Consider requirements below for each IELTS criterion while evaluating the essay.\n\n"
                    f"{format_rubric()}"
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def evaluate_essay(self, essay: Submission, num_similar: int = 1):
        """Evaluate essay using retrieved insights in the offline workflow."""
        results = []
        _, similar_insights = self.find_similar_essays(essay, num_similar=num_similar)
        message = self.build_evaluation_messages(essay, similar_insights)
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.2,
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Answer:")[-1].strip()
        match = re.search(r"\{[\s\S]*?\}", response)
        if match:
            answer = match.group(0)
            results.append(answer)
            self.save_results_json(
                answer, save_path=PATHS["results"]["panda_evaluations"]
            )
        else:
            print("JSON не найден:", response)
            return None
        
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            print("Ошибка при декодировании:", answer)
            return None

    def evaluate_llm(self, essays: np.ndarray, num_similar: int = 1):
        """
        Evaluate multiple essays and save results.
        """
        for essay in essays:
            result = self.evaluate_essay(essay, num_similar=num_similar)
            print(f"Evaluation result: {result}")
