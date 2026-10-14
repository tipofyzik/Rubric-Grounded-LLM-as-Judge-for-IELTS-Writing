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
from ..criteria import IELTS_CRITERIA, format_score_schema
from ..paths import PATHS


class PandaRetrievalEvaluator(PandaModel):
    def load_transormer_and_tokenizer(self, transformer_url: str = "BAAI/bge-base-en-v1.5"):
        """
        Load embedding model for retrieval (SentenceTransformer).
        """
        self.embedder = SentenceTransformer(transformer_url)

    def load_panda_insights(self, path: str):
        """
        Load PANDA insights from CSV file.
        """
        self.panda_insights = pd.read_csv(path)

    def build_retrieval_index(self, essays: np.ndarray):
        """
        Build embedding index for retrieval.
        """
        self.essay_corpus = essays
        self.essay_embeddings = self.embedder.encode(
            essays,
            batch_size=64,
            show_progress_bar=True
        )

    def find_similar_essays(self, train_essay: str, num_similar: int = 1):
        """
        Find top-k similar essays based on cosine similarity.
        """
        assert self.essay_embeddings is not None, "Embeddings not initialized"
        
        query_emb = self.embedder.encode([train_essay])
        similarities = cosine_similarity(query_emb, self.essay_embeddings)[0]
        
        top_idx = np.argsort(similarities)[::-1][:num_similar]
        
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

    def evaluate_essay(self, essay: str, num_similar: int = 1):
        """
        Evaluate essay using retrieved insights.
        """
        results = []
        _, similar_insights = self.find_similar_essays(
            essay, num_similar=num_similar
        )    
            
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
        
        rubric_sections = []
        for definition in IELTS_CRITERIA["criteria"].values():
            band_descriptors = definition["bands"]
            ordered_bands = sorted(band_descriptors, reverse=True)
            rubric_sections.append(
                definition["display_name"].upper()
                + "\n"
                + "\n".join(band_descriptors[band] for band in ordered_bands)
            )
        rubric_text = "\n\n".join(rubric_sections)
        message = [
            {
                "role": "system",
                "content": (
                    "You are an IELTS examiner. Return ONLY valid JSON.\n"
                    "Consider requirements below for each IELTS criterion while evaluating the essay.\n\n"
                    f"{rubric_text}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
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
