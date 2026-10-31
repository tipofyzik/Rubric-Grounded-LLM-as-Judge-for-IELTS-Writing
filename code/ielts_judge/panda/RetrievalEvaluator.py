"""Retrieve PANDA insights and evaluate essays with a local judge."""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .PandaModel import PandaModel
from ..criteria import PUBLIC_CRITERION_IDS
from ..prompts import JudgeContext, ReferenceContext, build_judge_messages, submission_parts
from ..scores import parse_scores, public_scores
from ..llm.generation import generate_response
from ..paths import PATHS
from ..data.preparation import text_hash
from ..data.experiment import validate_reference_corpus
from ..data.runtime import Submission


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

    def build_retrieval_index(self, essays: np.ndarray | pd.DataFrame, *, experiment_manifest=None):
        """
        Build embedding index for retrieval.
        """
        if experiment_manifest is not None:
            if not isinstance(essays, pd.DataFrame):
                raise ValueError("Experiment retrieval requires identified train rows")
            validate_reference_corpus(essays, experiment_manifest)
        self.corpus_frame = None
        if isinstance(essays, pd.DataFrame):
            frame = essays.copy()
            if not bool(frame['status'].eq('accepted').all()) or not bool(frame['split'].eq('train').all()):
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
        """Use the common prompt with identified source scores and original insights."""
        task, text = submission_parts(essay)
        references = []
        for item in similar_insights:
            if isinstance(item, ReferenceContext):
                references.append(item)
                continue
            identity = item.get("essay_id")
            frame = getattr(self, "corpus_frame", None)
            if frame is None or identity is None:
                raise ValueError("Offline references require identified source scores")
            matches = frame.loc[frame.essay_id == identity]
            if len(matches) != 1:
                raise ValueError("Reference ID does not match the train corpus")
            row = matches.iloc[0]
            labels = public_scores({key: row[key] for key in PUBLIC_CRITERION_IDS})
            explanation = item.get("insights")
            if not isinstance(explanation, str):
                explanation = "\n".join(str(value) for key, value in item.items() if key.startswith("insight_"))
            references.append(ReferenceContext(str(identity), str(row.task_prompt), labels, explanation))
        return build_judge_messages(task, text, JudgeContext(references=tuple(references)))

    def evaluate_essay(self, essay: Submission, num_similar: int = 1):
        """Offline inference shares messages, completion slicing, EOS and score parsing."""
        _, similar_insights = self.find_similar_essays(essay, num_similar=num_similar)
        generation = generate_response(
            self.model, self.tokenizer, self.build_evaluation_messages(essay, similar_insights),
            max_new_tokens=256, generation_settings={"do_sample": False},
            criterion_ids=PUBLIC_CRITERION_IDS,
        )
        try:
            result = parse_scores(generation.raw_text)
        except ValueError:
            return None
        self.save_results_json(result, save_path=PATHS["results"]["panda_evaluations"])
        return result

    def evaluate_llm(self, essays: np.ndarray, num_similar: int = 1):
        """
        Evaluate multiple essays and save results.
        """
        for essay in essays:
            result = self.evaluate_essay(essay, num_similar=num_similar)
            print(f"Evaluation result: {result}")
