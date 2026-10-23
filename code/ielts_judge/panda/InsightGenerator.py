"""Generate PANDA criterion insights from expert-scored essays."""

import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .PandaModel import PandaModel
from ..criteria import CRITERION_IDS, IELTS_CRITERIA, PUBLIC_CRITERION_IDS
from ..paths import PATHS


class PandaInsightGenerator(PandaModel):
    def __init__(self, service_token: str) -> None:
        """
        Initialize the LLMResponseObtainer with the necessary service token.

        Args:
            service_token (str): Hugging Face token, or an empty string for public models.
        """
        super().__init__(service_token)
        self.band_requirements = {
            definition["display_name"]: definition["bands"]
            for definition in IELTS_CRITERIA.values()
        }

    def build_insight_prompt(self, essay: str, score: np.ndarray, task_prompt: str | None = None) -> str:
        """Build the offline explanation prompt without loading a model."""
        score_by_criterion = dict(zip(CRITERION_IDS, score))
        criteria_scores = "\n".join(
            f"- {IELTS_CRITERIA[criterion_id]['display_name']}: "
            f"{score_by_criterion[criterion_id]}"
            for criterion_id in CRITERION_IDS
        )
        scoring_requirements = []
        for criterion_id in CRITERION_IDS:
            definition = IELTS_CRITERIA[criterion_id]
            value = float(score_by_criterion[criterion_id])
            if not math.isfinite(value) or not 0 <= value <= 9:
                raise ValueError("Criterion score must be within 0-9")
            criterion_score = math.trunc(value)
            bands = definition["bands"]
            criterion_bands = [band for band in range(max(0, criterion_score - 1), min(9, criterion_score + 1) + 1) if band in bands]
            scoring_requirements.append(
                f"{definition['display_name']}\n"
                + "\n".join(bands[band] for band in criterion_bands)
            )
        scoring_requirements_text = "\n\n".join(scoring_requirements)
        return f"""
                ### Instruction:
                Your task is to explain the source dataset's selected score for each criterion,
                using the band descriptors provided. Focus on differences between adjacent bands and link
                your reasoning to specific evidence from the essay.

                Source dataset scores (not independently certified examiner labels):
                {criteria_scores}

                For EACH criterion output EXACTLY 3 bullet points:
                [Criterion name]
                - Evidence:
                - Why not higher (score + 1):
                - Why not lower (score - 1):

                Original task:
                {task_prompt if task_prompt else 'Not supplied; Task Response cannot be checked against the original task.'}

                A source score of 0 is retained as 0; absent band-0 descriptors must not be replaced by band 1.

                Essay:
                {essay}

                Scoring Requirements:
                {scoring_requirements_text}

                ### Answer:
                """

    def retrieve_insights(self, essays: np.ndarray, scores: np.ndarray,
                          num_insights: int = 1, batch_size=12, submissions=None) -> None:
        """
        """
        if len(essays) != len(scores) or (submissions is not None and len(submissions) != len(essays)):
            raise ValueError("Essays, scores and submission records must align")
        if submissions is not None and len({row['essay_id'] for row in submissions}) != len(submissions):
            raise ValueError("Submission essay IDs must be unique")
        for i in range(0, len(essays), batch_size):
            batch_essays = essays[i:i+batch_size]
            batch_scores = scores[i:i+batch_size]
            batch_prompts = []
            torch.cuda.empty_cache()
            for offset, (essay, score) in enumerate(zip(batch_essays, batch_scores)):
                task = submissions[i + offset]['task_prompt'] if submissions is not None else None
                prompt = self.build_insight_prompt(essay, score, task_prompt=task)
                message = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(text)
                
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            start_time = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    num_return_sequences=num_insights,
                    do_sample=True
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            insights_path = PATHS["results"]["panda_insights"]
            os.makedirs(os.path.dirname(insights_path), exist_ok=True)
            file_exists = os.path.exists(insights_path)
            
            batch_results = []
            for j in range(len(batch_essays)):
                responses = decoded[num_insights*j:num_insights*(j+1)]
                cleaned = [r.split("### Answer:")[-1].strip() for r in responses]
                batch_results.append([batch_essays[j]] + cleaned)
            if submissions is not None:
                records = []
                for offset, values in enumerate(batch_results):
                    record = submissions[i + offset]
                    records.append({**record, **dict(zip(PUBLIC_CRITERION_IDS, map(float, batch_scores[offset]))),
                                    **{f'insight_{k}': v for k, v in enumerate(values[1:], 1)}})
                pd.DataFrame(records).to_csv(insights_path, mode='a', header=not file_exists, index=False)
                continue
            self.save_panda_insights(
                batch_results,
                save_path=insights_path,
                append=True,
                write_header=not file_exists
            )
            file_exists = True
            print(f"Time spent on {batch_size} essays: {time.time() - start_time}")

    def save_panda_insights(self, insights, save_path, append=False, write_header=True):
        columns = ["essay"] + [f"insight_{i}" for i in range(1, len(insights[0]))]
        df = pd.DataFrame(insights, columns=columns)
        
        df.to_csv(
            save_path,
            mode="a" if append else "w",
            header=write_header,
            index=False
        )
