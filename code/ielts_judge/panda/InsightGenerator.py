"""Generate PANDA criterion insights from expert-scored essays."""

import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .PandaModel import PandaModel
from ..criteria import CRITERION_IDS, IELTS_CRITERIA
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

    def build_insight_prompt(self, essay: str, score: np.ndarray) -> str:
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
            criterion_score = math.trunc(score_by_criterion[criterion_id])
            bands = definition["bands"]
            criterion_bands = (
                max(1, criterion_score - 1),
                max(1, criterion_score),
                min(9, criterion_score + 1),
            )
            scoring_requirements.append(
                f"{definition['display_name']}\n"
                + "\n".join(bands[band] for band in criterion_bands)
            )
        scoring_requirements_text = "\n\n".join(scoring_requirements)
        return f"""
                ### Instruction:
                Your task is to explain why the expert gave the selected score for each criterion,
                using the band descriptors provided. Focus on differences between adjacent bands and link
                your reasoning to specific evidence from the essay.

                Scores given by expert:
                {criteria_scores}

                For EACH criterion output EXACTLY 3 bullet points:
                [Criterion name]
                - Evidence:
                - Why not higher (score + 1):
                - Why not lower (score - 1):

                Essay:
                {essay}

                Scoring Requirements:
                {scoring_requirements_text}

                ### Answer:
                """

    def retrieve_insights(self, essays: np.ndarray, scores: np.ndarray,
                          num_insights: int = 1, batch_size=12) -> None:
        """
        """
        for i in range(0, len(essays), batch_size):
            batch_essays = essays[i:i+batch_size]
            batch_scores = scores[i:i+batch_size]
            batch_prompts = []
            torch.cuda.empty_cache()
            for essay, score in zip(batch_essays, batch_scores):
                prompt = self.build_insight_prompt(essay, score)
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
