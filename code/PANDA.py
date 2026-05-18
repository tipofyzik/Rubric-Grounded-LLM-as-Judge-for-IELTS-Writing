from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import pandas as pd
import numpy as np
import torch
import math
import time
import json
import re
import os

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from huggingface_hub import login



class PANDA:
    """
    """
    def __init__(self, SERVICE_TOKEN: str) -> None:
        """
        Initialize the LLMResponseObtainer with the necessary service token.

        Args:
            SERVICE_TOKEN (str): The API token for accessing the LLM service.
        """
        self.SERVICE_TOKEN = SERVICE_TOKEN
        self.band_requirements = {
            "Task Response": {
                9: "Band 9: Fully answer all parts of the question; Clear, strong position throughout; Ideas fully explained + well supported",
                8: "Band 8: Answer all parts well; Clear position; Ideas developed, minor gaps allowed",
                7: "Band 7: Answer main parts; Clear position; Ideas developed but may lack depth or precision",
                6: "Band 6: Addresses the task but unevenly;Position is present but may be unclear or repetitive; Ideas relevant but underdeveloped",
                5: "Band 5: Partially answers the question; Position unclear or weak; Ideas limited, repetitive, or irrelevant",
                4: "Band 4: Misunderstands or only partly addresses task; Position hard to find; Ideas unclear, poorly supported",
                3: "Band 3: Does not properly address task; No clear position; Very few ideas, mostly irrelevant",
                2: "Band 2: Barely related to task; No position; Almost no development",
                1: "Band 1: Off-topic or <20 words; No meaningful response"
            },
            "Coherence & Cohesion": {
                9: "Band 9: Perfect flow, effortless to read; Cohesion invisible (natural linking); Paragraphing excellent",
                8: "Band 8: Logical organisation; Good linking, minor lapses; Clear paragraphs",
                7: "Band 7: Clear progression; Range of linking words (some errors/overuse; Generally good paragraphing",
                6: "Band 6: Generally logical organisation; Some faulty or mechanical linking; Paragraphing inconsistent",
                5: "Band 5: Some organisation but not logical overall; Weak linking between sentences; Paragraphing weak or missing",
                4: "Band 4: No clear progression; Ideas poorly connected; Basic linking, often incorrect",
                3: "Band 3: No logical organisation; Minimal linking words; Hard to follow",
                2: "Band 2: Almost no organisation; No clear links",
                1: "Band 1: No structure at all"
            },
            "Lexical Resource": {
                9: "Band 9: Wide, precise, natural vocabulary; Sophisticated word choice; Almost no errors",
                8: "Band 8: Wide vocabulary; Some advanced/idiomatic use; Occasional minor mistakes",
                7: "Band 7: Enough range for flexibility; Some less common words; Occasional wrong word/awkward phrasing",
                6: "Band 6: Adequate but limited range; Some repetition or imprecision; Errors noticeable but meaning clear",
                5: "Band 5: Basic vocabulary; Frequent repetition; Noticeable word choice errors",
                4: "Band 4: Very limited vocabulary; Frequent wrong word use; Meaning sometimes unclear",
                3: "Band 3: Extremely limited vocabulary; Errors dominate; Meaning often unclear",
                2: "Band 2: Only a few words/phrases; Mostly memorised language",
                1: "Band 1: Almost no usable vocabulary"
            },
            "Grammatical Range & Accuracy": {
                9: "Band 9: Wide range of complex structures; Almost all sentences error-free",
                8: "Band 8: Wide range used well; Most sentences correct; Occasional minor errors",
                7: "Band 7: Mix of simple + complex; Frequent error-free sentences; Some grammar mistakes",
                6: "Band 6: Limited flexibility; Errors in complex sentences; Errors present but don’t block meaning",
                5: "Band 5: Mostly simple sentences; Errors frequent, sometimes confusing",
                4: "Band 4: Very limited structures; Frequent errors, may block meaning",
                3: "Band 3: Attempted sentences but mostly incorrect; Meaning often lost",
                2: "Band 2: Almost no correct sentences; Grammar barely controlled",
                1: "Band 1: No sentence structure"
            }
        }

    def load_model_and_tokenizer(self, model_url: str):
        """
        Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): The URL or identifier of the model to load.
        """
        login(self.SERVICE_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            dtype=torch.float16,
            device_map="auto"
        )
        accelerator = Accelerator()
        self.model = accelerator.prepare(self.model)

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
                prompt = f"""
                ### Instruction:
                Your task is to explain why the expert gave the selected score for each criterion, 
                using the band descriptors provided. Focus on differences between adjacent bands and link 
                your reasoning to specific evidence from the essay.
                
                Scores given by expert:
                - Task Response: {score[0]}
                - Coherence & Cohesion: {score[1]}
                - Lexical Resource: {score[2]}
                - Grammatical Range & Accuracy: {score[3]}
                
                For EACH criterion output EXACTLY 3 bullet points:
                [Criterion name]
                - Evidence:
                - Why not higher (score + 1): 
                - Why not lower (score - 1):
                
                Essay:
                {essay}
                
                Scoring Requirements:
                TASK RESPONSE
                {self.band_requirements["Task Response"][max(1, math.trunc(score[0]) - 1)]}
                {self.band_requirements["Task Response"][max(1, math.trunc(score[0]))]}
                {self.band_requirements["Task Response"][min(9, math.trunc(score[0]) + 1)]}
                
                COHERENCE & COHESION
                {self.band_requirements["Coherence & Cohesion"][max(1, math.trunc(score[1]) - 1)]}
                {self.band_requirements["Coherence & Cohesion"][max(1, math.trunc(score[1]))]}
                {self.band_requirements["Coherence & Cohesion"][min(9, math.trunc(score[1]) + 1)]}
                
                LEXICAL RESOURCE
                {self.band_requirements["Lexical Resource"][max(1, math.trunc(score[2]) - 1)]}
                {self.band_requirements["Lexical Resource"][max(1, math.trunc(score[2]))]}
                {self.band_requirements["Lexical Resource"][min(9, math.trunc(score[2]) + 1)]}
                
                GRAMMATICAL RANGE & ACCURACY
                {self.band_requirements["Grammatical Range & Accuracy"][max(1, math.trunc(score[3]) - 1)]}
                {self.band_requirements["Grammatical Range & Accuracy"][max(1, math.trunc(score[3]))]}
                {self.band_requirements["Grammatical Range & Accuracy"][min(9, math.trunc(score[3]) + 1)]}
                
                ### Answer:
                """
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
            file_exists = os.path.exists("/kaggle/working/panda_insights.csv")
            
            batch_results = []
            for j in range(len(batch_essays)):
                responses = decoded[num_insights*j:num_insights*(j+1)]
                cleaned = [r.split("### Answer:")[-1].strip() for r in responses]
                batch_results.append([batch_essays[j]] + cleaned)
            self.save_panda_insights(
                batch_results,
                save_path="/kaggle/working/panda_insights.csv",
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
    
    

    # Evaluating via PANDA
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
        similar_insights = [self.panda_insights[i] for i in top_idx]
        
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
        {{
        "task_achievement": number,
        "coherence_and_cohesion": number,
        "lexical_resource": number,
        "grammatical_range_and_accuracy": number
        }}

        Essay:
        {essay}

        ### Answer:
        """
        
        message = [
            {"role": "system", "content": """You are an IELTS examiner. Return ONLY valid JSON./n
            Consider requirements below for each IELTS criteria while evaluating given essay.
                    
            TASK RESPONSE
            Band 9: Fully answer all parts of the question; Clear, strong position throughout; Ideas fully explained + well supported
            Band 8: Answer all parts well; Clear position; Ideas developed, minor gaps allowed
            Band 7: Answer main parts; Clear position; Ideas developed but may lack depth or precision
            Band 6: Addresses the task but unevenly;Position is present but may be unclear or repetitive; Ideas relevant but underdeveloped
            Band 5: Partially answers the question; Position unclear or weak; Ideas limited, repetitive, or irrelevant
            Band 4: Misunderstands or only partly addresses task; Position hard to find; Ideas unclear, poorly supported
            Band 3: Does not properly address task; No clear position; Very few ideas, mostly irrelevant
            Band 2: Barely related to task; No position; Almost no development
            Band 1: Off-topic or <20 words; No meaningful response
            
            COHERENCE & COHESION
            Band 9: Perfect flow, effortless to read; Cohesion invisible (natural linking); Paragraphing excellent
            Band 8: Logical organisation; Good linking, minor lapses; Clear paragraphs
            Band 7: Clear progression; Range of linking words (some errors/overuse; Generally good paragraphing
            Band 6: Generally logical organisation; Some faulty or mechanical linking; Paragraphing inconsistent
            Band 5: Some organisation but not logical overall; Weak linking between sentences; Paragraphing weak or missing
            Band 4: No clear progression; Ideas poorly connected; Basic linking, often incorrect
            Band 3: No logical organisation; Minimal linking words; Hard to follow
            Band 2: Almost no organisation; No clear links
            Band 1: No structure at all
            
            LEXICAL RESOURCE
            Band 9: Wide, precise, natural vocabulary; Sophisticated word choice; Almost no errors
            Band 8: Wide vocabulary; Some advanced/idiomatic use; Occasional minor mistakes
            Band 7: Enough range for flexibility; Some less common words; Occasional wrong word/awkward phrasing
            Band 6: Adequate but limited range; Some repetition or imprecision; Errors noticeable but meaning clear
            Band 5: Basic vocabulary; Frequent repetition; Noticeable word choice errors
            Band 4: Very limited vocabulary; Frequent wrong word use; Meaning sometimes unclear
            Band 3: Extremely limited vocabulary; Errors dominate; Meaning often unclear
            Band 2: Only a few words/phrases; Mostly memorised language
            Band 1: Almost no usable vocabulary
            
            GRAMMATICAL RANGE & ACCURACY
            Band 9: Wide range of complex structures; Almost all sentences error-free
            Band 8: Wide range used well; Most sentences correct; Occasional minor errors
            Band 7: Mix of simple + complex; Frequent error-free sentences; Some grammar mistakes
            Band 6: Limited flexibility; Errors in complex sentences; Errors present but don’t block meaning
            Band 5: Mostly simple sentences; Errors frequent, sometimes confusing
            Band 4: Very limited structures; Frequent errors, may block meaning
            Band 3: Attempted sentences but mostly incorrect; Meaning often lost
            Band 2: Almost no correct sentences; Grammar barely controlled
            Band 1: No sentence structure
            """
            },
            {"role": "user", "content": prompt}]
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
            self.save_results_json(answer, save_path = "/kaggle/working/panda_evaluation_results.json")
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