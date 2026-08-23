# Rubric-Grounded LLM-as-Judge for IELTS Writing Evaluation

This repository contains the official codebase, dataset preprocessing utilities, and evaluation scripts for the Bachelor Thesis project: **"Rubric-Grounded LLM-as-Judge for IELTS Writing: Reliability, Calibration, and Preference Adaptation Using an Automated Essay Scoring Expert Model"**.

> 📄 <a href="https://docs.google.com/viewer?url=https://github.com/user-attachments/files/31278259/Rubric-Grounded.LLM-as-Judge.for.IELTS.Writing.pdf" target="_blank">
**Read the full thesis**
</a>

## 📌 Abstract & Overview
Automated Essay Scoring (AES) for standardized examinations like IELTS requires multi-criteria assessment strictly grounded in evaluation rubrics. This project investigates a hybrid evaluation framework using **Small Language Models (SLMs)** as judges. 

The system integrates three core components:
1. **AES Expert Model:** A multi-task ordinal regression model based on fine-tuned `RoBERTa-base` to generate preliminary domain-specific score estimates.
2. **SLMs as Evaluators:** Performance evaluation of instruction-tuned models (`Qwen2-1.5B`, `Qwen2.5-1.5B`, `Llama-3.2-3B`, and `Mistral-7B`) under varying structured prompt guidance.
3. **PANDA Preference Adaptation & Preliminary Guidance:** Incorporating domain insights derived via the PANDA framework alongside preliminary expert predictions to improve scoring consistency, calibration, and rank agreement.
---

## 🚀 Key Features
* **Multi-Criteria Assessment:** Evaluates all four official IELTS Writing Task 2 scoring criteria:
  * Task Achievement (TA)
  * Coherence and Cohesion (CC)
  * Lexical Resource (LR)
  * Grammatical Range and Accuracy (GRA)
* **PANDA Preference Adaptation:** Range-specific insight extraction focusing on why an essay received a specific score versus neighboring bands ($Score \pm 1$).
* **Hybrid Prompting Strategies:** Seamlessly combines zero-shot instructions, condensed IELTS rubrics, preference insights, and preliminary expert scores.
* **Parameter-Efficient Fine-Tuning (LoRA):** Instruction tuning applied to attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) using PEFT.
* **Rigorous Evaluation Framework:** Comprehensive analysis using **Quadratic Weighted Kappa (QWK)**, **RMSE**, **MAE**, **Adjacent Accuracy ($\pm 0.5$ band points)**, **Pearson ($r$) & Spearman ($r_s$) Correlations**, and **Calibration Curves**.
---

## 🔬 Experimental Setup & Prompting Strategies
We benchmarked 6 distinct prompting configurations across non-fine-tuned and fine-tuned SLMs:
* `p1`: **Zero-shot** – Baseline prompt requiring structured JSON output.
* `p2`: **Rubric-based (LLM-as-Judge)** – Includes shortened IELTS evaluation criteria.
* `p3`: **PANDA-based** – Enriched with retrieved preference insights.
* `p4`: **Rubric + PANDA** – Combined rubric grounding and preference adaptation.
* `p5`: **PANDA + Preliminary Scores** – Integrates preference insights with preliminary score predictions from the RoBERTa expert.
* `p6`: **Rubric + PANDA + Preliminary Scores** – Full hybrid prompt configuration.
---

## 📊 Key Results & Findings
* **Top Performing Prompt:** The PANDA-based approach combined with preliminary expert scores (`p5`) yielded the highest scoring agreement and calibration across models.
* **Non-Fine-Tuned Models:** `Llama-3.2-3B-Instruct` demonstrated superior prompt-adapted evaluation performance across 5 out of 6 prompt strategies.
<img width="8980" height="4974" alt="mean_qwk_across_runs" src="https://github.com/user-attachments/assets/987d47d5-5e3f-45e5-9c46-bbbe89a49f27" />
* **Fine-Tuned Models:** `Mistral-7B-Instruct-v0.2` fine-tuned with LoRA achieved the highest correlation and best alignment with perfect calibration lines among other fine-tuned models.
<img width="9601" height="4974" alt="mean_qwk_across_runs" src="https://github.com/user-attachments/assets/ef93860f-7c48-4660-b5fd-670e1dc009ed" />
* **AES RoBERTa Expert:** Provided reliable baseline estimates ($QWK \approx 0.650 - 0.661$) used to guide SLM evaluation.
---

## 🛠️ Tech Stack & Hardware Setup
* **Frameworks & Libraries:** PyTorch, Hugging Face `transformers`, `peft`, `datasets`, `accelerate`.
* **Hardware:**
  * **Fine-Tuning:** NVIDIA L4 GPU (24 GB)
  * **Inference:** 2x NVIDIA T4 GPUs (16 GB)
* **Inference Parameters:** `max_new_tokens=200`, `temperature=0.2`, `do_sample=True`.
---

## 📁 Repository Structure
```text
├── code/
│   ├── DatasetTransformer.py    # Preprocessing and dataset transformation utilities
│   ├── LLMAggregator.py         # Aggregates evaluation results and metrics across models
│   ├── LLMEvaluator.py          # Core evaluation pipeline for LLM-as-a-Judge assessment
│   ├── LLMFineTuner.py          # Fine-tuning module for SLMs using LoRA (PEFT)
│   ├── LLMResponseObtainer.py   # Handles response generation from LLMs/SLMs
│   ├── OrdinalRegression.py     # RoBERTa-base multi-task ordinal regression expert model
│   ├── PANDA.py                 # Preference Adaptation & Range Insight extraction framework
│   ├── RawDataAnalyzer.py       # Data analysis and summary statistics for raw dataset
│   ├── RegressorEvaluator.py    # Evaluation metrics for the RoBERTa regressor expert
│   ├── config.json              # Configuration file for paths, models, and hyperparameters
│   ├── main.py                  # Main execution pipeline entry point
│   └── prompt image.py          # Prompt generation and template formatting utilities
```

---

## 📄 Citation
If you use this repository or reference the findings in your work, please cite:  
```bibtex
@thesis{ielts_slm_judge_2026,
  author       = {Sergei Liapunov},
  title        = {Rubric-Grounded LLM-as-Judge for IELTS Writing: Reliability, Calibration, and Preference Adaptation Using an Automated Essay Scoring Expert Model},
  school       = {IU International University of Applied Sciences},
  year         = {2026},
  type         = {Bachelor's Thesis}
}
```
