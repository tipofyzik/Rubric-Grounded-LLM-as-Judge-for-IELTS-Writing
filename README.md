# Rubric-Grounded LLM-as-Judge for IELTS Writing:
### Reliability, Calibration, and Preference Adaptation Using an Automated Essay Scoring Expert Model

## 1. Introduction & Overview

Automated Essay Scoring (AES) remains a complex and challenging problem within Natural Language Processing (NLP) for educational technology. In high-stakes, standardized language examinations such as the **International English Language Testing System (IELTS)**, essay assessment requires far more than surface-level grammar checks or general semantic understanding. Evaluators must strictly adhere to predefined, multidimensional scoring rubrics across four distinct core criteria:
- **Task Achievement (TA)**
- **Coherence and Cohesion (CC)**
- **Lexical Resource (LR)**
- **Grammatical Range and Accuracy (GRA)**

This multidimensional requirement renders automated essay evaluation a highly demanding benchmark for modern machine learning systems.

---

### 1.1 Context & Problem Statement

With recent advances in Large Language Models (LLMs), the **LLM-as-a-Judge** framework has gained widespread adoption as a general-purpose evaluation paradigm. Rather than training strictly discriminative task-specific classifiers, LLMs utilize prompt-based reasoning and in-context learning to perform open-ended evaluation tasks.

However, despite their strong natural language understanding and generation capabilities, applying LLMs as structured evaluators reveals several persistent limitations:
1. **Sensitivity to Prompt Formulation**: Minor modifications in prompt wording or structure can lead to significant variances in output score predictions.
2. **Score Instability & High Variance**: Generative evaluators frequently produce inconsistent scores across repeated evaluations under identical configurations.
3. **Limited Rubric Alignment**: General-purpose LLMs struggle to strictly align their outputs with official band score criteria (e.g., IELTS half-band steps from 0 to 9).

Furthermore, existing literature overwhelmingly focuses on proprietary or large-scale language models, leaving the behavior of **Small Language Models (SLMs)**—typically defined as models with $\le 10\text{B}$ parameters—relatively unexplored in automated judgment settings. Unlike larger architectures, SLMs possess limited contextual recall, constrained attention windows, and heightened sensitivity to prompt structures. Consequently, SLMs serve as an ideal testbed for investigating how external structured guidance, preliminary expert signals, and preference adaptation influence scoring reliability and output calibration.

---

### 1.2 Proposed Hybrid Evaluation Framework

To address these constraints, this project proposes a **hybrid evaluation framework** designed to evaluate IELTS Writing Task 2 essays using Small Language Models as judges. The system integrates three key components into a cohesive evaluation workflow:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Input IELTS Essay                               │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│ Rubric Grounding   │     │ PANDA Preference   │     │ AES Expert Model   │
│ (Condensed IELTS   │     │ Insight            │     │ (Fine-Tuned        │
│ Descriptors)       │     │ Retrieval          │     │ RoBERTa Multi-Task)│
└────────────────────┘     └────────────────────┘     └────────────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      ▼
                      ┌──────────────────────────────┐
                      │    Structured SLM Prompt     │
                      └──────────────────────────────┘
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │    SLM Judge Evaluation      │
                      │ (Qwen / Llama 3.2 / Mistral) │
                      └──────────────────────────────┘
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │ Final JSON IELTS Band Scores │
                      │ & Diagnostic Evaluation      │
                      └──────────────────────────────┘
```

1. **Rubric-Grounded Prompting**: Official IELTS Writing Band Descriptors are systematically condensed and integrated into prompt templates to provide clear criteria guidance within the limited context windows of SLMs.
2. **Preference Adaptation (PANDA)**: Incorporates the *Preference Adaptation for Enhancing Domain-Specific Abilities of LLMs* (PANDA) method to retrieve domain-specific qualitative explanations ("insights") from an expert insight pool via cosine similarity.
3. **AES Expert Model Assistance**: A specialized **RoBERTa-base** multi-task ordinal regression model serves as an external quantitative expert. It provides preliminary band score estimates within the prompt to ground the SLM judge's final reasoning.

In this hybrid setup, the fine-tuned RoBERTa expert acts as an efficient numerical estimator, while the SLM performs final holistic judgment, synthesis, and score calibration based on all available contextual information.

---

### 1.3 Objectives & Scope

The primary objective of this study is to systematically evaluate how Small Language Models perform under varying levels of structured guidance, prompt engineering, and supervised fine-tuning.

Specifically, this repository investigates:
- **Prompt Engineering Efficacy**: Whether prompting strategies originally designed for large-scale models (e.g., zero-shot, rubric grounding, PANDA preference adaptation, and preliminary score integration) effectively transfer to compact architectures.
- **Model Architectural Comparisons**: Evaluating representative instruction-tuned SLMs across varying parameter scales ($\sim 1.5\text{B}$ to $7\text{B}$ parameters):
  - `Qwen2-1.5B-Instruct`
  - `Qwen2.5-1.5B-Instruct`
  - `Llama-3.2-3B-Instruct`
  - `Mistral-7B-Instruct-v0.2`
- **Inference vs. Fine-Tuning Dynamics**: Comparing non-fine-tuned SLMs relying purely on prompt-based adaptation against SLMs instruction-tuned via Low-Rank Adaptation (LoRA).
- **Scoring Reliability & Calibration**: Analyzing performance using comprehensive evaluation metrics, including Quadratic Weighted Kappa (QWK), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Adjacent Accuracy (within $\pm 0.5$ band points), Pearson/Spearman correlation coefficients, and calibration reliability diagrams.
