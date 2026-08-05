# Rubric-Grounded LLM-as-Judge for IELTS Writing: Reliability, Calibration, and Preference Adaptation Using an Automated Essay Scoring Expert Model

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


# 2. Related Literature

This section provides a theoretical background and reviews existing literature relevant to Automated Essay Scoring (AES), Small Language Models (SLMs), evaluation frameworks, preference adaptation, parameter-efficient fine-tuning, and model calibration techniques.

---

## 2.1 Automated Essay Scoring (AES)

Automated Essay Scoring (AES) is defined as the application of natural language processing (NLP) and computational models to evaluate and score written essays automatically [(Uto, 2021)](#references). From a machine learning perspective, the AES task can be formulated in three main paradigms:

1. **Classification Task:** Essay evaluation is modeled as a discrete classification problem where discrete score categories or proficiency levels are predicted [(Alikaniotis et al., 2016)](#references). Models predict a probability distribution over fixed grade bands using cross-entropy loss.
2. **Regression Task:** AES is treated as a continuous regression problem where the model predicts a continuous numeric score reflecting writing quality. Bounded numeric ranges are mapped, and loss functions such as Mean Squared Error (MSE) minimize deviations from ground-truth scores.
3. **Ordinal Regression Task:** Because essay scores are inherently ordinal (possessing a meaningful rank ordering), treating AES as ordinal regression ensures that prediction errors are penalized proportionally to their distance from the target score [(Gutiérrez et al., 2016)](#references).

### Evolution of AES Architectures
* **Traditional AES:** Relied heavily on manual feature engineering (e.g., surface features like essay length, lexical diversity metrics, syntactic trees, and spelling error counts).
* **Neural & Transformer-based AES:** Neural architectures and pretrained Transformer encoders learn complex semantic and structural representations directly from raw text. Although deep neural systems consistently outperform feature-engineered models [(Voss, 2025)](#references), they often lack interpretability and contextual reasoning. Nevertheless, their relative scoring stability makes them effective expert models to assist LLM/SLM judges.

---

## 2.2 Small Language Models (SLMs)

Small Language Models (SLMs) are lightweight language models, typically containing **$\le$ 10 billion parameters**, designed to run efficiently on resource-constrained hardware and low-end devices [(Johnson, 2025)](#references). Compared to large language models (LLMs), SLMs significantly reduce computational requirements, energy consumption, and memory footprint while enabling faster inference latency [(Cao et al., 2026)](#references).

### Core Challenges & Shortcomings of SLMs [(Robert, 2025)](#references)
* **Limited Generalization:** Constrained contextual recall, shorter attention context windows, and smaller internal knowledge bases lead to performance degradation outside their primary training domain.
* **Inconsistency & Hallucinations:** Higher sensitivity to prompt variations, leading to occasional reasoning errors or ungrounded outputs when encountering ambiguous or complex inputs.
* **Reduced Reasoning Complexity:** Difficulty in executing multi-step reasoning tasks that require nuanced contextual understanding.

### Mitigation Strategies [(Senel & Ozmen, 2025)](#references)
To address these limitations, several adaptation techniques are employed:
* **Retrieval-Augmented Generation (RAG):** Enhances grounding by providing external domain-specific context at inference time [(Lewis et al., 2020)](#references).
* **Prompt Engineering:** Designing structured prompts to maximize model performance without modifying weights [(Ye et al., 2023)](#references).
* **Instruction Tuning / Supervised Fine-Tuning (SFT):** Adapting pretrained base models using instruction-response pairs [(Bergmann, n.d.)](#references).
* **Calibration Techniques:** Post-processing or logit adjustments (e.g., temperature scaling) to stabilize model confidence and output determinism.

Because SLMs exhibit pronounced prompt sensitivity, studying their behavior under structured prompting and fine-tuning offers valuable insights into lightweight automated evaluation systems.

---

## 2.3 LLM-as-a-Judge Approach

The **LLM-as-a-Judge** paradigm employs generative language models as automated evaluators to assess open-ended text or natural language outputs produced by other AI models or human authors [(Pandey, 2026)](#references). Instead of relying strictly on supervised task training, this approach leverages in-context learning and zero-shot or few-shot reasoning.

In automated essay evaluation, the task is framed as a prompt engineering problem where the model is provided with explicit evaluation criteria or scoring rubrics [(Li et al., 2024)](#references). This allows the language model to act as a zero-shot scorer through direct inference.

### Key Limitations of LLM Evaluators [(Zheng et al., 2023)](#references)
* **Prompt Sensitivity:** High variance in scores caused by minor alterations in prompt phrasing or layout.
* **Score Instability & Bias:** Propensity toward central score bands, position bias, and overconfidence.
* **Rubric Misalignment:** Difficulty in consistently adhering to detailed, multi-dimensional assessment descriptors without structured guidance.

---

## 2.4 PANDA Approach (Preference Adaptation)

To mitigate reasoning inconsistency and limited contextual recall in language models, [(Liu et al., 2024)](#references) proposed **PANDA** (*Preference Adaptation for Enhancing Domain-Specific Abilities of LLMs*). PANDA aligns language model judgments with domain experts via a two-stage preference adaptation framework:

### 1. Learning Stage
* A domain expert model generates outputs or scores for domain training data.
* The target language model is prompted to explain **why** the expert model prefers specific outputs over alternatives, generating domain-specific insights (an "insight pool").

### 2. Inference Stage
* During inference, relevant insights are retrieved from the insight pool using **cosine similarity** matching.
* Retrieved insights are dynamically injected into the prompt, guiding the model via in-context preference adaptation.

```
+-------------------------------------------------------------------------------+
|                             PANDA Architecture                                |
+-------------------------------------------------------------------------------+
| (a) Learning Stage:                                                           |
|   Training Data --> Domain Expert Model --> Preference Samples                |
|                      --> LLM Insight Generation --> Insight Pool              |
|                                                                               |
| (b) Inference Stage:                                                          |
|   Input Query --> Cosine Similarity Retrieval --> Top Insights                |
|               --> In-Context Prompt Ingestion --> Adapted LLM Judgment        |
+-------------------------------------------------------------------------------+
```

Unlike conventional RAG systems that retrieve static factual data, PANDA retrieves behavioral and evaluative heuristics, enabling better alignment with nuanced domain requirements.

---

## 2.5 LLM Fine-Tuning & Parameter-Efficient Adaptation

Instruction fine-tuning adapts pretrained language models to specific downstream tasks using instruction-output pairs [(Zhang et al., 2023)](#references). Fine-tuning enhances task alignment, instruction adherence, output formatting consistency, and scoring calibration [(Xu et al., 2026; Kang et al., 2026)](#references).

### 2.5.1 Low-Rank Adaptation (LoRA)
To avoid the prohibitive memory and computational costs of full parameter fine-tuning, **Low-Rank Adaptation (LoRA)** freezes the pretrained model weight matrix $W_0 \in \mathbb{R}^{d 	imes k}$ and introduces a low-rank decomposition update $\Delta W$ [(Hu et al., 2021)](#references):

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

where $B \in \mathbb{R}^{d 	imes r}$, $A \in \mathbb{R}^{r 	imes k}$, and rank $r \ll \min(d, k)$.

During training, matrix $W_0$ remains frozen, and gradients are updated exclusively for the low-rank matrices $A$ and $B$. This dramatically lowers memory consumption and storage footprint while maintaining fine-tuning performance.

### 2.5.2 Target Attention Layers
Transformer architectures rely on self-attention mechanisms to compute contextual representations across sequence elements [(Vaswani et al., 2017)](#references). In parameter-efficient fine-tuning setups, LoRA adapters are typically attached to the key projection matrices of the self-attention blocks:
* `q_proj` (Query projection)
* `k_proj` (Key projection)
* `v_proj` (Value projection)
* `o_proj` (Output projection)

Adapting these attention layers allows the model to reweight contextual interactions and refine task-specific scoring logic without altering the underlying core language representation.

---

## 2.6 Calibration Techniques

Model calibration evaluates how accurately a model's predicted confidence scores reflect true empirical correctness. In small language models, calibration is particularly critical, as compact models frequently output overconfident probability distributions despite lower prediction accuracy.

### Temperature Scaling [(Wang, 2025)](#references)
Temperature scaling adjusts raw model output logits $z_i$ prior to applying the Softmax function using a scalar temperature parameter $T$:

$$p_i = rac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

* **$T = 1.0$:** Preserves original logit distribution.
* **$T > 1.0$:** Softens the probability distribution, increasing entropy and encouraging output diversity.
* **$T < 1.0$:** Sharpens the distribution, making outputs more deterministic and confident.

In automated essay scoring and ordinal evaluation tasks, calibration techniques ensure score stability and reduce extreme rating variance across inference runs.


# 3. Methodology

This section details the benchmark framework, dataset preparation, model selections, experimental prompting strategies, instruction-tuning parameters, expert model architecture, and evaluation metrics used to analyze **Small Language Models (SLMs)** as automated essay evaluators for IELTS Task 2.

---

## 3.1 Corpora Choice & Preprocessing

- **Dataset**: IELTS Writing Task 2 Evaluation Dataset ([Nguyen Minh Chi, 2024](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation)), comprising **10,323 IELTS Task 2 essays**.
- **Dataset Split**: A fixed **90% / 10% split** (random seed set to `0`, without shuffling) was implemented for reproducibility:
  - **Training set**: 9,290 essays
  - **Evaluation set**: 1,033 essays
- **Scoring Dimensions**: Evaluation is conducted across four official IELTS Writing criteria on a 0–9 band scale with a 0.5 step size:
  1. **Task Achievement (TA)**: Address of prompt requirements and argument development.
  2. **Coherence and Cohesion (CC)**: Structural organization and paragraph linking.
  3. **Lexical Resource (LR)**: Vocabulary range, precision, and accuracy.
  4. **Grammatical Range and Accuracy (GRA)**: Sentence structure diversity and correctness.
- **Annotations**: Each entry provides individual criteria scores, overall band scores, and expert explanatory feedback.
- **Preprocessing & Imputation**: Missing individual criterion scores were imputed using the overall band score.
- **Data Distribution & Skew**: The dataset exhibits noticeable class imbalance, heavily skewed towards the **6.0–7.5 score range** for Task Achievement and Coherence & Cohesion, and tightly concentrated around **band 6.5** for Lexical Resource and Grammatical Range & Accuracy.

---

## 3.2 Models Choice

Four instruction-tuned Small Language Models (SLMs) spanning from ~1.5B to 7B parameters were selected to evaluate how architectural scale and parameter capacity impact rubric compliance and scoring reliability:

| Model Name | Parameter Count | Selection Rationale / System Role |
| :--- | :---: | :--- |
| **Qwen2-1.5B-Instruct** | ~1.5B | Lower parameter bound for lightweight SLM evaluation |
| **Qwen2.5-1.5B-Instruct** | ~1.5B | Next-generation iteration; evaluates instruction-following improvements |
| **Llama-3.2-3B-Instruct** | ~3B | Mid-sized model optimized for efficient text generation and reasoning |
| **Mistral-7B-Instruct-v0.2** | ~7B | Upper parameter bound of the experimental benchmark |

All models were required to be instruction-tuned to handle multi-component inputs (rubrics, PANDA preference insights, preliminary expert scores) and output strictly structured JSON responses.

---

## 3.3 Experimental Setup & Architecture

### Hardware Infrastructure
- **Supervised Fine-Tuning (SFT)**: Executed on the Hugging Face platform using an **NVIDIA L4 (24 GB VRAM)** GPU.
- **Inference & Benchmarking**: Carried out on Kaggle utilizing **2× NVIDIA T4 (16 GB VRAM each)** GPUs, enabling multi-GPU execution.

### 3.3.1 Prompting Strategies
Six main prompting configurations were engineered to investigate how varying levels of structured context affect scoring accuracy:

1. **Zero-shot Prompting (`p1`)**: Baseline configuration containing core instructions, score boundaries (0–9 scale, 0.5 step), and mandatory JSON output schema without rubric descriptions or contextual examples.
2. **LLM-as-Judge / Rubric-based Prompting (`p2`)**: Integrates shortened, essentialized versions of official IELTS Writing Band Descriptors for all four criteria (condensed to prevent context-window truncation while maintaining rubric fidelity).
3. **PANDA-based Prompting (`p3`)**: Adapts the **Preference Adaptation for Enhancing Domain-Specific Abilities (PANDA)** framework:
   - *Insight Pool Generation*: Ground-truth expert band scores serve as the expert model proxy. For each training essay, the SLM generates structured explanations ("insights") explaining why the true band was assigned over neighboring scores within a range of $\pm 1.0$ score points.
   - *Insight Retrieval*: During inference, cosine similarity vector search retrieves the single most relevant insight (`top_k = 1`) from the pool to guide the model via in-context learning.
4. **Prompting with Preliminary Scores (`p5`)**: Custom hybrid strategy where pre-evaluated band predictions generated by a fine-tuned RoBERTa AES expert model are provided in the prompt as external expert reference points.
5. **Prompt Mixtures (`p4`, `p6`)**:
   - **Rubric + PANDA (`p4`)**: Combines condensed IELTS criteria rubrics with retrieved PANDA preference insights.
   - **Rubric + PANDA + Preliminary Scores (`p6`)**: Full multi-source context combining rubrics, PANDA insights, and RoBERTa preliminary scores.

### 3.3.2 Instruction Tuning Setup (LoRA SFT)
Supervised fine-tuning of SLMs was conducted on the 9,290 training essays using the standardized Zero-Shot base prompt format:
- **Parameter-Efficient Adaptation**: Low-Rank Adaptation (LoRA) applied to attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **LoRA Configurations**: Rank $r = 16$, $\alpha = 32$, Dropout = $0.05$.
- **Hyperparameters**: 1 Epoch, Learning Rate $3 \times 10^{-5}$, Optimizer `adamw_torch`, Linear LR Scheduler, Batch Size 2 with Gradient Accumulation Steps 4 (effective batch size 8), `model_max_length = 1536` tokens.
- **Adapter Merging**: `merge_adapter = True` to permanently integrate adapter weights into base weights after training, ensuring zero inference latency overhead.

### 3.3.3 AES Expert Model Setup (RoBERTa Multi-Task Ordinal Regression)
To supply independent preliminary scores for prompt strategy `p5` and `p6`:
- **Base Architecture**: Pretrained `RoBERTa-base` encoder.
- **Model Design**: RoBERTa Encoder $\rightarrow$ Masked Mean Pooling Layer $\rightarrow$ 4 Task-Specific Ordinal Regression Heads (one head per IELTS criterion).
- **Formulation**: Multi-task ordinal regression predicting $K-1$ binary thresholds ($P(y > k)$) per criterion.
- **Training Parameters**: Binary Cross-Entropy with Logits Loss, AdamW optimizer ($	ext{lr} = 2 \times 10^{-5}$), Cosine Annealing LR Scheduler, Batch Size 16, 15 Epochs, Max Length 512 tokens.

---

## 3.4 Evaluation Metrics

The framework evaluates model performance across seven quantitative metrics covering agreement, error distribution, rank consistency, and probability calibration:

1. **Quadratic Weighted Kappa (QWK)**: Measures inter-rater agreement on ordinal scales, quadratically penalizing larger score discrepancies:
   $$\text{QWK} = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}$$
2. **Mean Absolute Error (MAE)**: Measures average absolute prediction error magnitude:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$
3. **Root Mean Square Error (RMSE)**: Measures standard deviation of residuals, heavily penalizing large outliers:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$
4. **Adjacent Accuracy ($A_{\text{adj}}$ at $\pm 0.5$)**: Proportion of predictions falling within $\le 0.5$ band points of ground truth:
   $$A_{\text{adj}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(|\hat{y}_i - y_i| \le 0.5)$$
5. **Pearson Correlation Coefficient ($r_{x,y}$)**: Measures linear correlation between predicted and ground-truth scores.
6. **Spearman's Rank Correlation ($r_s$)**: Evaluates monotonic rank-order alignment between predictions and ground-truth grades.
7. **Calibration Curves (Reliability Diagrams)**: Evaluates structural calibration and system bias across the full band spectrum (0–9).

---

## 3.5 Comparative Systems Matrix

Evaluated configurations across non-fine-tuned and fine-tuned model variants:

| Configuration Code | Prompt Strategy Name | Non Fine-Tuned SLMs | Fine-Tuned SLMs |
| :---: | :--- | :---: | :---: |
| **`p1`** | **Zero-shot** | Baseline | Evaluated |
| **`p2`** | **Rubric-based** | Evaluated | Evaluated |
| **`p3`** | **PANDA-based** | Evaluated | N/A* |
| **`p4`** | **Rubric + PANDA-based** | Evaluated | Evaluated |
| **`p5`** | **PANDA-based + Preliminary Scores** | Evaluated | Evaluated |
| **`p6`** | **Rubric + PANDA-based + Preliminary Scores** | Evaluated | N/A* |

*\*Note: Fine-tuned SLMs strictly produce structured JSON band outputs post-SFT, rendering single-step insight generation (`p3`) and multi-prompt setups (`p6`) non-applicable.*

---

## 3.6 SLM Evaluation Approach & Protocol

- **Repeated Inference**: Due to non-deterministic sampling in language generation, each experimental setup was executed across **5 independent runs**.
- **Statistical Reliability**: Reported metrics are averaged over all 5 runs, complete with standard deviation ($\pm \sigma$) bounds.
- **Inference Parameters**: `max_new_tokens = 200`, `temperature = 0.2`, `do_sample = True`.
- **Insight Selection**: For PANDA configurations, only the top relevant insight (`top_k = 1`) per essay was retrieved via cosine similarity vector search.
