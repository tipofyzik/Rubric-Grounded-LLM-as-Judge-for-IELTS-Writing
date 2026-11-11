# IELTS Writing Evaluator

Score IELTS Writing Task 2 essays with small language models (SLMs) guided by the IELTS rubric, PANDA preference adaptation, and preliminary scores from an automated essay scoring (AES) model. The system retrieves PANDA notes from similar training essays to guide the judge, including explanations of their assigned scores and differences from neighboring bands.

The demo uses Qwen2.5-1.5B-Instruct and returns four experimental criterion scores (integers from 0 to 9). Other compatible models can be added through the model registry. The interface supports English, Russian and German and includes three sample essays.

[Demo](https://tipofyzik-ielts-writing-evaluator.hf.space) · [Install](#install) · [Architecture](docs/guides/Architecture.md) · [Research](code/offline_research/README.md) · [Docs](docs/README.md)

The question and essay are sent to Hugging Face for scoring. [Data handling](docs/guides/Troubleshooting.md)

[Demo video](docs/media/Demo.webm)

## Install

You need Python 3.11, access to a Hugging Face Space running the judge, a Supabase database with the matching corpus, and a compatible AES and PANDA release. The author's artifact repository is private. [Setup guide](docs/guides/Install.md)

1. **Install dependencies.** Run from the repository root.

   ```sh
   python Manage.py install local
   ```

2. **Configure access.** Copy [`.env.example`](.env.example) to `.env` and fill in your settings. [Settings](docs/guides/Install.md#configure-access)

3. **Restore assets.** Run one command for your release source.

   ```sh
   # Hugging Face
   python Manage.py setup --download-assets --warm-cache

   # Local directory
   python Manage.py setup --artifact-source "PATH_TO_RELEASE" --warm-cache
   ```

4. **Run the app.** Check the setup, start the app and score a sample essay. [Local UI](http://127.0.0.1:7860) · [CLI](code/local_backup/README.md)

   ```sh
   python Manage.py local --check
   python Manage.py local
   ```

## Method

The evaluation keeps the Qwen model fixed and varies the context supplied with the question and essay.

- **Rubric.** Condensed IELTS Writing Task 2 band descriptors.
- **PANDA.** Scores and generated notes from a similar training essay. The notes use the dataset's assigned scores and explain the evidence for each band.
- **AES.** Preliminary scores from a RoBERTa-based model that reads only the essay.

## Architecture

The hosted demo and local app share the same scoring engine. Both look up reference examples in Supabase and use the judge on Hugging Face. The local app runs the preliminary scorer and builds the search vector on your machine.

[Scoring](docs/guides/Scoring.md) · [Code](code/README.md)

## Data

The benchmark uses `ielts-v3`, a filtered and deduplicated dataset built from [Chillies](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation) and [Writing9](https://huggingface.co/datasets/ndtran0101/writing9-ielts-essays). Chillies supplies low-band examples.

| Source | Train | Test |
| --- | ---: | ---: |
| Chillies | 263 | 22 |
| Writing9 | 7,973 | 1,613 |
| **Total** | **8,236** | **1,635** |

AES training and PANDA retrieval use the train split. The benchmark compares predictions with the source labels on the test split. [Data preparation](docs/guides/PrepareData.md) · [Evaluation protocol](docs/model_evaluation_protocol.md#data-and-leakage-boundaries)

## Results

On Kaggle, we ran each of the six prompt variants once on the full test split with Qwen2.5-1.5B-Instruct, the model used in the demo. The metrics are averaged across the four criteria.

| Scoring setup | QWK ↑ | RMSE ↓ | MAE ↓ |
| --- | ---: | ---: | ---: |
| Qwen with the base prompt | 0.027 | 2.127 | 1.724 |
| Qwen with rubric, PANDA notes and preliminary scores | 0.606 | 1.393 | 0.981 |
| Essay scoring model alone | 0.607 | 1.391 | 0.980 |

Quadratic weighted kappa (QWK) measures agreement on the ordered band scale. Root mean squared error (RMSE) gives more weight to large errors. Mean absolute error (MAE) and RMSE are measured in band points.

The full prompt improved agreement over the base Qwen prompt. The standalone essay scorer led on all three aggregate metrics, and the full prompt returned the same score as that scorer in 6,536 of 6,540 cases. [Full results](docs/model_evaluation_report.md) · [Evaluation protocol](docs/model_evaluation_protocol.md)

## References

- Liu, A., et al. (2024). *PANDA: Preference Adaptation for Enhancing Domain-Specific Abilities of LLMs*. Findings of ACL 2024, pp. 10960–10977. [ACL Anthology](https://aclanthology.org/2024.findings-acl.651/)
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.05685)
- IELTS (2023). *Writing Band Descriptors*, Task 2. [Official descriptors](https://ielts.org/cdn/ielts-guides/ielts-writing-band-descriptors.pdf#page=7)
