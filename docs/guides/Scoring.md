# Scoring


1. UI snapshots the model key, English Task/Essay and selected display language. Editing inputs clears old results.
2. ScoringService validates the enabled model, nonempty inputs, character limit and minimal rendered prompt budget.
3. The accepted AES bundle predicts four essay-only priors; lengths/truncation are recorded.
4. BGE encodes the essay and Supabase `panda_match` retrieves compatible training references. Query ID/hash exclusion prevents self-reference.
5. Prompts assembles the common instruction, Task/Essay and enabled rubric/reference/AES blocks. References enter as scores and original insights; their full Task/Essay and IDs are not prompt context.
6. Complete references may be removed from the end to fit the budget. Task/Essay are not silently cut. Empty retrieval is a warning; database failure is an error.
7. The judge makes one bounded generation attempt. The shared parser accepts exactly four public integer score fields, including mathematically integral `6.0` and one whole JSON Markdown fence.
8. UI shows scores or the relevant failure and unlocks the form. Application logs do not persist the submission.

The public keys are `task_response`, `coherence_and_cohesion`, `lexical_resource`, and `grammatical_range_and_accuracy`. Fractions, booleans, missing/extra/duplicate fields and surrounding prose are rejected without rounding, filling or repair.

The final judge input limit is 3,840 tokens with up to 256 output tokens. The combined character limit is 40,000. The service's 120-second deadline excludes time waiting in the UI queue. Native CPU/GPU calls already in flight are not guaranteed to be preemptible.


[Docs](../README.md)
