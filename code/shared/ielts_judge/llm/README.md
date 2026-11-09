# Judge

The judge model takes a prompt and produces the final score estimates. The scoring engine checks its reply before showing the result.

[`Judge.py`](Judge.py) loads the model and sends it the prompt. [`Generation.py`](Generation.py) collects its reply.

[`ModelRegistry.py`](../ModelRegistry.py) supplies the pinned model profile and token budget. [`Scores.py`](../Scores.py) accepts exactly four criterion fields with integer scores from 0 to 9. See the [scoring flow](../../../../docs/guides/Scoring.md) for the output format and error behavior.
