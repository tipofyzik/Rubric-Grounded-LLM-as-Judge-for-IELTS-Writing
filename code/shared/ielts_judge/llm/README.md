# Judge model

The judge model takes a prompt and produces the final score estimates. The scoring engine checks its reply before showing the result.

[`Judge.py`](Judge.py) loads the model and sends it the prompt. [`Generation.py`](Generation.py) collects its reply.
