# Judge model

[`Judge.py`](Judge.py) loads the model, checks the prompt length and asks the model to score an essay.

- [`Generation.py`](Generation.py) generates the reply.
- [`JsonEosBlocker.py`](JsonEosBlocker.py) blocks the model's end token while the score object is incomplete.
