# Local

Use this folder to run the evaluator on your machine, either in a browser or from the terminal. The app makes its first estimate locally and uses online services to find an example and run Qwen.

After [local setup](../../docs/guides/Install.md), run these commands from the repository root.

```sh
python Manage.py local --check
python Manage.py local
```

The first command reports `"status": "ready"` when the required files, Python packages and hosted judge endpoint are available. To verify retrieval and scoring, submit an example in the browser or use the terminal command. [Local interface](http://127.0.0.1:7860)

Use two UTF-8 text files containing the English question and essay. Omit both file options to score the first built-in example.

```sh
python Manage.py local score --task-file question.txt --essay-file essay.txt
```

The command prints JSON. On success, it exits with code 0 and includes fields like these.

```json
{
  "status": "ok",
  "scores": {
    "task_response": 6,
    "coherence_and_cohesion": 6,
    "lexical_resource": 8,
    "grammatical_range_and_accuracy": 8
  }
}
```

Use `--output PATH` to also save the full result, including warnings and diagnostics, to a new JSON file.

[`App.py`](App.py) starts the browser app. [`Score.py`](Score.py) runs the terminal command. Both use [`Runtime.py`](Runtime.py) to load the local models and connect to the hosted services.
