# Local

Run the evaluator in a browser or from the terminal. The app uses Hugging Face and Supabase.

After [local setup](../../docs/guides/Install.md), run these commands from the repository root.

```sh
python Manage.py local --check
python Manage.py local
```

Open the local interface and score an example. [Local UI](http://127.0.0.1:7860)

## CLI

Use two UTF-8 files for the question and essay. Run `python Manage.py local score` to score the built-in example.

```sh
python Manage.py local score --task-file question.txt --essay-file essay.txt
```

The command prints the result as JSON. Add `--output PATH` to save it.

[UI](App.py) · [CLI source](Score.py) · [Score format](../../docs/guides/Scoring.md)
