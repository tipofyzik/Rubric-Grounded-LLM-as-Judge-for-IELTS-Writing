# Local app

Run the Gradio app or score essays from the terminal. Your machine makes the first score estimate and prepares the search query. Supabase finds a similar essay, and a Hugging Face Space runs Qwen.

## Run

After [setting up the local app](../../docs/guides/04Installation.md), run this from the repository root and open [localhost:7860](http://127.0.0.1:7860).

```sh
python Manage.py local
```

To score a question and essay from two text files, run this instead.

```sh
python Manage.py local score --task-file question.txt --essay-file essay.txt
```

## Code

- [`App.py`](App.py) starts Gradio.
- [`Score.py`](Score.py) reads the text files and prints the scores.
- [`Runtime.py`](Runtime.py) loads the local models and connects to the hosted services.
