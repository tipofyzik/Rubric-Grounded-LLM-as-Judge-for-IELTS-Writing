# Local app

Use this folder to run the evaluator on your machine, either in a browser or from the terminal. The app makes its first estimate locally and uses online services to find an example and run Qwen.

After [local setup](../../docs/guides/04Installation.md), start the app from the repository root.

```sh
python Manage.py local
```

Open [localhost:7860](http://127.0.0.1:7860). To score a question and essay from two text files, use the terminal command.

```sh
python Manage.py local score --task-file question.txt --essay-file essay.txt
```

[`App.py`](App.py) starts the browser app. [`Score.py`](Score.py) runs the terminal command. Both use [`Runtime.py`](Runtime.py) to load the local models and connect to the hosted services.
