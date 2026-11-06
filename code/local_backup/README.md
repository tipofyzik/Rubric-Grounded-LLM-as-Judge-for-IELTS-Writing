# Local Backup

Run AES/BGE on local CPU, retrieve from Supabase and call the matching hosted judge. App.py and Score.py share Runtime.py and the common ScoringService. This backup requires network services.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | IELTS local_backup component. |
| [App.py](App.py) | Launch the local IELTS demo with the accepted hybrid runtime. |
| [requirements.txt](requirements.txt) | Configuration, dependency or supporting source input. |
| [Runtime.py](Runtime.py) | Local hybrid startup and hosted judge transport, shared by CLI and UI. |
| [Score.py](Score.py) | Run the accepted scoring service once, without saving submission text. |

[Documentation catalogue](../../docs/README.md)
