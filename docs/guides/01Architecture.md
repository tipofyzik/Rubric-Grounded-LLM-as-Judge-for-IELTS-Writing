# Architecture and ownership


The three components depend on `shared`; shared modules never import a component. This keeps request behavior consistent while deployment, local transport and research remain independently owned.

```mermaid
flowchart LR
  Visitor[Browser] --> Web[HF Gradio web process]
  Web --> Service[Shared ScoringService]
  Service --> CPU[Spawned CPU AES and BGE]
  Service --> Database[Supabase HTTPS RPC]
  Service --> GPU[Qwen ZeroGPU generation]
  Local[Local UI or CLI] --> LocalService[Shared ScoringService]
  LocalService --> LocalCPU[Local CPU AES and BGE]
  LocalService --> Database
  LocalService --> GPU
  Research[Deferred offline tools] --> Shared[Shared models and contracts]
```

`Manage.py` selects an interpreter and command; `RuntimeAssets.py` restores compatible runtime inputs. Neither entry point trains models during normal startup. The hosted CPU process isolates AES/BGE native inference from ZeroGPU's CUDA worker lifecycle. The scoring service admits one synchronous request and owns its deadline. Gradio owns the bounded waiting queue and session state.

Read the [shared block](../../code/shared/README.md), [public block](../../code/public_demo/README.md) and [local block](../../code/local_backup/README.md) for their actual files and interfaces.


[Documentation catalogue](../README.md)
