# Failures, resource ownership and privacy


Input validation runs before expensive inference where possible. The service distinguishes invalid input, busy admission, AES/encoding errors, empty retrieval, database failures, context-budget failures, judge quota/timeouts, malformed output and metadata mismatch. Failures leave scores empty; UI callbacks release the submission form.

Gradio limits scoring concurrency to one and waiting requests to four. The shared service also rejects concurrent direct use of its loaded runtime. Each admitted request carries its original monotonic deadline; stages and decoding check it. In-flight native kernels may finish after a deadline. CPU worker commands use one synchronous connection and are serialized by service admission; shutdown belongs to the parent process.

The read-only `panda_match` request can retry one completed transient HTTP 500/502/503/504 response under the remaining budget. Writes/imports, authentication errors, database cancellation and client transport timeout are not automatically retried. This improves recovery for some failures; the original intermittent Supabase root cause has not been established.

Application persistence contains component versions, hashes and safe timings rather than submitted essays or raw judge completions. Tracing is disabled. The CPU child clears its last payload after a command. This describes application behavior; it is not a claim about every infrastructure provider's internal retention.

Diagnose with the error code, stage, available HTTP/database code and deployment revision. Do not paste tokens, submitted text or private corpus records into public issues.


[Documentation catalogue](../README.md)
