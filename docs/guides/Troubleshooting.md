# Troubleshooting


Input validation runs before expensive inference where possible. The service distinguishes invalid input, busy admission, AES/encoding errors, empty retrieval, database failures, context-budget failures, judge quota/timeouts, malformed output and metadata mismatch. Failures leave scores empty; UI callbacks release the submission form.

Gradio accepts one active scoring request and up to four waiting requests. The scoring service also serializes direct calls. Processing and decoding share a deadline that starts at admission. Native calls may finish after the deadline. The parent process owns the CPU worker connection and shutdown.

The read-only `panda_match` request can retry one completed transient HTTP 500/502/503/504 response under the remaining budget. Writes/imports, authentication errors, database cancellation and client transport timeout are not automatically retried. This improves recovery for some failures; the original intermittent Supabase root cause has not been established.

Application persistence contains component versions, hashes and safe timings rather than submitted essays or raw judge completions. Tracing is disabled. The CPU child clears its last payload after a command. This describes application behavior; it is not a claim about every infrastructure provider's internal retention.

Report the error code, failed operation, available HTTP or database code, and deployment revision. Keep credentials and private text out of public issues.


[Docs](../README.md)
