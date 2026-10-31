"""Supabase HTTPS/RPC transport, idempotent corpus ingestion and typed retrieval."""

import math
import re
import time
from urllib.parse import urlsplit

import requests
import yaml

from .criteria import PUBLIC_CRITERION_IDS
from .data.preparation import text_hash
from .panda.artifacts import canonical, digest, insights, parse_json, positive_int, scores, sha256
from .panda.corpus import Corpus, vector
from .prompts import ReferenceContext
from .paths import PROJECT_ROOT


def retrieval_k():
    """Read the shared inference setting at call time, without loading models."""
    config = yaml.safe_load((PROJECT_ROOT / "code/config.yaml").read_text(encoding="utf-8"))
    value = config["retrieval"]["k"]
    if not 1 <= positive_int(value, "retrieval.k") <= 20:
        raise ValueError("retrieval.k must be 1-20")
    return value


class RetrievalError(RuntimeError):
    """RPC transport, permissions, release or response failure; never an empty retrieval."""

    def __init__(self, message, code="retrieval_failure", *, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = {}
        supplied = diagnostics or {}
        for name in ("http_status", "attempts"):
            value = supplied.get(name)
            if type(value) is int and ((name == "http_status" and 100 <= value <= 599)
                                      or (name == "attempts" and 0 <= value <= 2)):
                self.diagnostics[name] = value
        database_code = supplied.get("database_code")
        if isinstance(database_code, str) and re.fullmatch(r"(?:[A-Z0-9]{5}|PGRST[0-9]{3})", database_code):
            self.diagnostics["database_code"] = database_code


class EncodingError(RuntimeError):
    """The query encoder failed before a database request was made."""


class SupabaseRPC:
    def __init__(self, url, key, *, timeout=30):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"))) or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Supabase URL must be an HTTPS origin (HTTP loopback is allowed for local tests)")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Supabase API key is required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout must be finite and positive")
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout

    def call(self, function, parameters, *, timeout=None):
        if function not in ("panda_begin", "panda_import", "panda_finalize", "panda_match"):
            raise ValueError("Unknown PANDA RPC function")
        headers = {"apikey": self.key, "Content-Type": "application/json"}
        # New sb_publishable_/sb_secret_ keys are API keys, not Bearer JWTs.
        if not self.key.startswith("sb_"):
            headers["Authorization"] = "Bearer " + self.key
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("RPC timeout must be finite and positive")
        budget = min(self.timeout, timeout) if timeout is not None else self.timeout
        deadline = time.monotonic() + budget
        payload = canonical(parameters).encode("utf-8")
        # Only the read RPC may recover from a completed transient HTTP failure.
        attempts = 2 if function == "panda_match" else 1
        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RetrievalError("Supabase RPC time budget expired", "retrieval_timeout",
                                     diagnostics={"attempts": attempt - 1})
            try:
                response = requests.post(self.url + "/rest/v1/rpc/" + function, headers=headers,
                                         data=payload, timeout=remaining, allow_redirects=False)
            except requests.Timeout as error:
                raise RetrievalError("Supabase RPC network/timeout failure", "retrieval_timeout",
                                     diagnostics={"attempts": attempt}) from error
            except requests.RequestException as error:
                raise RetrievalError("Supabase RPC network/timeout failure",
                                     diagnostics={"attempts": attempt}) from error
            try:
                diagnostics: dict[str, object] = {"http_status": response.status_code, "attempts": attempt}
                if response.status_code == 200:
                    try:
                        return parse_json(response.text)
                    except ValueError as error:
                        raise RetrievalError("Supabase RPC returned invalid JSON", "retrieval_invalid_response",
                                             diagnostics=diagnostics) from error
                try:
                    body = parse_json(response.text)
                except ValueError:
                    body = {}
                database_code = body.get("code") if isinstance(body, dict) else None
                diagnostics["database_code"] = database_code
                message = body.get("message", "") if isinstance(body, dict) else ""
                code = "retrieval_failure"
                if database_code == "57014":
                    code = "retrieval_timeout" if isinstance(message, str) and "statement timeout" in message.lower() else "retrieval_cancelled"
                elif response.status_code == 429:
                    code = "retrieval_quota"
                if (response.status_code in (500, 502, 503, 504) and code != "retrieval_cancelled"
                        and attempt < attempts and time.monotonic() < deadline):
                    continue
                raise RetrievalError(f"Supabase RPC {function} failed: HTTP {response.status_code}", code,
                                     diagnostics=diagnostics)
            finally:
                response.close()


def import_corpus(corpus, rpc, *, batch_size=50):
    """The database owns transactional batches, immutable rows and readiness; reruns are idempotent."""
    if not isinstance(corpus, Corpus):
        corpus = Corpus(corpus)
    if not 1 <= positive_int(batch_size, "batch_size") <= 100:
        raise ValueError("Import batch_size must be 1-100")
    manifest = corpus.manifest
    result = rpc.call("panda_begin", {"p_manifest": manifest})
    if not isinstance(result, dict) or result.get("corpus_id") != manifest["corpus_id"]:
        raise RetrievalError("Unexpected corpus registration response")
    batch = []
    imported = 0
    for line, _ in corpus.records():
        batch.append(line)
        if len(batch) == batch_size:
            _import_batch(rpc, manifest["corpus_id"], batch)
            imported += len(batch)
            batch = []
    if batch:
        _import_batch(rpc, manifest["corpus_id"], batch)
        imported += len(batch)
    result = rpc.call("panda_finalize", {"p_corpus_id": manifest["corpus_id"]})
    expected = {"corpus_id": manifest["corpus_id"], "state": "ready", "count": manifest["count"],
                "records_digest": manifest["records_digest"]}
    if result != expected or imported != manifest["count"]:
        raise RetrievalError("Database corpus verification differs from local artifacts")
    return result


def _import_batch(rpc, corpus_id, batch):
    result = rpc.call("panda_import", {"p_corpus_id": corpus_id, "p_records": batch})
    if result != {"count": len(batch)}:
        raise RetrievalError("Database did not acknowledge the full import batch")


class ReferenceRetriever:
    """Does not log or persist submissions; references expose only source dataset labels."""

    def __init__(self, rpc, manifest, encoder):
        digest(manifest["corpus_id"], "corpus_id")
        if manifest["embedding_profile_sha256"] != sha256(canonical(manifest["embedding"])) or encoder.profile != manifest["embedding"]:
            raise ValueError("Encoder/corpus profile mismatch")
        self.rpc, self.manifest, self.encoder = rpc, manifest, encoder

    def retrieve(self, essay, *, essay_id=None, k=None, threshold=None, timeout=None):
        if k is None:
            k = retrieval_k()
        if not isinstance(essay, str) or not essay.strip():
            raise ValueError("A nonempty essay is required")
        if essay_id is not None and (not isinstance(essay_id, str) or not essay_id.strip()):
            raise ValueError("essay_id must be nonempty or None")
        if not 1 <= positive_int(k, "k") <= 20:
            raise ValueError("k must be 1-20")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                      or not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise ValueError("Similarity threshold must be finite and within -1..1")
        started = time.monotonic()
        try:
            values, lengths = self.encoder.encode([essay])
            if len(values) != 1 or len(lengths) != 1:
                raise ValueError("Query encoder alignment failure")
            query_vector = vector(values[0]).tolist()
            positive_int(lengths[0], "query token length")
        except Exception as error:
            raise EncodingError("Query encoder failed") from error
        encoding_seconds = time.monotonic() - started
        query_hash = text_hash(essay)
        options = {}
        if timeout is not None:
            remaining = timeout - encoding_seconds
            if remaining <= 0:
                raise RetrievalError("Retrieval deadline expired after encoding", "retrieval_timeout")
            options["timeout"] = remaining
        rpc_started = time.monotonic()
        result = self.rpc.call("panda_match", {"p_corpus_id": self.manifest["corpus_id"],
            "p_embedding_profile_sha256": self.manifest["embedding_profile_sha256"],
            "p_query_embedding": query_vector, "p_query_hash": query_hash,
            "p_query_id": essay_id, "p_match_count": k, "p_threshold": threshold}, **options)
        rpc_seconds = time.monotonic() - rpc_started
        try:
            if not isinstance(result, dict) or result.get("corpus_id") != self.manifest["corpus_id"] or result.get("embedding_profile_sha256") != self.manifest["embedding_profile_sha256"] or result.get("count") != self.manifest["count"]:
                raise ValueError("RPC corpus/version mismatch")
            found = result["references"]
            if not isinstance(found, list) or len(found) > k:
                raise ValueError("RPC returned an invalid reference list")
            references, seen, hashes, similarities = [], set(), set(), []
            for row in found:
                identity, reference_hash = row["essay_id"], row["essay_hash"]
                if not isinstance(identity, str) or not identity.strip() or identity in seen or identity == essay_id:
                    raise ValueError("RPC returned duplicate/self essay ID")
                digest(reference_hash, "essay_hash")
                if reference_hash in hashes or reference_hash == query_hash or text_hash(row["essay"]) != reference_hash:
                    raise ValueError("RPC returned duplicate/self/incorrect essay hash")
                if not isinstance(row["task_prompt"], str) or not row["task_prompt"].strip():
                    raise ValueError("Reference task is missing")
                labels = scores(row["scores"])
                if self.manifest["schema_version"] == "panda-corpus-v2":
                    explanation = row["insights"]
                    if not isinstance(explanation, str):
                        raise ValueError("Reference insights must be the original completion text")
                else:
                    explanation = canonical(insights(row["insights"], row["essay"], labels))
                similarity = row["similarity"]
                if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not math.isfinite(similarity) or not -1.0001 <= similarity <= 1.0001 or (threshold is not None and similarity < threshold - 1e-6):
                    raise ValueError("Invalid similarity")
                if similarities and similarity > similarities[-1] + 1e-6:
                    raise ValueError("RPC results are not ordered by similarity")
                seen.add(identity)
                hashes.add(reference_hash)
                similarities.append(similarity)
                references.append(ReferenceContext(identity, row["task_prompt"], labels, explanation, row["essay"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RetrievalError(f"Invalid reference response: {error}") from error
        return {"status": "ok" if references else "retrieval_empty", "references": tuple(references),
                "corpus_id": self.manifest["corpus_id"], "similarities": similarities,
                "query_original_tokens": lengths[0], "query_used_tokens": min(lengths[0], self.encoder.profile["max_length"]),
                "query_truncated": lengths[0] > self.encoder.profile["max_length"],
                "encoding_seconds": encoding_seconds, "rpc_seconds": rpc_seconds}
