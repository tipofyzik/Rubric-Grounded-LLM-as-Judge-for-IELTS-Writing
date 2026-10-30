"""Strict, versioned PANDA artifacts; no model loading or network at import."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
from numbers import Real

from ..criteria import IELTS_CRITERIA, LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid)


def unwrap_completion(text):
    """Remove exactly one complete outer Markdown fence; never extract/repair JSON fragments."""
    wrapped = re.fullmatch(r"\s*```(?:json|text|markdown)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.S | re.I)
    return wrapped[1] if wrapped else text


def parse_completion(text):
    return parse_json(unwrap_completion(text))


def is_placeholder(text):
    return isinstance(text, str) and re.fullmatch(
        r"(?:not[\s_/-]*applicable|n[\s_/-]*a|none|null)[.!]?", text.strip(), re.I
    ) is not None


def adjacent_band_labels(band: int) -> dict[str, str]:
    """Name explanation directions using the supplied integer band and its real neighbors."""
    if type(band) is not int or not 0 <= band <= 9:
        raise ValueError("Adjacent-band text requires an integer score in 0-9")
    return {
        "evidence": "Evidence",
        "why_not_lower": f"Strength supporting band {band} rather than band {band - 1}" if band > 0
            else "Strength supporting band 0 (no lower band)",
        "why_not_higher": f"Weakness preventing band {band + 1}" if band < 9
            else "Weakness preventing a higher band (9 is maximum)",
    }


def parse_legacy_completion(text, labels, *, explicit_bands=False, allow_partial=False):
    """Parse text insights; allow_partial retains unvalidated diagnostic fields without fixing captions."""
    names = {}
    for name, definition in IELTS_CRITERIA.items():
        public = LEGACY_TO_PUBLIC[name]
        for alias in (public.replace("_", " "), definition["display_name"], definition["source_name"]):
            names[re.sub(r"[^a-z]", "", alias.lower())] = public
    result, criterion, field = {}, None, None
    for line in unwrap_completion(text).splitlines():
        line = line.strip()
        if not line:
            continue
        heading_line = re.sub(r"\s*\(Band\s+\d+\)\s*$", "", line.replace("**", ""), flags=re.I) if allow_partial else line
        heading = names.get(re.sub(r"[^a-z]", "", heading_line.lower()))
        if heading:
            if heading in result:
                if allow_partial:
                    criterion, field = None, None
                    continue
                raise ValueError("Duplicate legacy criterion")
            criterion, field = heading, None
            result[criterion] = {}
            continue
        marker = r"(?:[-*]\s*)?" if allow_partial else r"[-*]\s*"
        pattern = marker + (r"(Evidence|Strength supporting band [^:]+|Weakness preventing [^:]+)\s*:\s*(.*)"
                   if explicit_bands else
                   r"(Evidence|Why not higher|Why not lower)(?:\s*\([^)]*\))?\s*:\s*(.*)")
        bullet = re.fullmatch(pattern, line.replace("**", ""), re.I)
        if bullet and criterion:
            if explicit_bands:
                expected = {caption.lower(): key for key, caption in adjacent_band_labels(labels[criterion]).items()}
                field = expected.get(bullet[1].strip().lower())
                if field is None:
                    if not allow_partial:
                        raise ValueError(f"Explanation band numbers do not match source score: {criterion}")
                    field = "why_not_lower" if bullet[1].lower().startswith("strength") else "why_not_higher"
            else:
                field = bullet[1].lower().replace(" ", "_")
            if field in result[criterion]:
                if allow_partial:
                    field = None
                    continue
                raise ValueError("Duplicate legacy field")
            result[criterion][field] = bullet[2]
        elif criterion and field and not line.startswith(("-", "*", "#")):
            result[criterion][field] += " " + line
        elif allow_partial:
            if criterion and field and re.match(r"[-*]\s+\S", line):
                result[criterion][field] += (" " if result[criterion][field] else "") + re.sub(r"^[-*]\s+", "", line)
        else:
            raise ValueError("Expected a legacy criterion heading and three labelled bullets")
    # The old prompt did not prescribe the literal spelling at nonexistent adjacent bands.
    for criterion, item in result.items():
        for field, boundary in (("why_not_higher", 9), ("why_not_lower", 0)):
            if labels[criterion] == boundary and is_placeholder(item.get(field)):
                item[field] = "not_applicable"
    return result


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Single-writer atomic replacement; an interrupted write leaves the prior file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def digest(value, name="SHA256"):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name}")
    return value


def revision(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Model revision must be a pinned 40-character HF commit")
    return value


def scores(value):
    if not isinstance(value, dict) or set(value) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Exactly four public criterion scores are required")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           or not 0 <= float(v) <= 9 or float(v) % 1 for v in value.values()):
        raise ValueError("Scores must be finite integers in 0-9, without bool")
    return {key: int(value[key]) for key in PUBLIC_CRITERION_IDS}


def insights(value, essay, labels, *, require_quote=True):
    labels = scores(labels)
    if not isinstance(value, dict) or set(value) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Insights must cover exactly four criteria")
    for criterion, item in value.items():
        if not isinstance(item, dict) or set(item) != {"evidence", "why_not_higher", "why_not_lower"}:
            raise ValueError("Each insight needs evidence, why_not_higher and why_not_lower")
        if any(not isinstance(v, str) or not v.strip() for v in item.values()):
            raise ValueError("Insight fields must be nonempty strings")
        if is_placeholder(item["evidence"]):
            raise ValueError(f"Evidence cannot be a placeholder: {criterion}")
        if require_quote and item["evidence"] not in essay:
            raise ValueError(f"Insight evidence is not an exact essay substring: {criterion}")
        for field, boundary in (("why_not_higher", 9), ("why_not_lower", 0)):
            if (labels[criterion] == boundary and item[field] != "not_applicable") or (
                labels[criterion] != boundary and is_placeholder(item[field])
            ):
                raise ValueError(f"Invalid boundary explanation for {criterion}/{field}")
    return value
