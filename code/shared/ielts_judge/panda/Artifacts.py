"""Strict, versioned PANDA artifacts; no model loading or network at import."""
from typing import Any

import hashlib
import json
import math
import os
from pathlib import Path
import re
from numbers import Real

from ielts_judge.Criteria import IELTS_CRITERIA, LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS


def canonical(value: Any) -> Any:
    """Canonical.

    Args:
        value (Any): Value used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: Any) -> Any:
    """Sha256.

    Args:
        value (Any): Value used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_object(pairs: Any) -> Any:
    """Unique object.

    Args:
        pairs (Any): Pairs used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    """Parse json.

    Args:
        text (str): Text used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    def invalid(value: Any) -> None:
        """Invalid.

        Args:
            value (Any): Value used by this operation.

        Returns:
            None: This operation returns no value.
        """
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid)


def unwrap_completion(text: str) -> Any:
    """Remove exactly one complete outer Markdown fence; never extract/repair JSON fragments.

    Args:
        text (str): Text used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    wrapped = re.fullmatch(r"\s*```(?:json|text|markdown)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.S | re.I)
    return wrapped[1] if wrapped else text


def parse_completion(text: str) -> Any:
    """Parse completion.

    Args:
        text (str): Text used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return parse_json(unwrap_completion(text))


def is_placeholder(text: str) -> bool:
    """Is placeholder.

    Args:
        text (str): Text used by this operation.

    Returns:
        bool: The computed result for the supplied inputs.
    """
    return isinstance(text, str) and re.fullmatch(
        r"(?:not[\s_/-]*applicable|n[\s_/-]*a|none|null)[.!]?", text.strip(), re.I
    ) is not None


def adjacent_band_labels(band: int) -> dict[str, str]:
    """Name explanation directions using the supplied integer band and its real neighbors.

    Args:
        band (int): Band used by this operation.

    Returns:
        dict[str, str]: Named result fields and metadata produced by this operation.
    """
    if type(band) is not int or not 0 <= band <= 9:
        raise ValueError("Adjacent-band text requires an integer score in 0-9")
    return {
        "evidence": "Evidence",
        "why_not_lower": f"Strength supporting band {band} rather than band {band - 1}" if band > 0
            else "Strength supporting band 0 (no lower band)",
        "why_not_higher": f"Weakness preventing band {band + 1}" if band < 9
            else "Weakness preventing a higher band (9 is maximum)",
    }


def parse_legacy_completion(text: str, labels: Any, *, explicit_bands: bool=False, allow_partial: bool=False) -> Any:
    """Parse text insights; allow_partial retains unvalidated diagnostic fields without fixing captions.

    Args:
        text (str): Text used by this operation.
        labels (Any): Labels used by this operation.
        explicit_bands (bool): Explicit bands used by this operation.
        allow_partial (bool): Allow partial used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
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


def read_json(path: Any) -> Any:
    """Read json.

    Args:
        path (Any): Filesystem location to read, validate or write.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: Any, value: Any) -> None:
    """Single-writer atomic replacement; an interrupted write leaves the prior file intact.

    Args:
        path (Any): Filesystem location to read, validate or write.
        value (Any): Value used by this operation.

    Returns:
        None: This operation returns no value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def positive_int(value: Any, name: Any) -> Any:
    """Positive int.

    Args:
        value (Any): Value used by this operation.
        name (Any): Name used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def digest(value: Any, name: str="SHA256") -> str:
    """Calculate the SHA-256 digest of a file without loading the whole file into memory.

    Args:
        value (Any): Value used by this operation.
        name (str): Name used by this operation.

    Returns:
        str: The computed result for the supplied inputs.
    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name}")
    return value


def revision(value: Any) -> Any:
    """Revision.

    Args:
        value (Any): Value used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Model revision must be a pinned 40-character HF commit")
    return value


def scores(value: Any) -> dict[str, Any]:
    """Scores.

    Args:
        value (Any): Value used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    if not isinstance(value, dict) or set(value) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Exactly four public criterion scores are required")
    if any(isinstance(current_value, bool) or not isinstance(current_value, Real) or not math.isfinite(current_value)
           or not 0 <= float(current_value) <= 9 or float(current_value) % 1 for current_value in value.values()):
        raise ValueError("Scores must be finite integers in 0-9, without bool")
    return {key: int(value[key]) for key in PUBLIC_CRITERION_IDS}


def insights(value: Any, essay: str, labels: Any, *, require_quote: bool=True) -> Any:
    """Insights.

    Args:
        value (Any): Value used by this operation.
        essay (str): English essay text supplied for evaluation.
        labels (Any): Labels used by this operation.
        require_quote (bool): Require quote used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    labels = scores(labels)
    if not isinstance(value, dict) or set(value) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Insights must cover exactly four criteria")
    for criterion, item in value.items():
        if not isinstance(item, dict) or set(item) != {"evidence", "why_not_higher", "why_not_lower"}:
            raise ValueError("Each insight needs evidence, why_not_higher and why_not_lower")
        if any(not isinstance(current_value, str) or not current_value.strip() for current_value in item.values()):
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
