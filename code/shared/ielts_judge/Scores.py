"""Public judge scores with exact fields and bounded format tolerance."""

import json
import math
import re
from decimal import Decimal
from numbers import Real
from typing import Any, cast

from ielts_judge.Criteria import LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS

OUTPUT_SCHEMA_VERSION = "scores-integer-v2"


def public_scores(value: Any, *, legacy: bool=False) -> dict[str, int]:
    """Validate four criterion scores and normalize them to integers.

    Args:
        value (Any): Dictionary with one numeric score per criterion.
        legacy (bool): Accept legacy criterion IDs and translate them to public IDs.

    Returns:
        dict[str, int]: Scores in public criterion order, each an integer from 0 to 9.

    Raises:
        ValueError: A criterion is missing, unknown or duplicated after translation,
            or a score is outside the allowed range or not integer-valued.
    """
    if not isinstance(value, dict):
        raise ValueError("Scores must be a JSON object")
    result = {}
    for name, score in value.items():
        key = LEGACY_TO_PUBLIC.get(name, name) if legacy else name
        if key in result:
            raise ValueError("Duplicate public score key")
        if isinstance(score, bool) or not isinstance(score, (Real, Decimal)):
            raise ValueError("Scores must be finite integer values in 0-9")
        numeric = cast(Any, score)
        try:
            valid = 0 <= numeric <= 9 and math.isfinite(numeric) and numeric % 1 == 0
        except (ArithmeticError, ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError("Scores must be finite integer values in 0-9")
        result[key] = int(numeric)
    if set(result) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Exactly four public score keys are required")
    return {name: result[name] for name in PUBLIC_CRITERION_IDS}


def parse_scores(text: str) -> dict[str, int]:
    """Parse the judge reply into four IELTS criterion scores.

    Args:
        text (str): Judge reply containing one JSON object, optionally in a code fence.

    Returns:
        dict[str, int]: Scores from 0 to 9 keyed by public criterion ID.

    Raises:
        ValueError: The reply has invalid JSON, criterion IDs or scores.
    """
    if not isinstance(text, str):
        raise ValueError("Judge completion must be text")
    fence = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.S | re.I)
    if fence:
        text = fence[1]

    def unique(pairs: Any) -> Any:
        """Collect JSON object fields, rejecting duplicate keys.

        Args:
            pairs (Any): Ordered key-value pairs from the JSON decoder.

        Returns:
            Any: Dictionary containing the fields of one JSON object.

        Raises:
            ValueError: A key occurs more than once.
        """
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_: Any) -> None:
        """Reject a nonfinite JSON number before score validation.

        Args:
            _ (Any): Nonstandard literal passed by the JSON decoder.

        Raises:
            ValueError: The reply uses NaN or an infinity literal.
        """
        raise ValueError("Nonfinite JSON constant")

    # Decimal keeps tiny nonzero fractions from becoming an integer through float rounding.
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid, parse_float=Decimal)
    except ArithmeticError as error:
        raise ValueError("Invalid decimal score") from error
    return public_scores(value)
