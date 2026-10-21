"""Shared IELTS criterion definitions loaded from the project resource file."""

from typing import Any

import yaml

from .paths import PATHS

CRITERIA_PATH = PATHS["resources"]["ielts_criteria"]


def load_criteria() -> dict[str, dict[str, Any]]:
    """Load and validate criterion names and band descriptors from YAML.

    Returns:
        dict[str, dict[str, Any]]: Criteria keyed by their stable score identifiers.
    """
    with open(CRITERIA_PATH, "r", encoding="utf-8") as criteria_file:
        resource = yaml.safe_load(criteria_file)

    criteria = resource.get("criteria") if isinstance(resource, dict) else None
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError(f"No IELTS criteria are defined in {CRITERIA_PATH}.")

    for criterion_id, definition in criteria.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Criterion {criterion_id!r} must be a mapping.")
        if not isinstance(definition.get("display_name"), str):
            raise ValueError(f"Criterion {criterion_id!r} needs a display_name.")
        if not isinstance(definition.get("source_name"), str):
            raise ValueError(f"Criterion {criterion_id!r} needs a source_name.")
        bands = definition.get("bands")
        if not isinstance(bands, dict) or not bands:
            raise ValueError(f"Criterion {criterion_id!r} needs band descriptors.")
        if any(not isinstance(descriptor, str) for descriptor in bands.values()):
            raise ValueError(f"Criterion {criterion_id!r} has a non-text band descriptor.")

    return criteria


IELTS_CRITERIA = load_criteria()
CRITERION_IDS = tuple(IELTS_CRITERIA)
LEGACY_TO_PUBLIC = {
    criterion_id: "task_response" if criterion_id == "task_achievement" else criterion_id
    for criterion_id in CRITERION_IDS
}
PUBLIC_CRITERION_IDS = tuple(LEGACY_TO_PUBLIC.values())
CRITERION_NAMES = {
    criterion_id: definition["display_name"]
    for criterion_id, definition in IELTS_CRITERIA.items()
}
CRITERION_SOURCE_NAMES = {
    criterion_id: definition["source_name"]
    for criterion_id, definition in IELTS_CRITERIA.items()
}


def format_score_schema(criterion_ids: tuple[str, ...] = CRITERION_IDS) -> str:
    """Build the JSON score schema from the configured criterion identifiers.

    Returns:
        str: JSON-like score fields for insertion into an LLM prompt.
    """
    fields = [f'"{criterion_id}": number' for criterion_id in criterion_ids]
    return "{\n" + ",\n".join(fields) + "\n}"


def format_rubric() -> str:
    """Render the shared, abbreviated rubric without loading a model."""
    sections = []
    for definition in IELTS_CRITERIA.values():
        bands = definition["bands"]
        sections.append(
            definition["display_name"].upper()
            + "\n"
            + "\n".join(bands[band] for band in sorted(bands, reverse=True))
        )
    return "\n\n".join(sections)
