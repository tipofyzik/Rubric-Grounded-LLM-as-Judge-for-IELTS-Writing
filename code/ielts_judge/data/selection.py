"""Deterministic profile quotas applied after source validation and deduplication."""

from collections import defaultdict
import hashlib

import numpy as np


def selection_options(config):
    if config is None:
        return None
    names = {"enabled", "train_per_group", "test_per_group", "seed"}
    if not isinstance(config, dict) or set(config) - names:
        raise ValueError("Unknown dataset.selection options")
    if type(config.get("enabled")) is not bool:
        raise ValueError("dataset.selection.enabled must be a boolean")
    if not config["enabled"]:
        return None
    for name in ("train_per_group", "test_per_group", "seed"):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError(f"dataset.selection.{name} must be a nonnegative integer")
    return dict(config)


def _select_ids(groups, limit, seed):
    """Shared memberships consume every matching quota, including adjacent targets."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    memberships = defaultdict(list)
    for name, ids in groups.items():
        for essay_id in ids:
            memberships[essay_id].append(name)
    buckets = defaultdict(list)
    for essay_id, names in memberships.items():
        buckets[tuple(names)].append(essay_id)
    signatures = sorted(buckets)
    active = [name for name, ids in groups.items() if ids]
    if not signatures or limit == 0:
        return set()
    matrix = np.array([[int(name in signature) for signature in signatures] for name in active])
    upper = np.array([min(limit, len(groups[name])) for name in active])
    # Only membership signatures are optimization variables, not individual essays.
    # Maximizing filled places fills every quota whenever jointly feasible.
    result = milp(
        c=-matrix.sum(axis=0).astype(float), integrality=np.ones(len(signatures)),
        bounds=Bounds(np.zeros(len(signatures)), [len(buckets[key]) for key in signatures]),
        constraints=LinearConstraint(matrix, np.zeros(len(active)), upper),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Profile quota selection failed: {result.message}")
    amounts = np.rint(result.x).astype(int)
    if (matrix @ amounts > upper).any() or (amounts < 0).any():
        raise RuntimeError("Profile quota solver returned invalid counts")
    selected = set()
    for signature, count in zip(signatures, amounts):
        candidates = sorted(buckets[signature], key=lambda essay_id: (
            hashlib.sha256(f"{seed}:{essay_id}".encode("utf-8")).digest(), essay_id))
        selected.update(candidates[:count])
    return selected


def select_dataset(frame, profiles, options):
    """Keep source splits; retain valid unselected rows with a separate status."""
    report = {
        "schema_version": "profile-selection-v1", "config": options,
        "input_policy": "accepted rows after source validation and cross-source deduplication",
        "split_policy": "select within each existing split; never move essays between train and test",
        "quota_policy": "cap actual membership in every considered target/type/direction group; unique output IDs",
        "overlap_policy": "maximize total filled group places; shared essays consume all matching considered quotas",
        "ordering": "sorted membership signatures; within a signature SHA256(seed:essay_id), then essay_id",
        "splits": {},
    }
    selected_ids = set()
    for split in ("train", "test"):
        profile = profiles["by_split"][split]
        limit = options[f"{split}_per_group"]
        groups, entries = {}, {}
        for target in profile["counts"]:
            equal = profile["counts"][target]["all_equal"]
            groups[f"{target}/all_equal"] = equal
            entries[target] = {"all_equal": {"available": len(equal), "limit": limit}}
            for kind, directions in profile["direction_counts"][target].items():
                entries[target][kind] = {}
                for direction, ids in directions.items():
                    considered = not (kind == "within_1_only" and direction == "below_only")
                    entries[target][kind][direction] = {
                        "available": len(ids), "considered": considered,
                        "limit": limit if considered else None,
                    }
                    if considered:
                        groups[f"{target}/{kind}/{direction}"] = ids
        selected = _select_ids(groups, limit, options["seed"])
        for target, kinds in entries.items():
            equal_ids = sorted(selected.intersection(profile["counts"][target]["all_equal"]))
            kinds["all_equal"].update(selected=len(equal_ids), essay_ids=equal_ids)
            for kind, directions in profile["direction_counts"][target].items():
                for direction, ids in directions.items():
                    actual = sorted(selected.intersection(ids))
                    kinds[kind][direction].update(selected=len(actual), essay_ids=actual)
        input_count = int((frame.status.eq("accepted") & frame.split.eq(split)).sum())
        report["splits"][split] = {
            "input_essays": input_count, "selected_essays": len(selected),
            "not_selected_essays": input_count - len(selected),
            "essay_ids": sorted(selected), "groups": entries,
        }
        selected_ids.update(selected)
    rows = frame.copy()
    rows.loc[rows.status.eq("accepted") & ~rows.essay_id.isin(selected_ids), "status"] = "not_selected"
    return rows, report
