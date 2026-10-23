"""Globally disjoint criterion profiles shared by preparation and legacy analysis."""

import pandas as pd


def _score_profiles(frame, criteria, lower, upper, *, with_ids):
    all_scores = frame[list(criteria)]
    eligible = (all_scores.notna().all(axis=1) & all_scores.ge(lower).all(axis=1)
                & all_scores.le(upper).all(axis=1) & all_scores.mod(1).eq(0).all(axis=1))
    scores = all_scores.loc[eligible]
    if with_ids and "essay_id" in frame:
        ids = frame["essay_id"]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("Score profiles require unique, nonmissing essay IDs")
    elif with_ids:
        if not frame.index.is_unique:
            raise ValueError("Score profiles require a unique row index without essay_id")
        ids = pd.Series([f"row:{index}" for index in frame.index], index=frame.index)
    else:
        ids = None
    eligible_ids = ids.loc[eligible] if ids is not None else None

    def collect(group):
        return sorted(eligible_ids.loc[group].tolist()) if eligible_ids is not None else int(group.sum())
    targets = list(range(lower, upper + 1))
    equal = scores.max(axis=1).eq(scores.min(axis=1))
    distances = {target: scores.sub(target).abs().max(axis=1) for target in targets}
    within_one = ~equal & pd.DataFrame(distances).eq(1).any(axis=1)
    memberships = {name: pd.Series(False, index=scores.index)
                   for name in ("all_equal", "within_1_only", "within_2_only")}
    counts, directions = {}, {}
    for target in targets:
        differences = scores.sub(target)
        distance = distances[target]
        above, below = differences.gt(0).any(axis=1), differences.lt(0).any(axis=1)
        groups = {"all_equal": distance.eq(0), "within_1_only": ~equal & distance.eq(1),
                  "within_2_only": ~equal & ~within_one & distance.eq(2)}
        for name, group in groups.items():
            memberships[name] |= group
        label = str(target)
        counts[label] = {name: collect(group) for name, group in groups.items()}
        directions[label] = {
            name: {"above_only": collect(group & above & ~below),
                   "below_only": collect(group & below & ~above),
                   "both_sides": collect(group & above & below)}
            for name, group in groups.items() if name != "all_equal"}
    unmatched = ~(memberships["all_equal"] | memberships["within_1_only"] | memberships["within_2_only"])
    return {
        "schema_version": "score-profile-ids-v1" if with_ids else "score-profiles-v2", "score_range": [lower, upper],
        "criteria": list(criteria), "target_scores": targets,
        "target_policy": f"whole-band targets {lower}-{upper}",
        "input_essays": len(frame), "analyzed_essays": len(scores),
        "excluded_from_analysis": len(frame) - len(scores),
        "eligibility": f"four complete integer criterion scores from {lower} to {upper}",
        "target_presence_required": False,
        "type_assignment": "global priority: all_equal, then within_1_only at any target, then within_2_only at any target; types never overlap",
        "type_totals": {name: collect(group) for name, group in memberships.items()},
        "not_matching_any_type": collect(unmatched),
        "counting_rules": {
            "all_equal": "all four scores equal this target",
            "within_1_only": "exclude uniform vectors; maximum absolute deviation from this target is exactly 1",
            "within_2_only": "exclude uniform vectors and essays matching within_1_only at any target; maximum absolute deviation from this target is exactly 2",
            "overlap": "types are disjoint across all targets; an essay may match multiple targets within its assigned type",
            "type_totals": "unique essays per globally disjoint type; totals plus not_matching_any_type equal analyzed_essays",
        },
        "counts": counts,
        "direction_rules": {
            "above_only": "at least one score strictly above the target and none below it",
            "below_only": "at least one score strictly below the target and none above it",
            "both_sides": "at least one score strictly above and at least one strictly below this same target; equal scores do not affect direction",
        },
        "direction_counts": directions,
    }


def count_score_profiles(frame, criteria, lower=0, upper=9):
    return _score_profiles(frame, criteria, lower, upper, with_ids=False)


def score_profile_ids(frame, criteria, lower=0, upper=9):
    return _score_profiles(frame, criteria, lower, upper, with_ids=True)
