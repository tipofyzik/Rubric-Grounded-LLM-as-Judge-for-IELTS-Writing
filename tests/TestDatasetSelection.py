"""Profile IDs and final split quotas, including overlapping target groups."""
from typing import Any

from itertools import combinations
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from offline_research.data.Analysis import PreparedDataAnalyzer
from offline_research.data.Profiles import count_score_profiles, score_profile_ids
from offline_research.data.Selection import select_dataset, selection_options


def frame(patterns: Any, split: str="train") -> Any:
    """Frame.

    Args:
        patterns (Any): Patterns used by this operation.
        split (str): Split used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    result = pd.DataFrame(patterns, columns=PUBLIC_CRITERION_IDS)
    result["essay_id"] = [f"writing9:{split}:{current_index}" for current_index in range(len(result))]
    result["dataset_source"] = "writing9"
    result["status"] = "accepted"
    result["split"] = split
    return result


class DatasetSelectionTests(unittest.TestCase):
    """Manage dataset selection tests through the operations defined by this component."""
    def select(self, data: Any, train: int=2, test: int=1, seed: int=42) -> Any:
        """Select.

        Args:
            data (Any): Data used by this operation.
            train (int): Train used by this operation.
            test (int): Test used by this operation.
            seed (int): Seed used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        options = selection_options(dict(enabled=True, train_per_group=train, test_per_group=test, seed=seed))
        return select_dataset(data, PreparedDataAnalyzer(data).profiles(with_ids=True), options)

    def test_every_count_has_exact_source_ids_and_directions_partition_types(self) -> None:
        """Verify that every count has exact source ids and directions partition types.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(7, 7, 7, 7), (6, 6, 7, 7), (6, 6, 8, 8),
                      (6, 7, 7, 9), (4, 4, 9, 9), (None, 5, 5, 5)])
        profile = count_score_profiles(data, PUBLIC_CRITERION_IDS)
        members = score_profile_ids(data, PUBLIC_CRITERION_IDS)
        self.assertEqual(set(profile), set(members))
        self.assertNotIn("essay_ids", profile)
        self.assertNotIn("direction_essay_ids", profile)
        self.assertEqual(profile["schema_version"], "score-profiles-v2")
        self.assertEqual(members["counts"]["7"]["all_equal"], ["writing9:train:0"])
        self.assertEqual(members["direction_counts"]["7"]["within_1_only"]["below_only"], ["writing9:train:1"])
        self.assertEqual(members["direction_counts"]["7"]["within_1_only"]["both_sides"], ["writing9:train:2"])
        self.assertEqual(members["not_matching_any_type"], ["writing9:train:4"])
        self.assertEqual(profile["excluded_from_analysis"], 1)
        for target, kinds in profile["counts"].items():
            for kind, count in kinds.items():
                ids = members["counts"][target][kind]
                self.assertEqual(count, len(ids))
                if kind != "all_equal":
                    directions = members["direction_counts"][target][kind]
                    self.assertEqual(set(ids), set().union(*map(set, directions.values())))
                    self.assertEqual(count, sum(map(len, directions.values())))
                    for direction, values in directions.items():
                        self.assertEqual(len(values), profile["direction_counts"][target][kind][direction])

    def test_legacy_row_ids_and_bad_source_ids(self) -> None:
        """Verify that legacy row ids and bad source ids.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(6, 6, 6, 6)])
        legacy = score_profile_ids(data.drop(columns="essay_id"), PUBLIC_CRITERION_IDS)
        self.assertEqual(legacy["counts"]["6"]["all_equal"], ["row:0"])
        for bad in (pd.concat([data, data], ignore_index=True), data.assign(essay_id=None)):
            with self.assertRaisesRegex(ValueError, "unique, nonmissing"):
                score_profile_ids(bad, PUBLIC_CRITERION_IDS)

    def test_limits_preserve_splits_and_are_not_multiplied_by_sources(self) -> None:
        """Verify that limits preserve splits and are not multiplied by sources.

        Returns:
            None: This operation returns no value.
        """
        train = frame([(6, 6, 6, 6)] * 8 + [(6, 6, 7, 7)] * 8)
        train.loc[::2, "dataset_source"] = "chillies"
        test = frame([(6, 6, 6, 6)] * 4 + [(6, 6, 7, 7)] * 4, "test")
        rows, report = self.select(pd.concat([train, test], ignore_index=True))
        chosen = rows.loc[rows.status.eq("accepted")]
        self.assertEqual(chosen.groupby("split").size().to_dict(), {"train": 4, "test": 2})
        self.assertTrue(set(chosen.loc[chosen.split.eq("train"), "essay_id"]).issubset(set(train.essay_id)))
        self.assertTrue(set(chosen.loc[chosen.split.eq("test"), "essay_id"]).issubset(set(test.essay_id)))
        self.assertFalse(chosen.essay_id.duplicated().any())
        group = report["splits"]["train"]["groups"]["7"]["within_1_only"]["below_only"]
        self.assertFalse(group["considered"])
        self.assertIsNone(group["limit"])
        self.assertEqual(group["selected"], 2)

    def test_shared_target_memberships_never_exceed_caps_and_fill_when_feasible(self) -> None:
        """Verify that shared target memberships never exceed caps and fill when feasible.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(2, 2, 3, 5), (1, 3, 4, 5), (2, 4, 5, 6)])
        rows, report = self.select(data, train=1)
        actual = count_score_profiles(rows.loc[rows.status.eq("accepted")], PUBLIC_CRITERION_IDS)
        self.assertEqual(actual["counts"]["3"]["within_2_only"], 1)
        self.assertEqual(actual["counts"]["4"]["within_2_only"], 1)
        for split in report["splits"].values():
            for kinds in split["groups"].values():
                for kind, group in kinds.items():
                    entries = [group] if kind == "all_equal" else group.values()
                    for entry in entries:
                        if entry.get("considered", True):
                            self.assertLessEqual(entry["selected"], entry["limit"])

    def test_incompatible_overlaps_maximize_coverage_against_exhaustive_subsets(self) -> None:
        """Verify that incompatible overlaps maximize coverage against exhaustive subsets.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(2, 2, 3, 5), (3, 3, 4, 6), (1, 2, 3, 5), (6, 6, 6, 6)])
        rows, report = self.select(data, train=1)
        group_ids = []
        profile = score_profile_ids(data, PUBLIC_CRITERION_IDS)
        for target in profile["counts"]:
            group_ids.append(set(profile["counts"][target]["all_equal"]))
            for kind, directions in profile["direction_counts"][target].items():
                group_ids.extend(set(ids) for direction, ids in directions.items()
                                 if (kind, direction) != ("within_1_only", "below_only"))
        best = 0
        for size in range(len(data) + 1):
            for ids in combinations(data.essay_id, size):
                counts = [len(set(ids) & group) for group in group_ids]
                if max(counts, default=0) <= 1:
                    best = max(best, sum(counts))
        actual_ids = set(rows.loc[rows.status.eq("accepted"), "essay_id"])
        self.assertEqual(sum(len(actual_ids & group) for group in group_ids), best)
        self.assertEqual(report["splits"]["train"]["selected_essays"], len(actual_ids))

    def test_repeat_and_reordering_are_stable_and_seed_changes_membership(self) -> None:
        """Verify that repeat and reordering are stable and seed changes membership.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(6, 6, 6, 6)] * 30)
        _, first = self.select(data)
        _, repeat = self.select(data.sample(frac=1, random_state=3))
        _, changed = self.select(data, seed=17)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first["splits"]["train"]["essay_ids"], changed["splits"]["train"]["essay_ids"])

    def test_rare_extremes_retained_and_unmatched_or_excluded_are_not_promoted(self) -> None:
        """Verify that rare extremes retained and unmatched or excluded are not promoted.

        Returns:
            None: This operation returns no value.
        """
        data = frame([(0, 0, 0, 0), (9, 9, 9, 9), (0, 0, 9, 9), (6, 6, 6, 6)])
        data.loc[3, "status"] = "excluded"
        rows, _ = self.select(data)
        self.assertEqual(rows.status.tolist(), ["accepted", "accepted", "not_selected", "excluded"])
        self.assertEqual(data.status.tolist(), ["accepted", "accepted", "accepted", "excluded"])

    def test_empty_and_zero_quotas(self) -> None:
        """Verify that empty and zero quotas.

        Returns:
            None: This operation returns no value.
        """
        for data in (frame([]), frame([(6, 6, 6, 6)])):
            rows, report = self.select(data, train=0, test=0)
            self.assertFalse(rows.status.eq("accepted").any())
            self.assertEqual(report["splits"]["train"]["selected_essays"], 0)

    def test_invalid_config_rejected_and_disabled_is_explicit(self) -> None:
        """Verify that invalid config rejected and disabled is explicit.

        Returns:
            None: This operation returns no value.
        """
        valid = dict(enabled=True, train_per_group=250, test_per_group=100, seed=42)
        self.assertIsNone(selection_options(None))
        self.assertIsNone(selection_options({"enabled": False}))
        for changes in ({"train_per_group": -1}, {"test_per_group": True}, {"seed": 1.5},
                        {"enabled": 1}, {"typo": 100}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                selection_options(valid | changes)
        with self.assertRaisesRegex(ValueError, "test_per_group"):
            selection_options({"enabled": True, "train_per_group": 250, "seed": 42})


if __name__ == "__main__":
    unittest.main()
