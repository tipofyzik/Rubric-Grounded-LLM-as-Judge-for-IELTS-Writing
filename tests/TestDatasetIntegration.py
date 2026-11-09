"""Data contracts exercised through real CSV, JSONL and Parquet I/O."""
from typing import Any

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
sys.path.insert(0, str(ROOT))
from ielts_judge.Criteria import CRITERION_IDS, IELTS_CRITERIA, PUBLIC_CRITERION_IDS
from offline_research.data.DatasetTransformer import DatasetTransformer
from ielts_judge.data.Preparation import file_hash, legacy_labels, load_prepared, rounded_overall, source_paths, validate_splits
from offline_research.data.Profiles import count_score_profiles
from ielts_judge.Paths import PATHS, load_paths


class DatasetIntegrationTests(unittest.TestCase):
    """Manage dataset integration tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        (ROOT / "TEMP").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "TEMP")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "sources"
        self.output = self.root / "release"
        self.paths = source_paths(self.input)
        self.rows = {name: {current_text: [] for current_text in ("train", "test")} for name in self.paths}
        self.transformer = DatasetTransformer({"source_root": str(self.input)})

    def old(self, scores: Any=(2.5, 2, 3, 2), essay: str="Old essay", band: str="<4", evaluation: Any=None) -> Any:
        """Old.

        Args:
            scores (Any): Four criterion scores using the public IELTS score contract.
            essay (str): English essay text supplied for evaluation.
            band (str): Band used by this operation.
            evaluation (Any): Evaluation used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if evaluation is None:
            evaluation = '\n'.join(f"{IELTS_CRITERIA[name]['source_name']}: {score}"
                                   for name, score in zip(CRITERION_IDS, scores) if score is not None) + '\nOverall Band: 6.5'
        return dict(prompt="Discuss education.", essay=essay, evaluation=evaluation, band=band)

    def new(self, scores: Any=(6, 6, 7, 6), essay: str="New essay", source_id: str="new-1", **changes: Any) -> Any:
        """New.

        Args:
            scores (Any): Four criterion scores using the public IELTS score contract.
            essay (str): English essay text supplied for evaluation.
            source_id (str): Source id used by this operation.
            changes (Any): Changes used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        values = dict(zip(("band_task", "band_coherence", "band_lexical", "band_grammar"), scores))
        record = dict(id=source_id, question="Discuss education.", essay=essay, task_type="essay",
                      question_type="discussion", band_overall=rounded_overall(scores), **values)
        record.update(changes)
        return record

    def write(self) -> None:
        """Write.

        Returns:
            None: This operation returns no value.
        """
        for source, files in self.paths.items():
            for split, path in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = self.rows[source][split]
                if source == "chillies":
                    pd.DataFrame(rows, columns=("prompt", "essay", "evaluation", "band")).to_csv(path, index=False)
                else:
                    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding="utf-8")
        (self.paths['writing9']['train'].parent / 'band9_augmented.jsonl').write_text('INVALID; MUST NOT BE READ', encoding='utf-8')

    def prepare(self) -> Any:
        """Assemble the current operation's validated inputs and metadata.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        self.write()
        return self.transformer.prepare()

    def test_old_extraction_truncation_and_overall_interval_preserve_source(self) -> None:
        """Verify that old extraction truncation and overall interval preserve source.

        Returns:
            None: This operation returns no value.
        """
        original = self.old(scores=(3.9, 2.5, 3, 2), band=" <4\r\n", essay="  Exact\r\nEssay  ")
        self.rows['chillies']['train'] = [original]
        prepared = self.prepare()
        row = prepared.rows.iloc[0]
        self.assertEqual(tuple(row[list(PUBLIC_CRITERION_IDS)]), (3, 2, 3, 2))
        self.assertEqual(row.essay, original['essay'])
        self.assertEqual(row.evaluation, original['evaluation'])
        self.assertEqual(row.overall_label, '<4')
        self.assertTrue(pd.isna(row.overall))
        self.assertEqual(json.loads(row.source_scores_json)['task_response'], 3.9)
        self.assertEqual(json.loads(row.source_scores_json)['overall'], original['band'])
        self.assertEqual(row.status, 'accepted')
        self.assertEqual(list(legacy_labels(prepared.rows)), list(CRITERION_IDS))

    def test_missing_old_scores_are_not_replaced_by_overall(self) -> None:
        """Verify that missing old scores are not replaced by overall.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(evaluation='No criterion labels', band='8')]
        row = self.prepare().rows.iloc[0]
        self.assertEqual(row.status, 'excluded')
        self.assertTrue(row[list(PUBLIC_CRITERION_IDS)].isna().all())
        self.assertIn('missing_score:task_response', row.issues)

    def test_chillies_requires_all_low_integer_labels_and_low_source_overall(self) -> None:
        """Verify that chillies requires all low integer labels and low source overall.

        Returns:
            None: This operation returns no value.
        """
        cases = [
            ((0, 1, 2, 3), '<4', True),
            ((3.9, 2.5, 3, 0), '3.5', True),
            ((0, 0, 0, 0), '0', True),
            ((3, 3, 3, 4), '<4', False),
            ((3, 3, 3, 3), '4', False),
            ((3, 3, 3, 3), '4.5', False),
            ((3, 3, 3, None), '<4', False),
        ]
        self.rows['chillies']['train'] = [
            self.old(scores=scores, band=band, essay=f'Low-band case {current_index}')
            for current_index, (scores, band, _) in enumerate(cases)]
        prepared = self.prepare()
        for row, (_, _, accepted) in zip(prepared.rows.to_dict('records'), cases):
            with self.subTest(essay=row['essay']):
                self.assertEqual(row['status'] == 'accepted', accepted)
                if not accepted:
                    self.assertIn('chillies_outside_low_band_scope', row['issues'])
        self.assertTrue(prepared.extracted['chillies'].iloc[3].status == 'accepted')
        self.assertEqual(json.loads(prepared.rows.iloc[1].source_scores_json)['task_response'], 3.9)

    def test_out_of_scope_chillies_cannot_conflict_with_or_replace_writing9(self) -> None:
        """Verify that out of scope chillies cannot conflict with or replace writing9.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['test'] = [
            self.old(scores=(6.5, 6, 6, 6), essay='Conflicting high-band essay'),
            self.old(scores=(9, 9, 9, 9), essay='Compatible high-band essay', band='9')]
        self.rows['writing9']['train'] = [
            self.new(scores=(6, 6, 6, 6), essay='Conflicting high-band essay', source_id='w6'),
            self.new(scores=(9, 9, 9, 9), essay='Compatible high-band essay', source_id='w9'),
            self.new(scores=(1, 2, 3, 3), essay='Writing9 low scores stay valid', source_id='w-low')]
        prepared = self.prepare()
        self.assertEqual(len(prepared.split('train')), 3)
        self.assertEqual(len(prepared.split('test')), 0)
        self.assertTrue(prepared.rows.loc[prepared.rows.dataset_source.eq('writing9'), 'status'].eq('accepted').all())
        self.assertTrue(all('chillies_outside_low_band_scope' in issues
                            for issues in prepared.rows.loc[prepared.rows.dataset_source.eq('chillies'), 'issues']))
        self.transformer.save_processed(prepared, self.output)
        self.assertEqual(len(list((self.output/'plots/chillies').glob('*.png'))), 5)
        audit = json.loads((self.output/'audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['sources']['chillies']['accepted'], 0)

    def test_old_transformation_rebuilds_from_source_without_saved_csv(self) -> None:
        """Verify that old transformation rebuilds from source without saved csv.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(6.5, 6, 7, 6))]
        self.write()
        self.paths['writing9']['train'].unlink()
        extracted = self.transformer.transform_hugging_face_dataset()
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted.iloc[0].task_response, 6)
        self.assertEqual(json.loads(extracted.iloc[0].source_scores_json)['task_response'], 6.5)

    def test_source_range_checked_before_truncation_and_new_fraction_rejected(self) -> None:
        """Verify that source range checked before truncation and new fraction rejected.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(9.5, 6, 6, 6))]
        self.rows['writing9']['test'] = [self.new(scores=(6.5, 6, 6, 6))]
        rows = self.prepare().rows
        self.assertTrue(rows.status.eq('excluded').all())
        self.assertIn('invalid_score:task_response', rows.iloc[0].issues)
        self.assertIn('fractional_criterion:task_response', rows.iloc[1].issues)

    def test_writing9_mismatch_and_task1_are_explicit(self) -> None:
        """Verify that writing9 mismatch and task1 are explicit.

        Returns:
            None: This operation returns no value.
        """
        self.rows['writing9']['train'] = [self.new(band_overall=9, question='Write a letter to your manager.')]
        row = self.prepare().rows.iloc[0]
        self.assertIn('overall_mismatch', row.issues)
        self.assertIn('task1_prompt', row.issues)
        self.assertEqual(row.overall, 9)

    def test_cross_source_duplicate_prefers_test_and_preserves_extracted_rows(self) -> None:
        """Verify that cross source duplicate prefers test and preserves extracted rows.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(2, 2, 3, 2), essay='  Shared\r\nessay  ')]
        self.rows['writing9']['test'] = [self.new(scores=(2, 2, 3, 2), essay='Shared\nessay')]
        prepared = self.prepare()
        self.assertEqual(len(prepared.split('train')), 0)
        self.assertEqual(len(prepared.split('test')), 1)
        self.assertEqual(prepared.rows.iloc[0].status, 'duplicate')
        self.assertEqual(prepared.extracted['chillies'].iloc[0].status, 'accepted')
        self.assertEqual(prepared.extracted['chillies'].iloc[0].issues, [])
        self.assertEqual(prepared.rows.iloc[0].duplicate_of, prepared.rows.iloc[1].essay_id)

    def test_conflicts_exclude_every_copy_even_when_one_copy_invalid(self) -> None:
        """Verify that conflicts exclude every copy even when one copy invalid.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(2, 2, 3, 2), essay='Conflict')]
        self.rows['writing9']['test'] = [self.new(scores=(2.5, 2, 3, 2), essay='Conflict')]
        prepared = self.prepare()
        self.assertTrue(prepared.rows.status.eq('excluded').all())
        self.assertTrue(all('duplicate_essay_conflict' in values for values in prepared.rows.issues))

    def test_source_id_collision_is_scoped_to_source(self) -> None:
        """Verify that source id collision is scoped to source.

        Returns:
            None: This operation returns no value.
        """
        self.rows['writing9']['train'] = [self.new(essay='First'), self.new(essay='Second')]
        prepared = self.prepare()
        self.assertTrue(all('source_id_collision' in values for values in prepared.rows.issues))

    def test_profiles_are_exclusive_and_both_sides_requires_strict_opposites(self) -> None:
        """Verify that profiles are exclusive and both sides requires strict opposites.

        Returns:
            None: This operation returns no value.
        """
        patterns = [(7, 7, 7, 7), (6, 6, 7, 7), (6, 6, 8, 8), (6, 7, 7, 9), (4, 4, 9, 9)]
        frame = pd.DataFrame(patterns, columns=PUBLIC_CRITERION_IDS)
        report = count_score_profiles(frame, PUBLIC_CRITERION_IDS)
        self.assertEqual(report['type_totals'], dict(all_equal=1, within_1_only=2, within_2_only=1))
        self.assertEqual(report['not_matching_any_type'], 1)
        self.assertEqual(report['counts']['5']['within_2_only'], 0)
        self.assertEqual(report['direction_counts']['6']['within_1_only']['above_only'], 1)
        self.assertEqual(report['direction_counts']['7']['within_1_only']['below_only'], 1)
        self.assertEqual(report['direction_counts']['7']['within_1_only']['both_sides'], 1)
        self.assertEqual(report['direction_counts']['7']['within_2_only']['both_sides'], 1)

    def test_parquet_roundtrip_manifest_source_hashes_and_no_leakage(self) -> None:
        """Verify that parquet roundtrip manifest source hashes and no leakage.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old()]
        self.rows['writing9']['test'] = [self.new()]
        prepared = self.prepare()
        before = {str(candidate_path): file_hash(candidate_path) for files in self.paths.values() for candidate_path in files.values()}
        self.transformer.save_processed(prepared, self.output)
        train, test, manifest = load_prepared(self.output)
        self.assertEqual({candidate_path.relative_to(self.output).as_posix() for candidate_path in self.output.rglob('*.parquet')},
                         {'train.parquet', 'test.parquet'})
        self.assertFalse({'excluded.parquet', 'not_selected.parquet', 'extracted/chillies.parquet',
                          'extracted/writing9.parquet'} & manifest['artifacts'].keys())
        self.assertEqual((len(train), len(test)), (1, 1))
        self.assertEqual(train.iloc[0].essay_id, 'chillies:train:1')
        self.assertEqual(list(train[list(PUBLIC_CRITERION_IDS)].dtypes.astype(str)), ['Int64']*4)
        self.assertEqual(manifest['sources'][0]['rows_read'], 1)
        self.assertIn('source_selection_policy', manifest)
        self.assertEqual(before, {str(candidate_path): file_hash(candidate_path) for files in self.paths.values() for candidate_path in files.values()})
        for name in manifest['artifacts']:
            if name.endswith('.png'):
                with Image.open(self.output/name) as image:
                    self.assertEqual(image.size, (1200, 750))
        self.assertEqual(train.attrs['manifest_sha256'], file_hash(self.output/'manifest.json'))
        validate_splits(train.sample(frac=1), test.sample(frac=1))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits(train, train.assign(split='test', essay_id='other'))

    def test_corruption_partial_release_and_unknown_source_are_rejected(self) -> None:
        """Verify that corruption partial release and unknown source are rejected.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(), self.old(essay='Another')]
        self.write()
        self.transformer.prepare_and_save(self.output, limit_per_split=1)
        with self.assertRaisesRegex(ValueError, 'Partial'):
            load_prepared(self.output)
        self.assertEqual(len(load_prepared(self.output, allow_partial=True)[0]), 1)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            load_prepared(self.output, 'unknown', allow_partial=True)
        (self.output/'train.parquet').write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_prepared(self.output, allow_partial=True)

    def test_bad_output_rejected_before_sources_or_reports_change(self) -> None:
        """Verify that bad output rejected before sources or reports change.

        Returns:
            None: This operation returns no value.
        """
        self.write()
        with self.assertRaisesRegex(ValueError, 'outside source'):
            self.transformer.prepare_and_save(self.paths['chillies']['train'].parent)
        self.paths['writing9']['test'].unlink()
        with self.assertRaises(FileNotFoundError):
            self.transformer.prepare_and_save(self.output)
        self.assertFalse(self.output.exists())

    def test_explicit_checkout_paths_do_not_reuse_imported_global_locations(self) -> None:
        """Verify that explicit checkout paths do not reuse imported global locations.

        Returns:
            None: This operation returns no value.
        """
        checkout = self.root/'notebook-checkout'
        (checkout/'code/shared').mkdir(parents=True)
        (checkout/'code/shared/paths.yaml').write_text('datasets:\n  prepared: code/shared/resources/own-release\n', encoding='utf-8')
        selected = load_paths(checkout)
        self.assertEqual(Path(selected['datasets']['prepared']), checkout/'code/shared/resources/own-release')
        self.assertNotEqual(selected['datasets']['prepared'], PATHS['datasets']['prepared'])

    def test_failed_write_invalidates_manifest_and_retry_works(self) -> None:
        """Verify that failed write invalidates manifest and retry works.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old()]
        prepared = self.prepare()
        self.output.mkdir()
        (self.output/'manifest.json').write_text('{}')
        (self.output/'plots').write_text('physical file blocks directory creation')
        with self.assertRaises(OSError):
            self.transformer.save_processed(prepared, self.output)
        self.assertFalse((self.output/'manifest.json').exists())
        (self.output/'plots').unlink()
        self.transformer.save_processed(prepared, self.output)
        self.assertEqual(len(load_prepared(self.output)[0]), 1)

    def test_cleanup_can_be_rebuilt_from_verified_extracted_without_parsing_again(self) -> None:
        """Verify that cleanup can be rebuilt from verified extracted without parsing again.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old()]
        prepared = self.prepare()
        self.transformer.save_intermediate_datasets = True
        self.transformer.save_processed(prepared, self.output)
        # An older full release may still supply a verified extraction cache.
        manifest_path = self.output/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['schema_version'] = 'ielts-two-sources-v1'
        manifest.pop('save_intermediate_datasets')
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(pd.read_parquet(self.output/'not_selected.parquet').empty)
        parent_hash = file_hash(self.output/'manifest.json')
        with patch.object(self.transformer, 'extract_score', side_effect=AssertionError('Unexpected source parsing')):
            cached = self.transformer.prepare_from_extracted(self.output)
        self.assertEqual(cached.parent_manifest, parent_hash)
        self.assertEqual(cached.rows.essay_id.tolist(), prepared.rows.essay_id.tolist())
        newer = self.root/'next-release'
        self.transformer.save_intermediate_datasets = False
        self.transformer.save_processed(cached, newer)
        self.assertEqual(load_prepared(newer)[2]['parent_manifest_sha256'], parent_hash)
        self.assertEqual({candidate_path.name for candidate_path in newer.rglob('*.parquet')}, {'train.parquet', 'test.parquet'})

    def test_profile_selection_saves_only_final_splits_and_preserves_audit(self) -> None:
        """Verify that profile selection saves only final splits and preserves audit.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(2, 2, 2, 2), essay='Rare low essay')]
        self.rows['writing9']['train'] = [self.new(scores=(6, 6, 6, 6), essay=f'Train {current_index}', source_id=f'tr-{current_index}') for current_index in range(5)]
        self.rows['writing9']['test'] = [self.new(scores=(6, 6, 6, 6), essay=f'Test {current_index}', source_id=f'te-{current_index}') for current_index in range(4)]
        self.rows['writing9']['test'].append(self.new(scores=(4, 4, 9, 9), essay='No matching profile', source_id='wide'))
        self.rows['writing9']['train'].append(self.new(essay='Invalid overall', source_id='bad', band_overall=9))
        prepared = self.prepare()
        self.transformer.selection = dict(enabled=True, train_per_group=2, test_per_group=1, seed=42)
        self.transformer.save_processed(prepared, self.output)
        train, test, manifest = load_prepared(self.output)
        self.assertEqual((len(train), len(test)), (3, 1))
        self.assertEqual({candidate_path.relative_to(self.output).as_posix() for candidate_path in self.output.rglob('*.parquet')},
                         {'train.parquet', 'test.parquet'})
        self.assertFalse({'excluded.parquet', 'not_selected.parquet', 'extracted/chillies.parquet',
                          'extracted/writing9.parquet'} & manifest['artifacts'].keys())
        actual = pd.concat([train, test], ignore_index=True).sort_values('essay_id')
        expected = prepared.rows.loc[prepared.rows.essay_id.isin(actual.essay_id)].sort_values('essay_id')
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
        before = json.loads((self.output/'score_profiles_before_selection.json').read_text())
        after = json.loads((self.output/'score_profile_counts.json').read_text())
        after_ids = json.loads((self.output/'score_profile_ids.json').read_text())
        self.assertFalse((self.output/'score_profile_ids_before_selection.json').exists())
        self.assertEqual(before['analyzed_essays'], 11)
        self.assertEqual(after['analyzed_essays'], 4)
        self.assertNotIn('essay_ids', before)
        self.assertNotIn('essay_ids', after)
        self.assertEqual(set(after), set(after_ids))
        self.assertEqual(set(after_ids['by_split']['train']['counts']['6']['all_equal']), set(train.loc[train.task_response.eq(6), 'essay_id']))
        audit = json.loads((self.output/'audit.json').read_text())
        self.assertEqual(audit['status_counts']['not_selected'], 7)
        self.assertEqual(audit['status_counts']['excluded'], 1)
        self.assertEqual(manifest['selection'], self.transformer.selection)
        self.assertTrue(prepared.rows.status.eq('accepted').sum() == 11)
        self.assertEqual(sum(len(frame) for frame in prepared.extracted.values()), 12)
        missing_artifact = dict(manifest)
        missing_artifact['artifacts'] = {name: info for name, info in manifest['artifacts'].items() if name != 'selection.json'}
        (self.output/'manifest.json').write_text(json.dumps(missing_artifact))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_prepared(self.output)

    def test_optional_intermediate_exports_preserve_splits_and_reports(self) -> None:
        """Verify that optional intermediate exports preserve splits and reports.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old(scores=(2, 2, 2, 2), essay='Low essay')]
        self.rows['writing9']['train'] = [self.new(scores=(6, 6, 6, 6), essay=f'Train {current_index}', source_id=f'tr-{current_index}') for current_index in range(4)]
        self.rows['writing9']['test'] = [self.new(scores=(6, 6, 6, 6), essay=f'Test {current_index}', source_id=f'te-{current_index}') for current_index in range(3)]
        self.rows['writing9']['train'].append(self.new(essay='Invalid overall', source_id='bad', band_overall=9))
        prepared = self.prepare()
        selection = dict(enabled=True, train_per_group=2, test_per_group=1, seed=42)
        compact = DatasetTransformer({'source_root': str(self.input)}, selection=selection)
        compact.save_processed(prepared, self.output)
        compact_train, compact_test, compact_manifest = load_prepared(self.output)
        full_output = self.root/'with-intermediates'
        full = DatasetTransformer({'source_root': str(self.input)}, selection=selection,
                                  save_intermediate_datasets=True)
        full.save_processed(prepared, full_output)
        full_train, full_test, full_manifest = load_prepared(full_output)
        extra = {'excluded.parquet', 'not_selected.parquet', 'extracted/chillies.parquet', 'extracted/writing9.parquet'}
        self.assertFalse(compact_manifest['save_intermediate_datasets'])
        self.assertTrue(full_manifest['save_intermediate_datasets'])
        self.assertEqual({candidate_path.relative_to(full_output).as_posix() for candidate_path in full_output.rglob('*.parquet')},
                         {'train.parquet', 'test.parquet'} | extra)
        self.assertTrue(extra.issubset(full_manifest['artifacts']))
        for name in ('train.parquet', 'test.parquet', 'audit.json', 'score_profile_counts.json',
                     'score_profile_ids.json', 'score_profiles_before_selection.json', 'selection.json'):
            self.assertEqual(file_hash(self.output/name), file_hash(full_output/name))
        pd.testing.assert_frame_equal(compact_train, full_train)
        pd.testing.assert_frame_equal(compact_test, full_test)
        self.assertEqual(len(pd.read_parquet(full_output/'excluded.parquet')), 1)
        self.assertEqual(len(pd.read_parquet(full_output/'not_selected.parquet')), 4)
        for source, extracted in prepared.extracted.items():
            pd.testing.assert_frame_equal(pd.read_parquet(full_output/'extracted'/f'{source}.parquet'),
                                          extracted.sort_values('essay_id').reset_index(drop=True))
        altered = dict(full_manifest)
        altered['artifacts'] = {name: value for name, value in full_manifest['artifacts'].items() if name != 'not_selected.parquet'}
        (full_output/'manifest.json').write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_prepared(full_output)

    def test_main_uses_configured_profile_quotas_and_invalid_config_keeps_release(self) -> None:
        """Verify that main uses configured profile quotas and invalid config keeps release.

        Returns:
            None: This operation returns no value.
        """
        from offline_research import PrepareData as runner
        self.rows['writing9']['train'] = [self.new(scores=(6, 6, 6, 6), essay=f'Train {current_index}', source_id=f'tr-{current_index}') for current_index in range(4)]
        self.rows['writing9']['test'] = [self.new(scores=(6, 6, 6, 6), essay=f'Test {current_index}', source_id=f'te-{current_index}') for current_index in range(3)]
        self.write()
        config = {'tasks': {'transform_datasets': 1}, 'dataset': {'source': 'combined',
                  'selection': dict(enabled=True, train_per_group=2, test_per_group=1, seed=42)}}
        with patch.dict(PATHS, datasets={'source_root': str(self.input), 'prepared': str(self.output)}):
            train, test, _ = runner.load_data(config)
        self.assertEqual((len(train), len(test)), (2, 1))
        config['dataset']['save_intermediate_datasets'] = True
        full_output = self.root/'full-from-main'
        with patch.dict(PATHS, datasets={'source_root': str(self.input), 'prepared': str(full_output)}):
            full_train, full_test, full_manifest = runner.load_data(config)
        self.assertEqual((len(full_train), len(full_test)), (2, 1))
        self.assertTrue(full_manifest['save_intermediate_datasets'])
        self.assertEqual(len(list(full_output.rglob('*.parquet'))), 6)
        original_manifest = file_hash(self.output/'manifest.json')
        self.transformer.selection = dict(enabled=True, train_per_group=-1, test_per_group=1, seed=42)
        with self.assertRaisesRegex(ValueError, 'train_per_group'):
            self.transformer.save_processed(self.transformer.prepare(), self.output)
        self.assertEqual(file_hash(self.output/'manifest.json'), original_manifest)
        self.assertEqual((len(load_prepared(self.output)[0]), len(load_prepared(self.output)[1])), (2, 1))

    def test_invalid_intermediate_export_flag_keeps_existing_release(self) -> None:
        """Verify that invalid intermediate export flag keeps existing release.

        Returns:
            None: This operation returns no value.
        """
        self.rows['chillies']['train'] = [self.old()]
        prepared = self.prepare()
        self.transformer.save_processed(prepared, self.output)
        original_manifest = file_hash(self.output/'manifest.json')
        for invalid in ('false', None, 0, 1):
            self.transformer.save_intermediate_datasets = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'save_intermediate_datasets must be a boolean'):
                self.transformer.save_processed(prepared, self.output)
            self.assertEqual(file_hash(self.output/'manifest.json'), original_manifest)
        self.transformer.save_intermediate_datasets = True
        self.transformer.save_processed(prepared, self.output)
        self.assertEqual(len(load_prepared(self.output)[0]), 1)

    def test_real_samples_match_verified_extraction_cache_and_saved_parquet(self) -> None:
        """Verify that real samples match verified extraction cache and saved parquet.

        Returns:
            None: This operation returns no value.
        """
        transformer = DatasetTransformer(PATHS['datasets'])
        prepared = transformer.prepare(limit_per_split=64)
        baseline = ROOT/'data/datasets/processed/ielts-v2/extracted/writing9.parquet'
        if baseline.is_file():
            expected = pd.read_parquet(baseline, filters=[('source_line', '<=', 64)]).sort_values('essay_id')
            actual = prepared.extracted['writing9'].sort_values('essay_id')
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
        transformer.save_processed(prepared, self.output)
        train, test, manifest = load_prepared(self.output, allow_partial=True)
        self.assertEqual(manifest['partial_input'], True)
        self.assertEqual(len(train)+len(test), int(prepared.rows.status.eq('accepted').sum()))


if __name__ == '__main__':
    unittest.main()
