"""PANDA membership, recovery, artifacts, HTTP boundaries, packaging and real encoder checks."""

import ast
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import socket
import subprocess
import tempfile
import threading
import unittest
from typing import Any
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from TestExperiment import rows
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.Experiment import prepare_experiment
from ielts_judge.panda.Artifacts import adjacent_band_labels, canonical, insights, parse_completion, parse_json, parse_legacy_completion, read_json, sha256, write_json
from offline_research.panda.Contracts import generation_profile, load_result, result_path, row_input
from offline_research.panda.Generation import run_insights
from offline_research.panda.Comparison import run_comparison
from ielts_judge.panda.Corpus import Corpus, Encoder, embedding_profile, vector
from offline_research.panda.Corpus import build_corpus
from ielts_judge.Database import ReferenceRetriever, RetrievalError, SupabaseRPC, import_corpus, retrieval_k


def explanation(record: Any) -> dict[str, Any]:
    """Explanation.

    Args:
        record (Any): Record used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    return {name: {"evidence": record["essay"],
        "why_not_higher": "not_applicable" if record["scores"][name] == 9 else "Evidence does not meet the next descriptor.",
        "why_not_lower": "not_applicable" if record["scores"][name] == 0 else "Evidence meets this descriptor."}
        for name in PUBLIC_CRITERION_IDS}


def batch_generator(generate: Any) -> Any:
    """Adapt existing per-essay test fixtures to the production batch callback.

    Args:
        generate (Any): Generate used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    def generate_batch(records: Any, *current_arguments: Any) -> Any:
        """Generate batch.

        Args:
            records (Any): Records used by this operation.
            current_arguments (Any): Current arguments used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        results = []
        for record in records:
            try:
                results.append(generate(record, *current_arguments))
            except ValueError as error:
                results.append(error)
        return results
    return generate_batch


class FixtureEncoder:
    """Manage fixture encoder through the operations defined by this component."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.profile = embedding_profile()
        self.calls = 0
    def encode(self, texts: Any) -> Any:
        """Encode.

        Args:
            texts (Any): Texts used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        self.calls += 1
        values = np.zeros((len(texts), 768), dtype=np.float32)
        for index, text in enumerate(texts):
            values[index, int(hashlib.sha256(text.encode()).hexdigest()[:4],16) % 768] = 1
        return values, [len(text) for text in texts]


class PandaCorpusTests(unittest.TestCase):
    """Manage panda corpus tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.train, self.test = rows("train", 4), rows("test", 3)
        _, self.experiment = prepare_experiment(self.train, self.test)
        self.profile = generation_profile()
        self.calls = []
    def generate(self, record: Any) -> Any:
        """Generate.

        Args:
            record (Any): Record used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        self.calls.append(record["essay_id"])
        return canonical(explanation(record)), {"stop_reason": "eos"}
    def corpus(self) -> Any:
        """Corpus.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        output = self.path / "generation"
        run_insights(self.train, self.experiment, output, self.profile, batch_generator(self.generate))
        target = self.path / "corpus"
        build_corpus(self.train.iloc[::-1], self.experiment, output, target, embedding_profile(), FixtureEncoder(), batch_size=2)
        return Corpus(target)

    def test_resumes_pilot_and_skips_accepted_even_after_reorder(self) -> None:
        """Verify that resumes pilot and skips accepted even after reorder.

        Returns:
            None: This operation returns no value.
        """
        output = self.path / "run"
        pilot = run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate),limit=2)
        self.assertEqual((pilot["accepted"],pilot["pending"]),(2,2))
        result = run_insights(self.train.iloc[::-1],self.experiment,output,self.profile,batch_generator(self.generate))
        self.assertTrue(result["complete"])
        self.assertEqual(len(self.calls),4)
        run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate))
        self.assertEqual(len(self.calls),4)

    def test_rejects_changed_profile_train_partial_and_stale_response(self) -> None:
        """Verify that rejects changed profile train partial and stale response.

        Returns:
            None: This operation returns no value.
        """
        output = self.path / "run"
        run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate))
        changed = {**self.profile,"seed":1}
        with self.assertRaisesRegex(ValueError,"profile/train"):
            run_insights(self.train,self.experiment,output,changed,batch_generator(self.generate))
        with self.assertRaisesRegex(ValueError,"entire"):
            run_insights(self.train.iloc[:-1],self.experiment,self.path/"partial",self.profile,batch_generator(self.generate))
        first = self.train.to_dict("records")[0]
        path = result_path(output, first["essay_id"])
        data = read_json(path)
        data["input_sha256"] = "b"*64
        write_json(path,data)
        with self.assertRaisesRegex(ValueError,"Stale"):
            load_result(output,first,read_json(output/"generation_manifest.json"))

    def test_failure_is_persisted_and_only_explicitly_retried(self) -> None:
        """Verify that failure is persisted and only explicitly retried.

        Returns:
            None: This operation returns no value.
        """
        output = self.path/"run"
        def failed(record: Any) -> Any:
            """Failed.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return canonical(explanation(record)), {"stop_reason":"length"}
        report = run_insights(self.train,self.experiment,output,self.profile,batch_generator(failed))
        self.assertEqual(report["failed"],4)
        for row in self.train.to_dict("records"):
            saved=read_json(result_path(output,row["essay_id"]))
            self.assertEqual(saved["insights"],explanation(row_input(row)))
            self.assertEqual(saved["error"],"Generation did not finish with EOS")
        run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate))
        self.assertEqual(self.calls,[])
        report = run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate),retry_failed=True)
        self.assertTrue(report["complete"])
        self.assertTrue(all(read_json(result_path(output,row["essay_id"]))["error"] is None
                            for row in self.train.to_dict("records")))

    def test_failed_validation_retains_complete_raw_text_in_corpus(self) -> None:
        """Verify that failed validation retains complete raw text in corpus.

        Returns:
            None: This operation returns no value.
        """
        output=self.path/"invalid-evidence"
        def generate(record: Any) -> Any:
            """Generate.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            value=explanation(record)
            value[PUBLIC_CRITERION_IDS[0]]["evidence"]="An invented quote absent from the essay."
            return canonical(value),{"stop_reason":"eos"}
        report=run_insights(self.train,self.experiment,output,self.profile,batch_generator(generate))
        self.assertEqual(report["failed"],4)
        for row in self.train.to_dict("records"):
            saved=read_json(result_path(output,row["essay_id"]))
            self.assertEqual(saved["insights"],parse_json(saved["completion"]))
            self.assertIn("not an exact essay substring",saved["error"])
            self.assertEqual(saved["status"],"failed")
        encoder=FixtureEncoder()
        target=self.path/"raw-corpus"
        build_corpus(self.train,self.experiment,output,target,embedding_profile(),encoder)
        corpus=Corpus(target)
        for _,record in corpus.records():
            saved=read_json(result_path(output,record["essay_id"]))
            self.assertEqual(record["insights"],saved["completion"])
            self.assertEqual(record["generation_error"],saved["error"])
            self.assertEqual(record["generation_status"],"failed")
        self.assertEqual(corpus.manifest["count"],len(self.train))

    def test_unparseable_completion_keeps_raw_error_and_null_insights(self) -> None:
        """Verify that unparseable completion keeps raw error and null insights.

        Returns:
            None: This operation returns no value.
        """
        output=self.path/"unparseable"
        report=run_insights(self.train,self.experiment,output,self.profile,
                           batch_generator(lambda record:('{"unfinished":',{"stop_reason":"eos"})))
        self.assertEqual(report["failed"],4)
        for row in self.train.to_dict("records"):
            saved=read_json(result_path(output,row["essay_id"]))
            self.assertEqual(saved["completion"],'{"unfinished":')
            self.assertIsNone(saved["insights"])
            self.assertTrue(saved["error"])

        target=self.path/"partial-corpus"
        build_corpus(self.train,self.experiment,output,target,embedding_profile(),FixtureEncoder())
        self.assertTrue(all(record["insights"]=='{"unfinished":' for _,record in Corpus(target).records()))

    def test_failed_adjacent_text_keeps_recognized_fields_without_accepting_or_inventing(self) -> None:
        """Verify that failed adjacent text keeps recognized fields without accepting or inventing.

        Returns:
            None: This operation returns no value.
        """
        profile=generation_profile({"prompt_variant":"adjacent_text"})
        for variant in ("separator","numbers","heading","no-marker","nested","partial"):
            output=self.path/variant
            def generate(record: Any) -> Any:
                """Generate.

                Args:
                    record (Any): Record used by this operation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                text="\n\n".join(name.replace("_"," ").title()+"\n"+"\n".join(
                    f"- {caption}: {explanation(record)[name][field]}"
                    for field,caption in adjacent_band_labels(score).items())
                    for name,score in record["scores"].items())
                if variant=="separator":text=text.replace("\n\n","\n\n---\n\n")
                elif variant=="numbers":
                    score=next(iter(record["scores"].values()))
                    text=text.replace(f"band {score} rather than band {score-1}",f"band {score-1} rather than band {score}",1)
                elif variant=="heading":text=text.replace("Task Response",f"### Task Response (Band {record['scores']['task_response']})",1)
                elif variant=="no-marker":text=text.replace("\n- ","\n")
                elif variant=="nested":text=text.replace("- Evidence: "+record["essay"],"- Evidence:\n  - "+record["essay"],1)
                else:text=text.replace("- Evidence:","- Other:",1)
                return text,{"stop_reason":"eos"}
            with self.subTest(variant=variant):
                report=run_insights(self.train,self.experiment,output,profile,batch_generator(generate))
                self.assertEqual(report["failed"],4)
                for row in self.train.to_dict("records"):
                    saved=read_json(result_path(output,row["essay_id"]))
                    expected=explanation(row_input(row))
                    if variant=="partial":expected["task_response"].pop("evidence")
                    self.assertEqual(saved["insights"],expected)
                    self.assertTrue(saved["error"])
                    self.assertEqual(saved["status"],"failed")

    def test_process_interruption_keeps_prior_atomic_results(self) -> None:
        """Verify that process interruption keeps prior atomic results.

        Returns:
            None: This operation returns no value.
        """
        output = self.path/"run"
        self.profile = {**self.profile, "batch_size": 1}
        def interrupted(record: Any) -> Any:
            """Interrupted.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            if self.calls:
                raise KeyboardInterrupt()
            return self.generate(record)
        with self.assertRaises(KeyboardInterrupt):
            run_insights(self.train,self.experiment,output,self.profile,batch_generator(interrupted))
        self.assertEqual(len(list((output/"responses").glob("*.json"))),1)
        run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate))
        self.assertEqual(len(self.calls),4)

    def test_strict_json_criteria_quotes_scores_and_boundaries(self) -> None:
        """Verify that strict json criteria quotes scores and boundaries.

        Returns:
            None: This operation returns no value.
        """
        record = row_input(self.train.to_dict("records")[0])
        for invalid in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError): parse_json(invalid)
        for mutate in (lambda sample:sample.pop(PUBLIC_CRITERION_IDS[0]),
                       lambda sample:sample[PUBLIC_CRITERION_IDS[0]].update(evidence="invented quote"),
                       lambda sample:sample[PUBLIC_CRITERION_IDS[0]].update(why_not_higher="not_applicable")):
            bad = explanation(record)
            mutate(bad)
            with self.assertRaises(ValueError): insights(bad,record["essay"],record["scores"])
        for value in (True,6.5,float("nan"),10,-1):
            labels = {**record["scores"],PUBLIC_CRITERION_IDS[0]:value}
            with self.assertRaises(ValueError): insights(explanation(record),record["essay"],labels)
        record["scores"] = {name:0 if current_index%2 else 9 for current_index,name in enumerate(PUBLIC_CRITERION_IDS)}
        self.assertEqual(insights(explanation(record),record["essay"],record["scores"]),explanation(record))

    def test_complete_outer_fence_is_accepted_without_rewriting_raw_response(self) -> None:
        """Verify that complete outer fence is accepted without rewriting raw response.

        Returns:
            None: This operation returns no value.
        """
        output = self.path / "fenced"
        def fenced(record: Any) -> Any:
            """Fenced.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return "```json\n" + canonical(explanation(record)) + "\n```", {"stop_reason": "eos"}
        report = run_insights(self.train,self.experiment,output,self.profile,batch_generator(fenced))
        self.assertTrue(report["complete"])
        first = self.train.to_dict("records")[0]
        result = load_result(output,first,read_json(output/"generation_manifest.json"))
        assert result is not None
        self.assertTrue(result["completion"].startswith("```json\n"))
        for text in ('before\n```json\n{}\n```', '```json\n{}', '```json\n{"x":1,"x":2}\n```',
                     '```json\n{"x":NaN}\n```', '```json\n{}\n```\nafter', '{}{}'):
            with self.subTest(text=text), self.assertRaises(ValueError): parse_completion(text)

    def test_placeholder_spellings_rejected_in_every_required_field(self) -> None:
        """Verify that placeholder spellings rejected in every required field.

        Returns:
            None: This operation returns no value.
        """
        record = row_input(self.train.to_dict("records")[0])
        record["scores"] = {name:4 for name in PUBLIC_CRITERION_IDS}
        for value in ("not_applicable", "Not applicable", "not applicable", "NOT-APPLICABLE.", "N/A", "n_a", "None", "null!"):
            for field in ("evidence", "why_not_higher", "why_not_lower"):
                bad = explanation(record)
                bad[PUBLIC_CRITERION_IDS[0]][field] = value
                with self.subTest(value=value,field=field), self.assertRaises(ValueError):
                    insights(bad,record["essay"],record["scores"])
        record["scores"] = {name:0 for name in PUBLIC_CRITERION_IDS}
        bad = explanation(record)
        bad[PUBLIC_CRITERION_IDS[0]]["why_not_lower"] = "Not applicable"
        with self.assertRaises(ValueError): insights(bad,record["essay"],record["scores"])

    def legacy_completion(self, record: Any) -> Any:
        """Legacy completion.

        Args:
            record (Any): Record used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return "\n\n".join(name.replace("_", " ").title() + "\n- Evidence: A description of the essay.\n"
            "- Why not higher (score + 1): " + item["why_not_higher"] + "\n"
            "- Why not lower (score - 1): " + item["why_not_lower"]
            for name,item in explanation(record).items())

    def selection(self, ids: Any) -> dict[str, Any]:
        """Selection.

        Args:
            ids (Any): Ids used by this operation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        return {"schema_version":"panda-comparison-selection-v1",
            "dataset_manifest_sha256":self.experiment["dataset_manifest_sha256"],
            "train_records_sha256":self.experiment["train"]["records_sha256"], "essay_ids":ids}

    def test_comparison_pairs_same_ids_settings_and_resumes_without_advancing(self) -> None:
        """Verify that comparison pairs same ids settings and resumes without advancing.

        Returns:
            None: This operation returns no value.
        """
        ids = list(self.train.essay_id)[:2]
        calls = []
        def generate(record: Any,variant: Any) -> Any:
            """Generate.

            Args:
                record (Any): Record used by this operation.
                variant (Any): Variant used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            calls.append((record["essay_id"],variant))
            text = self.legacy_completion(record) if variant == "legacy_text" else canonical(explanation(record))
            return text,{"stop_reason":"eos"}
        output = self.path / "compare"
        first = run_comparison(self.train,self.experiment,output,self.profile,self.selection(ids),batch_generator(generate),limit=1)
        self.assertTrue(all(current_record["pending"]==1 for current_record in first["variants"].values()))
        report = run_comparison(self.train.iloc[::-1],self.experiment,output,self.profile,self.selection(ids),batch_generator(generate))
        self.assertTrue(all(current_record["complete"] for current_record in report["variants"].values()))
        self.assertIsNone(report["semantic_winner"])
        run_comparison(self.train,self.experiment,output,self.profile,self.selection(ids),batch_generator(generate))
        self.assertEqual(set(calls),{(current_index,current_value) for current_index in ids for current_value in ("simple_json","legacy_text")})
        self.assertEqual(len(calls),4)
        manifest = read_json(output/"comparison_manifest.json")
        array,current_band = manifest["profiles"].values()
        for key in ("model_id","revision","max_input_tokens","max_new_tokens","seed","do_sample","dtype","rubric_sha256"):
            self.assertEqual(array[key],current_band[key])
        pairs = read_json(output/"paired_review.json")["records"]
        self.assertEqual([row["essay_id"] for row in pairs],ids)
        self.assertTrue(all(set(row["variants"])=={"simple_json","legacy_text"} for row in pairs))
        with self.assertRaisesRegex(ValueError,"changed"):
            run_comparison(self.train,self.experiment,output,self.profile,self.selection(ids[::-1]),batch_generator(generate))
        with self.assertRaisesRegex(ValueError,"Comparison directory"):
            run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate))
        with self.assertRaisesRegex(ValueError,"differs"):
            build_corpus(self.train,self.experiment,output/"simple_json",self.path/"corpus",embedding_profile(),FixtureEncoder())
        with self.assertRaisesRegex(ValueError,"differs from the experiment"):
            build_corpus(self.train,self.experiment,output/"legacy_text",self.path/"legacy-corpus",embedding_profile(),FixtureEncoder())

    def test_comparison_rejects_test_ids_duplicates_and_another_release(self) -> None:
        """Verify that comparison rejects test ids duplicates and another release.

        Returns:
            None: This operation returns no value.
        """
        for selection in (self.selection([self.test.essay_id.iloc[0]]), self.selection([self.train.essay_id.iloc[0]]*2),
                          {**self.selection(list(self.train.essay_id)[:1]),"train_records_sha256":"f"*64}):
            with self.assertRaises(ValueError):
                run_comparison(self.train,self.experiment,self.path/"compare",self.profile,selection,batch_generator(self.generate))
        self.assertEqual(self.calls,[])
        self.assertFalse((self.path/"compare").exists())

    def test_legacy_text_parser_preserves_prose_evidence_and_rejects_missing_or_duplicate_fields(self) -> None:
        """Verify that legacy text parser preserves prose evidence and rejects missing or duplicate fields.

        Returns:
            None: This operation returns no value.
        """
        record = row_input(self.train.to_dict("records")[0])
        text = self.legacy_completion(record)
        parsed = parse_legacy_completion(text,record["scores"])
        insights(parsed,record["essay"],record["scores"],require_quote=False)
        with self.assertRaises(ValueError): insights(parsed,record["essay"],record["scores"])
        for bad in (text+"\n"+text, text.replace("- Evidence:","- Other:",1)):
            with self.assertRaises(ValueError): parse_legacy_completion(bad,record["scores"])

    def test_simple_prompt_defines_direction_and_only_actual_boundary_placeholders(self) -> None:
        """Verify that simple prompt defines direction and only actual boundary placeholders.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        generator = PandaInsightGenerator("")
        prompt = generator.build_insight_prompt("Essay",np.array([0,1,8,9]),"Task",structured=True,simple=True)
        template = parse_json(prompt.split("OUTPUT TEMPLATE:\n")[1].split("\nADJACENT DESCRIPTORS")[0])
        self.assertEqual(template["task_response"]["why_not_lower"],"not_applicable")
        self.assertEqual(template["grammatical_range_and_accuracy"]["why_not_higher"],"not_applicable")
        self.assertEqual(sum(current_value=="not_applicable" for item in template.values() for current_value in item.values()),2)
        self.assertIn("demonstrated strength",prompt)
        self.assertIn("never TASK",prompt)
        legacy = generator.build_insight_prompt("Essay",np.array([0,1,8,9]),"Task")
        # Golden text obtained independently from the implementation at HEAD eabda93.
        self.assertEqual(sha256(legacy),"a2c56f93c63758ad7ba7ca54bba4b5a46bb74c04c3d66a9f18ba9ef6e04279f6")
        self.assertIn("For EACH criterion output EXACTLY 3 bullet points:",legacy)
        self.assertNotIn("OUTPUT TEMPLATE",legacy)
        with self.assertRaisesRegex(ValueError,"prompt_variant"): generation_profile({"prompt_variant":"typo"})

    def test_incomplete_insights_fail_before_embedding(self) -> None:
        """Verify that incomplete insights fail before embedding.

        Returns:
            None: This operation returns no value.
        """
        output=self.path/"run"
        run_insights(self.train,self.experiment,output,self.profile,batch_generator(self.generate),limit=1)
        encoder=FixtureEncoder()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            build_corpus(self.train,self.experiment,output,self.path/"corpus",embedding_profile(),encoder)
        self.assertEqual(encoder.calls,0)

    def test_original_short_prompt_matches_historical_085e4a1(self) -> None:
        """Verify that original short prompt matches historical 085e4a1.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        generator=PandaInsightGenerator("")
        prompt=generator.build_insight_prompt("Essay",np.array([0,1,8,9]),"Task",original=True)
        # Independently evaluated from the method in commit 085e4a1.
        self.assertEqual(sha256(prompt),"7e77c4b059dae4beb01fb7067af2a3afcdee5323a1e99d4517e1de257807aec1")
        self.assertEqual(prompt,generator.build_insight_prompt("Essay",np.array([0,1,8,9]),"Another task",original=True))
        self.assertIn("Scores given by expert:",prompt)
        self.assertNotIn("Original task:",prompt)
        self.assertNotIn("OUTPUT TEMPLATE",prompt)

    def test_explicit_band_prompt_substitutes_each_source_score_and_handles_boundaries(self) -> None:
        """Verify that explicit band prompt substitutes each source score and handles boundaries.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        generator=PandaInsightGenerator("")
        prompt=generator.build_insight_prompt("Essay",np.array([0,1,6,9]),explicit_bands=True)
        template=prompt.split("For EACH criterion output EXACTLY 3 bullet points:\n")[1].split("\n\n                Essay:\n")[0]
        template="\n".join(line[16:] if line.startswith(" "*16) else line for line in template.splitlines())
        sections=template.split("\n\n")
        self.assertIn("- Strength supporting band 0 (no lower band): not_applicable",sections[0])
        self.assertIn("- Weakness preventing band 1:",sections[0])
        self.assertIn("- Strength supporting band 1 rather than band 0:",sections[1])
        self.assertIn("- Weakness preventing band 2:",sections[1])
        self.assertIn("- Strength supporting band 6 rather than band 5:",sections[2])
        self.assertIn("- Weakness preventing band 7:",sections[2])
        self.assertIn("- Strength supporting band 9 rather than band 8:",sections[3])
        self.assertIn("- Weakness preventing a higher band (9 is maximum): not_applicable",sections[3])
        self.assertNotIn("band -1",template)
        self.assertNotIn("band 10",template)
        self.assertNotIn("score + 1",template)
        for band in range(1,9):
            labels=adjacent_band_labels(band)
            self.assertIn(f"band {band} rather than band {band-1}",labels["why_not_lower"])
            self.assertEqual(labels["why_not_higher"],f"Weakness preventing band {band+1}")
        for invalid in (True,6.5,-1,10):
            with self.assertRaises(ValueError):adjacent_band_labels(invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):generator.build_insight_prompt("Essay",np.array([1,2,6.5,9]),explicit_bands=True)

    def test_adjacent_prompt_preserves_original_outside_explanation_and_before_answering(self) -> None:
        """Verify that adjacent prompt preserves original outside explanation and before answering.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        generator=PandaInsightGenerator("")
        for values in ([0,1,8,9],[2,4,6,7],*[([band]*4) for band in range(10)]):
            original=generator.build_insight_prompt("Essay",np.array(values),original=True)
            adjacent=generator.build_insight_prompt("Essay",np.array(values),explicit_bands=True)
            check_start=adjacent.index("                Before answering:\n")
            check_end=adjacent.index("                Scores given by expert:\n",check_start)
            adjacent=adjacent[:check_start]+adjacent[check_end:]
            block_start=original.index("                [Criterion name]\n")
            block_end=original.index("\n\n                Essay:\n",block_start)
            self.assertTrue(adjacent.startswith(original[:block_start]),values)
            self.assertTrue(adjacent.endswith(original[block_end:]),values)

    def test_adjacent_task_is_inserted_before_essay_without_other_prompt_changes(self) -> None:
        """Verify that adjacent task is inserted before essay without other prompt changes.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        generator=PandaInsightGenerator("")
        for values in ([0,1,8,9],[2,4,6,7]):
            for task in ("Discuss both views and give your opinion.","First task line.\nSecond task line."):
                essay="Essay text with Task: and Essay: inside it."
                baseline=generator.build_insight_prompt(essay,np.array(values),explicit_bands=True)
                actual=generator.build_insight_prompt(essay,np.array(values),task,explicit_bands=True)
                expected=baseline.replace("                Essay:\n",
                    f"                Essay task (use ONLY for Task Response evaluation):\n                {task}\n\n                Essay:\n",1)
                self.assertEqual(actual,expected)
                self.assertEqual(generator.build_insight_prompt(essay,np.array(values),task,original=True),
                                 generator.build_insight_prompt(essay,np.array(values),original=True))

    def test_explicit_band_parser_maps_strength_and_weakness_and_rejects_wrong_numbers(self) -> None:
        """Verify that explicit band parser maps strength and weakness and rejects wrong numbers.

        Returns:
            None: This operation returns no value.
        """
        labels=dict(zip(PUBLIC_CRITERION_IDS,[0,1,6,9]))
        captions=[("Strength supporting band 0 (no lower band)","Weakness preventing band 1"),
            ("Strength supporting band 1 rather than band 0","Weakness preventing band 2"),
            ("Strength supporting band 6 rather than band 5","Weakness preventing band 7"),
            ("Strength supporting band 9 rather than band 8","Weakness preventing a higher band (9 is maximum)")]
        text="\n\n".join(name.replace("_"," ").title()+"\n- Evidence: Essay\n"
            f"- {strength}: "+("not_applicable" if band==0 else "Demonstrated strength.")+"\n"
            f"- {weakness}: "+("not_applicable" if band==9 else "Specific limiting weakness.")
            for (name,band),(strength,weakness) in zip(labels.items(),captions))
        parsed=parse_legacy_completion(text,labels,explicit_bands=True)
        insights(parsed,"Essay",labels,require_quote=False)
        self.assertEqual(parsed["lexical_resource"]["why_not_lower"],"Demonstrated strength.")
        self.assertEqual(parsed["lexical_resource"]["why_not_higher"],"Specific limiting weakness.")
        duplicate_field=text.replace("- Evidence: Essay","- Evidence: Essay\n- Evidence: Different text",1)
        for duplicate in (duplicate_field,text+"\n"+text):
            self.assertEqual(parse_legacy_completion(duplicate,labels,explicit_bands=True,allow_partial=True),parsed)
        for bad in (text.replace("band 6 rather than band 5","band 5 rather than band 4"),
                    text.replace("Weakness preventing band 7","Weakness preventing band 8"),
                    text.replace("- Evidence:","- Other:",1),duplicate_field,text+"\n"+text):
            with self.assertRaises(ValueError):parse_legacy_completion(bad,labels,explicit_bands=True)

    def test_adjacent_text_resume_and_verbatim_corpus(self) -> None:
        """Verify that adjacent text resume and verbatim corpus.

        Returns:
            None: This operation returns no value.
        """
        profile=generation_profile({"prompt_variant":"adjacent_text"})
        self.assertEqual(profile["prompt_version"],"panda-insights-adjacent-text-v5")
        def generate(record: Any) -> Any:
            """Generate.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return "\n\n".join(name.replace("_"," ").title()+"\n"+"\n".join(
                f"- {caption}: {explanation(record)[name][field]}"
                for field,caption in adjacent_band_labels(score).items()) for name,score in record["scores"].items()),{"stop_reason":"eos"}
        output=self.path/"adjacent"
        report=run_insights(self.train,self.experiment,output,profile,batch_generator(generate))
        self.assertEqual(report["accepted"],len(self.train))
        again=run_insights(self.train,self.experiment,output,profile,batch_generator(generate))
        self.assertEqual(again["attempted_this_invocation"],0)
        with self.assertRaisesRegex(ValueError,"changed"):
            run_insights(self.train,self.experiment,output,generation_profile({"prompt_variant":"original_text"}),batch_generator(generate))
        target=self.path/"corpus"
        build_corpus(self.train,self.experiment,output,target,embedding_profile(),FixtureEncoder())
        for _,record in Corpus(target).records():
            self.assertEqual(record["insights"],read_json(result_path(output,record["essay_id"]))["completion"])

    def test_original_text_full_train_can_build_raw_corpus(self) -> None:
        """Verify that original text full train can build raw corpus.

        Returns:
            None: This operation returns no value.
        """
        profile=generation_profile({"prompt_variant":"original_text"})
        self.assertEqual(profile["prompt_version"],"panda-insights-original-085e4a1-v1")
        output=self.path/"original"
        calls=[]
        def generate(record: Any) -> Any:
            """Generate.

            Args:
                record (Any): Record used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            calls.append(record["essay_id"])
            return self.legacy_completion(record),{"stop_reason":"eos"}
        report=run_insights(self.train,self.experiment,output,profile,batch_generator(generate))
        self.assertEqual(report["accepted"],len(self.train))
        run_insights(self.train,self.experiment,output,profile,batch_generator(generate))
        self.assertEqual(len(calls),len(self.train))
        encoder=FixtureEncoder()
        manifest=build_corpus(self.train,self.experiment,output,self.path/"corpus",embedding_profile(),encoder)
        self.assertEqual(manifest["schema_version"],"panda-corpus-v2")
        self.assertGreater(encoder.calls,0)

    def test_corpus_roundtrip_hashes_labels_and_content_binding(self) -> None:
        """Verify that corpus roundtrip hashes labels and content binding.

        Returns:
            None: This operation returns no value.
        """
        corpus=self.corpus()
        records=[record for _,record in corpus.records()]
        self.assertEqual([record["essay_id"] for record in records],sorted(self.train.essay_id))
        self.assertEqual(corpus.manifest["count"],4)
        self.assertEqual(len(records[0]["criteria"]),4)
        path=corpus.path/"records.jsonl"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"hash mismatch"): Corpus(corpus.path)

    def test_embedding_profile_and_vector_validation(self) -> None:
        """Verify that embedding profile and vector validation.

        Returns:
            None: This operation returns no value.
        """
        for config in ({"revision":"main"},{"max_length":513},{"max_length":True},{"model_id":"unknown"},{"typo":1}):
            with self.assertRaises(ValueError): embedding_profile(config)
        for value in ([0.0]*768,[float("nan")]+[0.0]*767,[True]+[0.0]*767,[1.0,0.0]):
            with self.assertRaises(ValueError): vector(value)

    def test_runtime_profiles_reject_mismatch_without_network(self) -> None:
        """Verify that runtime profiles reject mismatch without network.

        Returns:
            None: This operation returns no value.
        """
        corpus=self.corpus()
        encoder=FixtureEncoder()
        encoder.profile={**encoder.profile,"revision":"b"*40}
        with self.assertRaisesRegex(ValueError,"mismatch"): ReferenceRetriever(None,corpus.manifest,encoder)

    def test_transport_uses_real_http_and_handles_failure_then_recovery(self) -> None:
        """Verify that transport uses real http and handles failure then recovery.

        Returns:
            None: This operation returns no value.
        """
        state={"status":200,"body":{"ok":True},"headers":None}
        class Handler(BaseHTTPRequestHandler):
            """Manage handler through the operations defined by this component."""
            def log_message(self,format: Any,*current_arguments: Any) -> None:
                """Log message.

                Args:
                    format (Any): Format used by this operation.
                    current_arguments (Any): Current arguments used by this operation.

                Returns:
                    None: This operation returns no value.
                """
                pass
            def do_POST(self) -> None:
                """Do POST.

                Returns:
                    None: This operation returns no value.
                """
                state["headers"]=dict(self.headers)
                state["request"]=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(state["status"])
                self.end_headers()
                body=state["body"]
                self.wfile.write((body if isinstance(body,str) else canonical(body)).encode())
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        rpc=SupabaseRPC(f"http://127.0.0.1:{server.server_port}","sb_publishable_test")
        self.assertEqual(rpc.call("panda_match",{"value":1}),{"ok":True})
        self.assertNotIn("Authorization",state["headers"])
        for code in (301,401,403,500):
            state["status"]=code
            with self.assertRaisesRegex(RetrievalError,f"HTTP {code}"): rpc.call("panda_match",{})
        state.update(status=200,body="not JSON")
        with self.assertRaisesRegex(RetrievalError,"JSON"):rpc.call("panda_match",{})
        state["body"]={"ok":True}
        self.assertEqual(rpc.call("panda_match",{}),{"ok":True})
        for url in ("http://example.com","https://x.test/a","https://user:pass@x.test","https://x.test?secret=x"):
            with self.assertRaises(ValueError): SupabaseRPC(url,"key")

    def test_real_connection_failure_is_not_empty_retrieval(self) -> None:
        """Verify that real connection failure is not empty retrieval.

        Returns:
            None: This operation returns no value.
        """
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1",0))
            port=reserved.getsockname()[1]
            # Bound but not listening: the real HTTP connection is refused.
            with self.assertRaisesRegex(RetrievalError,"network/timeout"):
                SupabaseRPC(f"http://127.0.0.1:{port}","key",timeout=1).call("panda_match",{})

    def test_import_requires_complete_acknowledgement_and_matching_final_digest(self) -> None:
        """Verify that import requires complete acknowledgement and matching final digest.

        Returns:
            None: This operation returns no value.
        """
        corpus=self.corpus()
        manifest=corpus.manifest
        class RPC:
            """Manage rpc through the operations defined by this component."""
            def __init__(self) -> None:
                """Initialize this component's settings and collaborators.

                Returns:
                    None: This operation returns no value.
                """
                self.batches=[];self.bad=False
            def call(self,name: Any,params: Any) -> dict[str, Any]:
                """Call.

                Args:
                    name (Any): Name used by this operation.
                    params (Any): Params used by this operation.

                Returns:
                    dict[str, Any]: Named result fields and metadata produced by this operation.
                """
                if name=="panda_begin":return {"corpus_id":manifest["corpus_id"],"state":"draft"}
                if name=="panda_import":
                    self.batches.append(params["p_records"])
                    return {"count":len(params["p_records"])-(1 if self.bad else 0)}
                return {"corpus_id":manifest["corpus_id"],"state":"ready","count":manifest["count"],"records_digest":manifest["records_digest"]}
        rpc=RPC()
        self.assertEqual(import_corpus(corpus,rpc,batch_size=3)["state"],"ready")
        self.assertEqual([len(current_band) for current_band in rpc.batches],[3,1])
        rpc.bad=True
        with self.assertRaisesRegex(RetrievalError,"acknowledge"):import_corpus(corpus,rpc,batch_size=3)

    def test_real_bundle_unpacks_and_audits_in_a_fresh_process(self) -> None:
        """Verify that real bundle unpacks and audits in a fresh process.

        Returns:
            None: This operation returns no value.
        """
        release=ROOT/"data/datasets/processed/ielts-v3"
        if not release.is_dir():self.skipTest("Real prepared release is not installed")
        spec=importlib.util.spec_from_file_location("panda_builder_real",ROOT/"code/offline_research/tools/BuildPandaKaggle.py")
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        archive_path=module.build(self.path/"bundle")
        notebook=json.loads((archive_path.parent/"PandaGeneration.ipynb").read_text(encoding="utf-8"))
        scope={}
        exec(notebook["cells"][1]["source"],scope)
        bundle_manifest=read_json(archive_path.parent/"bundle_manifest.json")
        unpacked=scope["prepare_project"](archive_path.parent,self.path/"unpacked",expected_sha256=bundle_manifest["archive_sha256"],expected_files=bundle_manifest["files"])
        result=subprocess.run([sys.executable,"-B","code/offline_research/ManageCorpus.py","audit","--output","audit"],cwd=unpacked,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(read_json(unpacked/"audit/audit.json")["count"],8236)

    def test_retriever_validates_response_empty_self_hash_and_versions(self) -> None:
        """Verify that retriever validates response empty self hash and versions.

        Returns:
            None: This operation returns no value.
        """
        corpus=self.corpus()
        records=[row for _,row in corpus.records()]
        row=records[0]
        entry={key:row[key] for key in ("essay_id","essay_hash","task_prompt","essay")}
        entry.update(scores={current_key:current_value["score"] for current_key,current_value in row["criteria"].items()},
                     insights=row["insights"],similarity=0.8)
        class RPC:
            """Manage rpc through the operations defined by this component."""
            def __init__(self) -> None:
                """Initialize this component's settings and collaborators.

                Returns:
                    None: This operation returns no value.
                """
                self.response={}
            def call(self,*current_arguments: Any) -> Any:
                """Call.

                Args:
                    current_arguments (Any): Current arguments used by this operation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                self.parameters=current_arguments[1]; return self.response
        rpc=RPC()
        rpc.response={"corpus_id":corpus.manifest["corpus_id"],"embedding_profile_sha256":corpus.manifest["embedding_profile_sha256"],"count":4,"references":[entry]}
        retriever=ReferenceRetriever(rpc,corpus.manifest,FixtureEncoder())
        self.assertEqual(retriever.retrieve("A new unrelated essay")["references"][0].essay_id,row["essay_id"])
        self.assertEqual(rpc.parameters["p_match_count"],1)
        self.assertEqual(retriever.retrieve("Another essay",reference_count=2)["references"][0].insights,row["insights"])
        self.assertEqual(rpc.parameters["p_match_count"],2)
        rpc.response["references"][0]["insights"]={}
        with self.assertRaisesRegex(RetrievalError,"completion"):retriever.retrieve("New text")
        rpc.response["references"][0]["insights"]=row["insights"]
        with self.assertRaisesRegex(RetrievalError,"hash"):retriever.retrieve(row["essay"],essay_id="new-source-id")
        with self.assertRaisesRegex(RetrievalError,"self"):retriever.retrieve("New essay",essay_id=row["essay_id"])
        rpc.response["references"]=[]
        self.assertEqual(retriever.retrieve("New essay")["status"],"retrieval_empty")
        rpc.response["corpus_id"]="b"*64
        with self.assertRaisesRegex(RetrievalError,"version"):retriever.retrieve("New essay")

    def test_reference_count_is_read_from_config_and_validated(self) -> None:
        """Verify that reference count is read from config and validated.

        Returns:
            None: This operation returns no value.
        """
        project=self.path/"project"
        (project/"code/shared").mkdir(parents=True)
        path=project/"code/shared/config.yaml"
        with patch("ielts_judge.Database.PROJECT_ROOT",project):
            for value in (1,3,20):
                path.write_text(f"retrieval:\n  k: {value}\n")
                self.assertEqual(retrieval_k(),value)
            for value in ("true","0","21","1.5"):
                path.write_text(f"retrieval:\n  k: {value}\n")
                with self.assertRaises(ValueError):retrieval_k()

    def test_real_sentence_transformer_encoder_lengths_and_normalization(self) -> None:
        # Real tokenizer/transformer/pooling/dense inference with local random weights, no downloads.
        """Verify that real sentence transformer encoder lengths and normalization.

        Returns:
            None: This operation returns no value.
        """
        import torch
        from transformers import BertConfig, BertModel
        from transformers.models.bert.tokenization_bert import BertTokenizer
        from sentence_transformers import SentenceTransformer, models
        model_dir=self.path/"tiny"
        model_dir.mkdir()
        (model_dir/"vocab.txt").write_text("\n".join(["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","unique","essay","train","new"]))
        BertTokenizer(vocab_file=str(model_dir/"vocab.txt")).save_pretrained(model_dir)
        BertModel(BertConfig(vocab_size=9,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,intermediate_size=24,max_position_embeddings=32)).save_pretrained(model_dir,safe_serialization=False)
        tiny=SentenceTransformer(modules=[models.Transformer(str(model_dir)),models.Pooling(12),models.Dense(12,768,activation_function=torch.nn.Identity())],device="cpu")
        profile=embedding_profile({"max_length":8})
        with patch("sentence_transformers.SentenceTransformer",return_value=tiny): encoder=Encoder(profile)
        embeddings,lengths=encoder.encode(["new essay","unique "*20])
        self.assertGreater(lengths[1],8)
        self.assertEqual(embeddings.shape,(2,768))
        for row in embeddings: vector(row)
        del encoder, tiny
        import gc
        gc.collect()

    def test_corpus_recomputed_hash_cannot_hide_changed_task(self) -> None:
        """Verify that corpus recomputed hash cannot hide changed task.

        Returns:
            None: This operation returns no value.
        """
        corpus=self.corpus()
        records=[record for _,record in corpus.records()]
        records[0]["task_prompt"]="A changed task under the same essay ID"
        lines=[canonical(record) for record in records]
        (corpus.path/"records.jsonl").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
        manifest=deepcopy(corpus.manifest)
        manifest["records_sha256"]=sha256("\n".join(lines)+"\n")
        manifest["records_digest"]=sha256("".join(current_record["essay_id"]+":"+sha256(line)+"\n" for current_record,line in zip(records,lines)))
        identity={current_key:current_value for current_key,current_value in manifest.items() if current_key not in {"corpus_id","count","embedding_profile_sha256","truncated_count"}}
        manifest["corpus_id"]=sha256(canonical(identity))
        write_json(corpus.path/"manifest.json",manifest)
        with self.assertRaisesRegex(ValueError,"entire experiment train"):Corpus(corpus.path)

    def test_hf_completion_slices_tokens_and_rejects_input_over_budget(self) -> None:
        """Verify that hf completion slices tokens and rejects input over budget.

        Returns:
            None: This operation returns no value.
        """
        import torch
        from types import SimpleNamespace
        from transformers import BatchEncoding
        from offline_research.panda.Generation import HFInsightGenerator
        from offline_research.panda.InsightGenerator import PandaInsightGenerator
        record=row_input(self.train.to_dict("records")[0])
        result=canonical(explanation(record))
        case=self
        prompts=[]
        class Tokenizer:
            """Manage tokenizer through the operations defined by this component."""
            pad_token_id=0
            def apply_chat_template(self,messages: list[dict[str, str]],**keyword_arguments: Any) -> str:
                """Apply chat template.

                Args:
                    messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    str: The computed result for the supplied inputs.
                """
                prompts.append(messages[0]["content"])
                return "formatted input"
            def __call__(self,*current_arguments: Any,**keyword_arguments: Any) -> Any:
                """Call.

                Args:
                    current_arguments (Any): Current arguments used by this operation.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                case.assertFalse(keyword_arguments["truncation"])
                return BatchEncoding({"input_ids":torch.tensor([[1,2,3,4,5,6]])})
            def decode(self,tokens: Any,**keyword_arguments: Any) -> Any:
                """Decode.

                Args:
                    tokens (Any): Tokens used by this operation.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                case.assertEqual(list(tokens),[8,9])
                return result
            def pad(self,features: Any,**keyword_arguments: Any) -> Any:
                """Pad.

                Args:
                    features (Any): Features used by this operation.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                return BatchEncoding({"input_ids":torch.stack([item["input_ids"] for item in features])})
        model=SimpleNamespace(device="cpu",config=SimpleNamespace(max_position_embeddings=100),
            generation_config=SimpleNamespace(eos_token_id=9),
            generate=lambda **keyword_arguments:torch.tensor([[1,2,3,4,5,6,8,9]]),eval=lambda:None)
        generator=HFInsightGenerator({**self.profile,"max_input_tokens":8,"max_new_tokens":10})
        runtime=SimpleNamespace(tokenizer=Tokenizer(),model=model,
            build_insight_prompt=PandaInsightGenerator("").build_insight_prompt,
            load_model_and_tokenizer=lambda model_id,revision:None)
        with patch.dict("os.environ",{"HF_TOKEN":"hf_test_not_a_real_token"}), patch("torch.cuda.is_available",return_value=True), patch("offline_research.panda.InsightGenerator.PandaInsightGenerator",return_value=runtime) as constructor:
            completion,diagnostics=generator(record)
            _,other=generator(record,prompt_variant="legacy_text")
            _,original=generator(record,prompt_variant="original_text")
            constructor.assert_called_once_with("hf_test_not_a_real_token")
            self.assertEqual(diagnostics["seed"],other["seed"])
            self.assertEqual(diagnostics["seed"],original["seed"])
            self.assertEqual(prompts[-1],runtime.build_insight_prompt(
                record["essay"],np.array(list(record["scores"].values())),original=True))
            _,adjacent=generator(record,prompt_variant="adjacent_text")
            self.assertEqual(diagnostics["seed"],adjacent["seed"])
            self.assertEqual(prompts[-1],runtime.build_insight_prompt(
                record["essay"],np.array(list(record["scores"].values())),record["task_prompt"],explicit_bands=True))
            self.assertIn("Essay task (use ONLY for Task Response evaluation):\n                "+record["task_prompt"]+"\n\n                Essay:",prompts[-1])
            self.assertIn("Before answering:",prompts[-1])
            self.assertEqual(completion,result)
            self.assertEqual(diagnostics["stop_reason"],"eos")
            generator.profile={**generator.profile,"max_input_tokens":4}
            with self.assertRaisesRegex(ValueError,"budget exceeded"):generator(record)

    def test_builder_cli_accepts_no_arguments_and_explicit_paths(self) -> None:
        """Verify that builder cli accepts no arguments and explicit paths.

        Returns:
            None: This operation returns no value.
        """
        spec=importlib.util.spec_from_file_location("panda_builder_cli",ROOT/"code/offline_research/tools/BuildPandaKaggle.py")
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        project=self.path/"project"
        archive=self.path/"bundle.zip"
        with patch.object(module,"ROOT",project), patch.object(module,"build",return_value=archive) as build, patch("builtins.print"):
            self.assertEqual(module.main([]),archive)
            build.assert_called_once_with(project/"results/panda_generation/kaggle",None)
            build.reset_mock()
            destination,prepared=self.path/"custom-output",self.path/"prepared"
            self.assertEqual(module.main(["--output",str(destination),"--prepared",str(prepared)]),archive)
            build.assert_called_once_with(destination,prepared)

    def notebook_scope(self) -> Any:
        """Notebook scope.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location("panda_notebook_settings",ROOT/"code/offline_research/tools/BuildPandaKaggle.py")
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook=module.notebook("a"*64,{"code/offline_research/ManageCorpus.py":"b"*64})
        source=next(cell["source"] for cell in notebook["cells"] if "MODEL_ID =" in cell["source"])
        scope: dict[str, Any]={"project_root":self.path}
        secrets=SimpleNamespace(UserSecretsClient=lambda:SimpleNamespace(get_secret=lambda name:"hf_test_not_a_real_token"))
        with patch.dict(sys.modules,{"kaggle_secrets":secrets}):
            exec(source,scope)
        return scope,notebook

    def test_notebook_model_selection_pins_revision_and_reuses_saved_commit(self) -> None:
        """Verify that notebook model selection pins revision and reuses saved commit.

        Returns:
            None: This operation returns no value.
        """
        from types import SimpleNamespace
        import yaml
        scope,_=self.notebook_scope()
        base=self.path/"code/offline_research/resources/panda_generation.yaml"
        base.parent.mkdir(parents=True)
        base.write_bytes((ROOT/"code/offline_research/resources/panda_generation.yaml").read_bytes())
        configure=scope["configure_generation"]
        secret="hf_test_not_a_real_token"
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value=SimpleNamespace(sha="b"*40)
            target=configure(self.path,"example/another-model",None,"other_run",secret)
            api.assert_called_once_with(token=secret)
            api.return_value.model_info.assert_called_once_with("example/another-model",revision="main")
            profile=yaml.safe_load(target.read_text())
            self.assertEqual(profile["generator"]["model_id"],"example/another-model")
            self.assertEqual(profile["generator"]["revision"],"b"*40)
            self.assertNotIn(secret,target.read_text())
            write_json(self.path/"other_run/generation_manifest.json",{"generator":{"model_id":"example/another-model","revision":"c"*40}})
            api.reset_mock()
            target=configure(self.path,"example/another-model",None,"other_run",secret)
            api.assert_not_called()
            self.assertEqual(yaml.safe_load(target.read_text())["generator"]["revision"],"c"*40)
        self.assertEqual(base.read_bytes(),(ROOT/"code/offline_research/resources/panda_generation.yaml").read_bytes())
        target=configure(self.path,"Qwen/Qwen2.5-1.5B-Instruct",None,"original_run",None,"original_text")
        self.assertEqual(yaml.safe_load(target.read_text())["generator"]["prompt_variant"],"original_text")
        self.assertEqual(yaml.safe_load(target.read_text())["generator"]["batch_size"],12)
        target=configure(self.path,"Qwen/Qwen2.5-1.5B-Instruct",None,"batched_run",None,"adjacent_text",5)
        self.assertEqual(yaml.safe_load(target.read_text())["generator"]["batch_size"],5)
        for invalid in (0,-1,True,1.5,"4"):
            with self.subTest(batch_size=invalid), self.assertRaisesRegex(ValueError,"BATCH_SIZE"):
                configure(self.path,"Qwen/Qwen2.5-1.5B-Instruct",None,"invalid_batch",None,"adjacent_text",invalid)
        with self.assertRaisesRegex(ValueError,"RUN_NAME"):
            configure(self.path,"example/model",None,"../escape",None)

    def test_notebook_secret_is_optional_and_errors_do_not_echo_secret(self) -> None:
        """Verify that notebook secret is optional and errors do not echo secret.

        Returns:
            None: This operation returns no value.
        """
        from types import SimpleNamespace
        from unittest.mock import Mock
        scope,_=self.notebook_scope()
        get_secret=Mock(return_value="hf_test_not_a_real_token")
        client=SimpleNamespace(get_secret=get_secret)
        with patch.dict(sys.modules,{"kaggle_secrets":SimpleNamespace(UserSecretsClient=lambda:client)}):
            self.assertIsNone(scope["notebook_token"](False,"HF_TOKEN"))
            get_secret.assert_not_called()
            self.assertEqual(scope["notebook_token"](True,"HF_TOKEN"),"hf_test_not_a_real_token")
            get_secret.assert_called_once_with("HF_TOKEN")
            get_secret.side_effect=RuntimeError("Backend response contains hf_test_not_a_real_token")
            with self.assertRaises(RuntimeError) as failure:scope["notebook_token"](True,"HF_TOKEN")
            self.assertNotIn("hf_test_not_a_real_token",str(failure.exception))
            self.assertTrue(failure.exception.__suppress_context__)

    def test_notebook_passes_profile_and_secret_only_in_child_environment(self) -> None:
        """Verify that notebook passes profile and secret only in child environment.

        Returns:
            None: This operation returns no value.
        """
        from unittest.mock import Mock
        import os
        scope,notebook=self.notebook_scope()
        run_dir=scope["run_dir"]
        write_json(run_dir/"comparison_report.json",{"selected_count":40})
        write_json(run_dir/"generation_report.json",{"accepted":0})
        profile=self.path/"selected.profile.yaml"
        configure=Mock(return_value=profile)
        source=next(cell["source"] for cell in notebook["cells"] if "profile_path = configure_generation(" in cell["source"])
        scope.update(python=Path(sys.executable),subprocess=subprocess,os=os,configure_generation=configure)
        scope["BATCH_SIZE"]=4
        for token,mode in (("hf_test_not_a_real_token","compare"),(None,"generate")):
            scope["hf_token"]=token
            scope["RUN_MODE"]=mode
            output=io.StringIO()
            with patch.dict(os.environ,{"HF_TOKEN":"ambient_token"}), patch("subprocess.run") as run, redirect_stdout(output):
                exec(source,scope)
            command=run.call_args.args[0]
            self.assertEqual(command[3],mode)
            self.assertEqual(command[command.index("--profile")+1],str(profile))
            self.assertEqual(run.call_args.kwargs["env"].get("HF_TOKEN"),token)
            self.assertEqual(run.call_args.kwargs["env"]["HF_DEACTIVATE_ASYNC_LOAD"],"1")
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONUNBUFFERED"],"1")
            self.assertEqual(configure.call_args.args[-2],"adjacent_text")
            self.assertEqual(configure.call_args.args[-1],4)
            self.assertNotIn("hf_test_not_a_real_token",str(command)+output.getvalue())

    def test_hf_token_reaches_both_loaders_without_persistent_login(self) -> None:
        """Verify that hf token reaches both loaders without persistent login.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.PandaModel import PandaModel
        token="hf_test_not_a_real_token"
        with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("huggingface_hub.login") as login:
            PandaModel(token).load_model_and_tokenizer("example/private-model",revision="b"*40)
            login.assert_not_called()
            tokenizer.assert_called_once_with("example/private-model",revision="b"*40,token=token)
            self.assertEqual(model.call_args.kwargs["token"],token)
            self.assertEqual(model.call_args.kwargs["revision"],"b"*40)

    def test_generated_notebook_is_python_and_archive_safe(self) -> None:
        """Verify that generated notebook is python and archive safe.

        Returns:
            None: This operation returns no value.
        """
        spec=importlib.util.spec_from_file_location("panda_builder",ROOT/"code/offline_research/tools/BuildPandaKaggle.py")
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook=module.notebook("a"*64,{"code/offline_research/ManageCorpus.py":"b"*64})
        settings,_=self.notebook_scope()
        self.assertEqual(settings["RUN_MODE"],"generate")
        self.assertEqual(settings["PROMPT_VARIANT"],"adjacent_text")
        self.assertEqual(settings["RUN_NAME"],"panda_adjacent_bands_v5_batch12")
        self.assertEqual(settings["BATCH_SIZE"],12)
        self.assertIs(settings["USE_HF_TOKEN"],True)
        for cell in notebook["cells"]:
            if cell["cell_type"]=="code":ast.parse(cell["source"])
        scope={}
        exec(notebook["cells"][1]["source"],scope)
        source=self.path/"input"
        source.mkdir()
        with zipfile.ZipFile(source/"bad.zip","w") as archive:
            archive.writestr("code/shared/paths.yaml","{}")
            archive.writestr("code/offline_research/ManageCorpus.py","pass")
            archive.writestr("../escape.py","bad")
        with self.assertRaisesRegex(ValueError,"Unsafe archive"):
            scope["prepare_project"](source,self.path/"working")


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--sql-fixture":
        case=PandaCorpusTests()
        case.setUp()
        try:
            corpus=case.corpus()
            destination=Path(sys.argv[2])
            destination.mkdir(parents=True,exist_ok=True)
            for name in ("manifest.json","records.jsonl"):
                (destination/name).write_bytes((corpus.path/name).read_bytes())
            print(destination)
        finally:case.doCleanups()
    else:unittest.main()
