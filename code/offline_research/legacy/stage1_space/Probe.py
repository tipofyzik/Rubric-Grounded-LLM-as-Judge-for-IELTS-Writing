"""Manual Stage 1 probe: call the real judge and identify each submission."""
from typing import Any

import hashlib
import json
from uuid import uuid4

from ielts_judge.Prompts import JudgeContext, ReferenceContext


DEFAULT_TASK = (
    "Some people think schools should teach practical skills alongside academic "
    "subjects. To what extent do you agree?"
)
DEFAULT_ESSAY = """Schools should teach practical skills alongside academic subjects because students need both kinds of knowledge. Academic subjects help young people understand the world, while practical lessons help them manage everyday responsibilities.

For example, mathematics gives students a foundation for understanding percentages and interest. A lesson about personal budgets can then show how these ideas apply to rent, savings and borrowing. Cooking lessons can similarly connect scientific knowledge about nutrition to decisions students make every day.

However, schools have limited teaching time. Adding many separate subjects could reduce the attention given to reading, mathematics and science. Schools should therefore integrate practical activities into existing lessons where possible. Students might compare household energy costs in mathematics or practise writing a job application in a language class.

Practical education should also recognise that students have different circumstances. A school should provide useful opportunities without assuming that every family can supply equipment or pay for materials. Simple classroom activities can make these lessons accessible.

In conclusion, I agree that practical skills belong in education. Combining them with academic learning can help students use their knowledge, provided that schools plan the activities carefully and preserve time for essential subjects."""


def synthetic_context() -> JudgeContext:
    """Artificial wiring fixture; these are not AES predictions or human labels.

    Returns:
        JudgeContext: The computed result for the supplied inputs.
    """
    return JudgeContext(
        aes_priors={
            "task_achievement": 5.0,
            "coherence_and_cohesion": 5.5,
            "lexical_resource": 5.0,
            "grammatical_range_and_accuracy": 5.5,
        },
        references=(ReferenceContext(
            essay_id="synthetic-reference-1",
            task_prompt="Should schools include practical lessons?",
            scores={
                "task_achievement": 6.0,
                "coherence_and_cohesion": 6.0,
                "lexical_resource": 6.0,
                "grammatical_range_and_accuracy": 6.0,
            },
            insights="Synthetic test fixture: a clear position and connected supporting examples.",
            essay="Practical lessons can help students apply academic knowledge to everyday decisions.",
        ),),
    )


def run_probe(judge: Any, *, model_key: str, task_prompt: str, essay: str,
              use_synthetic_context: bool = False) -> dict:
    """Use Judge.predict without output parsing, caching, persistence or retries.

    Args:
        judge (Any): Loaded judge adapter providing token counting and generation.
        model_key (str): Enabled immutable judge profile key from the model registry.
        task_prompt (str): English Task 2 question against which the essay is assessed.
        essay (str): English essay text supplied for evaluation.
        use_synthetic_context (bool): Use synthetic context used by this operation.

    Returns:
        dict: Named result fields and metadata produced by this operation.
    """
    context = synthetic_context() if use_synthetic_context else JudgeContext()
    result = judge.predict(
        model_key=model_key, task_prompt=task_prompt, essay=essay, context=context,
    )
    submission = json.dumps(
        {"task_prompt": task_prompt, "essay": essay}, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return {
        **result,
        "probe_run_id": uuid4().hex,
        "submission_sha256": hashlib.sha256(submission).hexdigest(),
        "context_mode": "synthetic_fixture" if use_synthetic_context else "empty",
        "reference_ids": [reference.essay_id for reference in context.references],
        "uses_real_aes_or_retrieval": False,
    }
