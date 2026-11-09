"""One Task 2 inference prompt shared by serving and research callers."""
from typing import Any

from dataclasses import dataclass, field
import json
import math
from typing import Mapping

from ielts_judge.Criteria import CRITERION_IDS, LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS, format_rubric

PROMPT_VERSION = "task2-scores-v2"
CONTEXT_SETUPS = {
    "base": (False, False, False),
    "rubric": (True, False, False),
    "panda": (False, True, False),
    "rubric_panda": (True, True, False),
    "panda_aes": (False, True, True),
    "rubric_panda_aes": (True, True, True),
}
INSTRUCTION = """### Instruction:
Evaluate the essay using the four IELTS Writing Task 2 criteria.
Use Essay task when evaluating Task Response.
Assess the other criteria from Essay.
Use additional context when provided.

Before answering:
1. Check how the essay answers the task.
2. Assess each criterion using evidence from Essay.
3. Consider the supplied context, if any.
4. Check that all four scores are supported by the essay."""
OUTPUT = """### Output:
Return ONLY a JSON object with exactly these keys:
task_response, coherence_and_cohesion,
lexical_resource, grammatical_range_and_accuracy.
Each value must be an integer from 0 to 9.
Do not add explanations or additional fields.

### Answer:"""


@dataclass(frozen=True)
class ReferenceContext:
    """Keep the source scores and raw insights of one retrieved training essay."""
    essay_id: str
    task_prompt: str
    scores: Mapping[str, float]
    insights: str
    essay: str = ""


@dataclass(frozen=True)
class JudgeContext:
    """Select the rubric, references and essay-only AES guidance for one prompt."""
    aes_priors: Mapping[str, float] = field(default_factory=dict)
    references: tuple[ReferenceContext, ...] = ()
    include_rubric: bool = True


def _public_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Validate context scores and translate legacy criterion IDs.

    Args:
        scores (Mapping[str, float]): Criterion names and finite scores from 0 to 9.

    Returns:
        dict[str, float]: The same values keyed by public criterion IDs.

    Raises:
        ValueError: A criterion is unknown or duplicated, or a value is invalid.
    """
    result = {}
    for key, value in scores.items():
        if key not in CRITERION_IDS and key not in PUBLIC_CRITERION_IDS:
            raise ValueError(f"Unknown context criterion: {key!r}.")
        public_key = LEGACY_TO_PUBLIC.get(key, key)
        if public_key in result:
            raise ValueError(f"Duplicate context criterion: {public_key}.")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 9:
            raise ValueError(f"Invalid context score for {key}.")
        result[public_key] = value
    return result


def submission_parts(submission: Any) -> tuple[str, str]:
    """Extract the two input fields allowed into inference.

    Args:
        submission (Any): Dictionary with nonempty task_prompt and essay strings.
            Other fields, including dataset labels, are ignored.

    Returns:
        tuple[str, str]: Original task prompt and essay, in that order.

    Raises:
        ValueError: The record or either required field is invalid.
    """
    if not isinstance(submission, dict):
        raise ValueError("Inference requires a record with task_prompt and essay")
    task, essay = submission.get("task_prompt"), submission.get("essay")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task_prompt must be a nonempty string.")
    if not isinstance(essay, str) or not essay.strip():
        raise ValueError("essay must be a nonempty string.")
    return task, essay


def build_judge_messages(task_prompt: str, essay: str,
                         context: JudgeContext | None = None) -> list[dict[str, str]]:
    """Build system and user chat messages for one scoring request.

    The prompt includes the submission text as supplied. Reference examples
    contribute source scores and insights. Reference IDs, tasks and essays
    stay outside the prompt.

    Args:
        task_prompt (str): IELTS Writing Task 2 question.
        essay (str): Essay to score.
        context (JudgeContext | None): Optional rubric, reference insights and AES
            guidance. None uses the rubric-only default.

    Returns:
        list[dict[str, str]]: Ordered system and user messages with role and content.
    """
    submission_parts({"task_prompt": task_prompt, "essay": essay})
    if context is None:
        context = JudgeContext()
    if not isinstance(context, JudgeContext):
        raise TypeError("context must be JudgeContext or None.")
    if type(context.include_rubric) is not bool:
        raise ValueError("include_rubric must be boolean")
    blocks = [f"### Essay task:\n{task_prompt}", f"### Essay:\n{essay}"]
    if context.include_rubric:
        blocks.append("### Rubric:\n" + format_rubric())
    if context.references:
        examples = []
        for index, reference in enumerate(context.references, 1):
            if not isinstance(reference, ReferenceContext) or not isinstance(reference.insights, str):
                raise ValueError("References require scores and original insights text")
            labels = _public_scores(reference.scores)
            if set(labels) != set(PUBLIC_CRITERION_IDS):
                raise ValueError("References require four source scores")
            examples.append(f"Example {index}\nScores: {json.dumps(labels, ensure_ascii=False, allow_nan=False)}"
                            f"\nInsights: {reference.insights}")
        blocks.append("### Reference insights:\n"
                      "The following scores and insights refer to similar essays.\n"
                      "Use them as guidance when assessing the current essay.\n\n" + "\n\n".join(examples))
    if context.aes_priors:
        priors = _public_scores(context.aes_priors)
        blocks.append("### Preliminary scores:\n"
                      "These are essay-only AES estimates for the current essay.\n"
                      "Make your own final assessment using the essay task and essay.\n\n"
                      + json.dumps(priors, ensure_ascii=False, allow_nan=False))
    return [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": "\n\n".join([*blocks, OUTPUT])}]
