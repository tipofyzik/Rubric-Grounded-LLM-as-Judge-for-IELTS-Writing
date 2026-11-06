"""Multilingual Gradio presentation of the UI-independent scoring service."""
from typing import Any, Iterator

import json
from pathlib import Path

import gradio as gr

from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Localization import DEFAULT_LANGUAGE, LANGUAGES, MESSAGES, translate


EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "resources/demo_examples.json"
ERROR_CODES = {
    "invalid_input", "input_too_long", "context_budget_exceeded", "service_busy",
    "judge_quota", "invalid_output", "judge_metadata_mismatch",
}


def load_examples() -> Any:
    """Load examples.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))["examples"]


def render_result(result: Any, language: str=DEFAULT_LANGUAGE) -> Any:
    """Translate presentation only; preserve the original service result.

    Args:
        result (Any): Result used by this operation.
        language (str): Language used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if result is None:
        return [None] * 4 + ["", translate(language, "ready"), None]
    if result["status"] != "ok":
        error = result["error"]
        code, stage = error["code"], error["stage"]
        if code in ERROR_CODES:
            key = "error." + code
        elif code.endswith("_timeout"):
            key = "error.timeout"
        elif stage in {"retrieval", "aes", "judge"}:
            key = "error." + stage
        else:
            key = "error.default"
        message = translate(language, key)
        return [None] * 4 + ["", translate(language, "failed", message=message), result]

    warnings = [
        translate(language, "warning." + code)
        if "warning." + code in MESSAGES[DEFAULT_LANGUAGE] else str(code)
        for code in result.get("warnings", [])
    ]
    metadata = result["metadata"]
    if metadata.get("aes", {}).get("input", {}).get("truncated"):
        warnings.append(translate(language, "warning.aes_truncated"))
    if metadata.get("retrieval", {}).get("query_truncated"):
        warnings.append(translate(language, "warning.query_truncated"))
    status = translate(language, "completed", seconds=result["timings"]["total_seconds"])
    return [str(result["scores"][key]) for key in PUBLIC_CRITERION_IDS] + ["\n\n".join(warnings), status, result]


def render_presentation(language: str, result: Any=None, status_key: str="ready") -> Any:
    """Prepare messages once so switching language needs no server request.

    Args:
        language (str): Language used by this operation.
        result (Any): Result used by this operation.
        status_key (str): Status key used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if result is None:
        messages = {code: ["", translate(code, status_key)] for _, code in LANGUAGES}
        selected = messages.get(language, messages[DEFAULT_LANGUAGE])
        values = [None] * 4 + [*selected, None]
    else:
        values = render_result(result, language)
        messages = {code: render_result(result, code)[4:6] for _, code in LANGUAGES}
    return values + [messages]


def build_demo(service: Any) -> Any:
    """Connect the shared scoring service to the localized Gradio submission interface.

    Args:
        service (Any): Service used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    profiles = [service.judge.profile]
    examples = load_examples()
    initial = DEFAULT_LANGUAGE

    def example_labels(language: str) -> list[Any]:
        """Example labels.

        Args:
            language (str): Language used by this operation.

        Returns:
            list[Any]: The computed result for the supplied inputs.
        """
        return [translate(language, "example." + item["id"]) for item in examples]

    def limit_text(language: str) -> Any:
        """Limit text.

        Args:
            language (str): Language used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return translate(language, "limits", max_chars=f"{service.limits.max_input_chars:,}")

    with gr.Blocks(title="IELTS Writing Task 2", analytics_enabled=False) as demo:
        heading = gr.Markdown(translate(initial, "heading"))
        language = gr.Dropdown(
            choices=LANGUAGES, value=initial, label=translate(initial, "language"),
            allow_custom_value=False, filterable=False,
        )
        model = gr.Dropdown(
            choices=[(profile.label, profile.key) for profile in profiles],
            value=service.judge.profile.key, label=translate(initial, "model"), allow_custom_value=False,
        )
        task = gr.Textbox(label=translate(initial, "task"), lines=3, placeholder=translate(initial, "task_placeholder"))
        essay = gr.Textbox(label=translate(initial, "essay"), lines=12, placeholder=translate(initial, "essay_placeholder"))
        example_picker = gr.Examples(
            examples=[[item["task_prompt"], item["essay"]] for item in examples],
            example_labels=example_labels(initial), inputs=[task, essay],
            cache_examples=False, run_on_click=False, label=translate(initial, "examples"),
            api_visibility="private",
        )
        # Loading an example must not restore the original language's field labels.
        for props in example_picker.dataset.component_props:
            props.pop("label", None)
            props.pop("placeholder", None)
        limits = gr.Markdown(limit_text(initial))
        button = gr.Button(translate(initial, "evaluate"), variant="primary")
        status = gr.Markdown(translate(initial, "ready"), elem_id="scoring-status")
        with gr.Row():
            scores = [
                gr.Textbox(label=translate(initial, "score." + key), value="",
                           placeholder=translate(initial, "no_score"), interactive=False)
                for key in PUBLIC_CRITERION_IDS
            ]
        warnings = gr.Markdown("", elem_id="scoring-warnings")
        disclaimer = gr.Markdown(translate(initial, "disclaimer"))
        with gr.Accordion(translate(initial, "run_details"), open=False) as details_section:
            details = gr.JSON(label=translate(initial, "diagnostics"), value=None)
        translated_messages = gr.JSON(value=render_presentation(initial)[-1], visible=False)
        submission = gr.State(value=None, time_to_live=900)
        result_outputs = [*scores, warnings, status, details, translated_messages]
        controls = [language, model, task, essay, example_picker.dataset, button]

        def control_updates(enabled: Any) -> Any:
            """Control updates.

            Args:
                enabled (Any): Enabled used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return [gr.update(interactive=enabled) for _ in range(4)] + [
                gr.update(visible=enabled), gr.update(interactive=enabled),
            ]

        def prepare(selected_language: Any, selected_model: Any, task_prompt: str, essay_text: str) -> Any:
            """Assemble the current operation's validated inputs and metadata.

            Args:
                selected_language (Any): Selected language used by this operation.
                selected_model (Any): Selected model used by this operation.
                task_prompt (str): English Task 2 question against which the essay is assessed.
                essay_text (str): Original English essay text; source content is preserved.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            snapshot = (selected_language, selected_model, task_prompt, essay_text)
            return [snapshot] + render_presentation(selected_language, status_key="waiting") + control_updates(False)

        def evaluate(snapshot: Any) -> Iterator[Any]:
            """Evaluate.

            Args:
                snapshot (Any): Snapshot used by this operation.

            Returns:
                Iterator[Any]: The computed result for the supplied inputs.
            """
            selected_language, selected_model, task_prompt, essay_text = snapshot
            yield render_presentation(selected_language, status_key="running")
            result = service.score(model_key=selected_model, task_prompt=task_prompt, essay=essay_text)
            yield render_presentation(selected_language, result)

        def unlock() -> Any:
            """Unlock.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return [None] + control_updates(True)

        def event_failure(selected_language: Any) -> Any:
            """Event failure.

            Args:
                selected_language (Any): Selected language used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            return unlock() + render_presentation(selected_language, status_key="interrupted")

        def language_updates(selected_language: Any) -> dict[str, Any]:
            """Language updates.

            Args:
                selected_language (Any): Selected language used by this operation.

            Returns:
                dict[str, Any]: Named result fields and metadata produced by this operation.
            """
            updates = [
                gr.update(value=translate(selected_language, "heading")),
                gr.update(label=translate(selected_language, "language")),
                gr.update(label=translate(selected_language, "model")),
                gr.update(label=translate(selected_language, "task"), placeholder=translate(selected_language, "task_placeholder")),
                gr.update(label=translate(selected_language, "essay"), placeholder=translate(selected_language, "essay_placeholder")),
                gr.update(label=translate(selected_language, "examples"), sample_labels=example_labels(selected_language)),
                gr.update(value=limit_text(selected_language)),
                gr.update(value=translate(selected_language, "evaluate")),
                gr.update(value=translate(selected_language, "disclaimer")),
                gr.update(label=translate(selected_language, "run_details")),
            ]
            updates += [
                gr.update(label=translate(selected_language, "score." + key),
                          placeholder=translate(selected_language, "no_score"))
                for key in PUBLIC_CRITERION_IDS
            ]
            return {"content": updates, "details": gr.update(label=translate(selected_language, "diagnostics"))}

        ready_messages = json.dumps(render_presentation(initial, status_key="ready_current")[-1])
        clear_js = "(language) => { const messages = " + ready_messages + "; return [null, null, null, null, ...(messages[language] || messages.en), null, messages]; }"
        for component in (model, task, essay):
            component.change(fn=None, inputs=language, js=clear_js, outputs=result_outputs,
                             queue=False, api_visibility="private")
        prepared = button.click(
            prepare, inputs=[language, model, task, essay], outputs=[submission, *result_outputs, *controls],
            queue=False, trigger_mode="once", api_name="prepare_submission", api_visibility="undocumented",
        )
        evaluated = prepared.success(
            evaluate, inputs=submission, outputs=result_outputs, concurrency_limit=1,
            concurrency_id="scoring", trigger_mode="once", api_name="evaluate", api_visibility="undocumented",
            show_progress="full", show_progress_on=status,
        )
        evaluated.success(unlock, outputs=[submission, *controls], queue=False, api_visibility="private")
        evaluated.failure(event_failure, inputs=language, outputs=[submission, *controls, *result_outputs], queue=False, api_visibility="private")

        updates = json.dumps({code: language_updates(code) for _, code in LANGUAGES})
        initial_messages = json.dumps(render_presentation(initial)[-1])
        language.change(
            fn=None, inputs=[language, translated_messages],
            js="(language, messages) => { const updates = " + updates + "; const labels = updates[language] || updates.en; const views = messages || " + initial_messages + "; return [...labels.content, ...(views[language] || views.en), labels.details]; }",
            outputs=[heading, language, model, task, essay, example_picker.dataset, limits,
                     button, disclaimer, details_section, *scores, warnings, status, details],
            queue=False, api_visibility="private",
        )
    demo.queue(max_size=4, default_concurrency_limit=1, api_open=False)
    return demo
