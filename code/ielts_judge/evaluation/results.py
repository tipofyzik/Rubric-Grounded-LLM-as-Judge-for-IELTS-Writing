"""Readable exports and notebook previews, including existing evaluation runs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


def export_results(run, destination=None):
    run = Path(run).resolve()
    destination = Path(destination).resolve() if destination is not None else run / "exports"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    plan = manifest.get("plan", {})
    models = "_".join(plan.get("model_labels", {}).values()) or "models"
    models = re.sub(r"[^a-zA-Z0-9.-]+", "_", models).strip("_")[:100]
    stamp = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", run.name)
    date = "-".join(stamp.groups()[:3]) + "_" + "-".join(stamp.groups()[3:]) if stamp else datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"IELTS_Evaluation_{models}_{len(plan.get('essay_ids', []))}_essays_{len(plan.get('setups', []))}_prompts_{plan.get('repeats', 1)}_repeat_{date}"
    archive = destination / (stem + ".zip")
    number = 2
    while archive.exists():
        archive = destination / f"{stem}_{number}.zip"
        number += 1
    files = [p for p in sorted(run.rglob("*")) if p.is_file() and "exports" not in p.relative_to(run).parts
             and not p.resolve().is_relative_to(destination) and not p.name.endswith(".tmp") and p.name != "export_manifest.json"]
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    inventory = {p.relative_to(run).as_posix(): digest(p) for p in files}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, run.name + "/" + path.relative_to(run).as_posix())
        output.writestr(run.name + "/export_manifest.json", json.dumps({"schema_version": "model-evaluation-export-v1", "files": inventory}, indent=2))
    return archive


def show_results(run, destination="/kaggle/working/IELTS_Evaluation_Results"):
    from IPython.display import display, Markdown, FileLink, Image
    run = Path(run).resolve()
    archive = export_results(run, destination)
    display(Markdown(f"**Готовый ZIP:** `{archive.name}`\n\nЗапуск: `{run}`"))
    display(FileLink(os.path.relpath(archive, Path.cwd())))
    reports = list((run / "reports").glob("*/report_manifest.json"))
    if not reports:
        display(Markdown("Отчёт ещё не создан; архив содержит сохранённые результаты."))
        return archive
    report = max(reports, key=lambda p: p.stat().st_mtime).parent
    display(Markdown("**Графики:** adjacent accuracy — доля оценок в пределах ±1 band; "
        "MAE/RMSE — ошибка; Pearson/Spearman — корреляция; QWK — согласие с оценками датасета; "
        "calibration — средний прогноз при заданном эталонном балле. Ниже показаны calibration curves."))
    images = sorted((report / "plots").glob("*.png"))
    curves = [p for p in images if "score_response" in p.name
              or re.search(r"_calibration_(?:grid|TR|CC|LR|GRA)\.png$", p.name)]
    if not curves:
        display(Markdown("Calibration curves не найдены. Проверьте выбранные типы графиков в конфиге отчёта."))
    for path in curves:
        display(Markdown(f"**Calibration curves:** `{path.name}`"))
        display(Image(filename=str(path)))
    display(Markdown("**Все изображения в отчёте:**"))
    for path in images:
        display(FileLink(os.path.relpath(path, Path.cwd())))
    return archive
