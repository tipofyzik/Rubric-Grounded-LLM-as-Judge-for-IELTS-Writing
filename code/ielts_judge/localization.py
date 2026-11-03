"""Application UI text; language selection belongs to each browser session."""

import json
from pathlib import Path


DEFAULT_LANGUAGE = "en"
LANGUAGES = (("English", "en"), ("Русский", "ru"), ("Deutsch", "de"))
LOCALES_PATH = Path(__file__).resolve().parents[1] / "resources/locales"
MESSAGES = {
    code: json.loads((LOCALES_PATH / f"{code}.json").read_text(encoding="utf-8"))
    for _, code in LANGUAGES
}


def translate(language, key, **values):
    messages = MESSAGES.get(language, MESSAGES[DEFAULT_LANGUAGE])
    template = messages.get(key, MESSAGES[DEFAULT_LANGUAGE][key])
    return template.format(**values)
