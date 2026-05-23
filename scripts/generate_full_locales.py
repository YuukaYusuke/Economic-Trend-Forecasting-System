"""
Generate complete locale JSON files from English base + machine translation.

Skips: ar (manual), jv (manual Javanese script).
Run: python scripts/generate_full_locales.py
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "locales"
EN_PATH = LOCALES / "en.json"

SKIP = frozenset({"en", "ar", "jv", "id", "ja", "zh"})  # jv: locales/jv.json (aksara Jawa, manual)

TARGETS = {
    "id": "id",
    "ja": "ja",
    "zh": "zh-CN",
    "ko": "ko",
    "ms": "ms",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "th": "th",
}

# Jangan terjemahkan placeholder / tag HTML / kode
PLACEHOLDER_RE = re.compile(
    r"(\{[^}]+\}|<[^>]+>|</[^>]+>|→|·|LSTM|MAE|RMSE|MAPE|CSV|PDF|USD|"
    r"dataset\.csv|Frankfurter|Alpha Vantage|reportlab|models/|NAIK|TURUN|DATAR)"
)


def flat(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def unflat(flat_dict: dict[str, str]) -> dict:
    root: dict = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return root


def protect(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def repl(m: re.Match) -> str:
        tokens.append(m.group(0))
        return f"⟦{len(tokens) - 1}⟧"

    return PLACEHOLDER_RE.sub(repl, text), tokens


def restore(text: str, tokens: list[str]) -> str:
    for i, tok in enumerate(tokens):
        text = text.replace(f"⟦{i}⟧", tok)
    return text


def translate_text(text: str, translator: GoogleTranslator) -> str:
    if not text.strip():
        return text
    protected, tokens = protect(text)
    if not re.sub(r"⟦\d+⟧", "", protected).strip():
        return text
    try:
        out = translator.translate(protected)
    except Exception:
        time.sleep(1.5)
        out = translator.translate(protected)
    return restore(out or text, tokens)


def translate_dict(base: dict, translator: GoogleTranslator) -> dict:
    f = flat(base)
    out: dict[str, str] = {}
    total = len(f)
    for i, (k, v) in enumerate(f.items(), 1):
        out[k] = translate_text(v, translator)
        if i % 20 == 0:
            print(f"  {i}/{total}")
        time.sleep(0.08)
    return unflat(out)


def main():
    base = json.loads(EN_PATH.read_text(encoding="utf-8"))
    for code, google_target in TARGETS.items():
        if code in SKIP:
            print("skip", code)
            continue
        print("translating", code, "->", google_target)
        tr = GoogleTranslator(source="en", target=google_target)
        merged = translate_dict(base, tr)
        path = LOCALES / f"{code}.json"
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
