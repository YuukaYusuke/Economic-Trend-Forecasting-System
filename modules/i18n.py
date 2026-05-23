"""
Sistem terjemahan berbasis dictionary per bahasa.

Bahasa terdaftar: id, en, ar, ja, zh, ko, ms, es, fr, de, pt, th, jv (aksara Jawa)

Tambah bahasa baru:
  1. Salin locales/en.json → locales/<kode>.json lalu terjemahkan
  2. Daftarkan di locales/languages.json (set rtl: true untuk Arab/Hebrew)
  3. Jalankan: python scripts/generate_locales.py (jika pakai overrides)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
LANGUAGES_FILE = LOCALES_DIR / "languages.json"
DEFAULT_LANG = "jv"
FALLBACK_ORDER = ("id", "en")


@lru_cache(maxsize=1)
def _load_languages_meta() -> dict:
    if not LANGUAGES_FILE.exists():
        return {"id": {"name": "Indonesia", "short": "ID"}, "en": {"name": "English", "short": "EN"}}
    with open(LANGUAGES_FILE, encoding="utf-8") as f:
        return json.load(f)


def available_languages() -> list[str]:
    """Kode bahasa yang punya file locales/<kode>.json"""
    meta = _load_languages_meta()
    codes = [p.stem for p in LOCALES_DIR.glob("*.json") if p.name != "languages.json"]
    registered = [c for c in meta if c in codes]
    return registered or sorted(codes)


@lru_cache(maxsize=32)
def _load_dictionary(lang_code: str) -> dict:
    path = LOCALES_DIR / f"{lang_code}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve(node: dict, key: str):
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, str) else None


def lang() -> str:
    code = st.session_state.get("lang", DEFAULT_LANG)
    if code not in available_languages():
        return DEFAULT_LANG
    return code


def t(key: str, **kwargs) -> str:
    """Ambil string; fallback: bahasa aktif → id → en → key."""
    chain = []
    for code in (lang(), *FALLBACK_ORDER):
        if code not in chain:
            chain.append(code)

    for code in chain:
        value = _resolve(_load_dictionary(code), key)
        if value is not None:
            text = value.format(**kwargs) if kwargs else value
            return text.replace("*", "")

    return key


def tp(page: str, field: str, **kwargs) -> str:
    return t(f"pages.{page}.{field}", **kwargs)


def tn(nav_key: str) -> str:
    return t(f"nav.{nav_key}")


def language_label(code: str) -> str:
    meta = _load_languages_meta().get(code, {})
    return meta.get("name", code.upper())


def language_short(code: str) -> str:
    meta = _load_languages_meta().get(code, {})
    return meta.get("short", code.upper())


def is_rtl(code: str | None = None) -> bool:
    code = code or lang()
    return bool(_load_languages_meta().get(code, {}).get("rtl", False))


def format_direction(direction: str) -> str:
    mapping = {
        "NAIK": ("dir_up", "📈"),
        "TURUN": ("dir_down", "📉"),
        "DATAR": ("dir_flat", "➡️"),
        "UP": ("dir_up", "📈"),
        "DOWN": ("dir_down", "📉"),
    }
    key, icon = mapping.get(direction, (None, ""))
    label = t(key) if key else direction
    return f"{label} {icon}".strip()


def clear_locale_cache():
    """Panggil setelah edit file JSON saat development."""
    _load_dictionary.cache_clear()
    _load_languages_meta.cache_clear()
