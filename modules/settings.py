"""Konfigurasi aman untuk lokal & Streamlit Cloud (secrets / env)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Muat .env ke os.environ (lokal, tanpa secrets.toml)."""
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


def _secret_get(key: str) -> str | None:
    """Baca secret Streamlit; aman jika secrets.toml tidak ada."""
    try:
        import streamlit as st

        value = str(st.secrets[key]).strip()
        return value or None
    except Exception:
        return None


def _secret_section(section: str, field: str) -> str | None:
    try:
        import streamlit as st

        block = st.secrets.get(section, None)
        if block is None:
            return None
        if isinstance(block, dict):
            value = str(block.get(field, "")).strip()
        else:
            value = str(getattr(block, field, "")).strip()
        return value or None
    except Exception:
        return None


def get_alphavantage_api_key() -> str | None:
    """API key Alpha Vantage — dari env, .env, atau st.secrets."""
    env = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if env:
        return env

    flat = _secret_get("ALPHAVANTAGE_API_KEY")
    if flat:
        return flat

    nested = _secret_section("alphavantage", "api_key")
    return nested


def allow_runtime_training() -> bool:
    """
    Latih model saat runtime (lambat, tidak disarankan di Cloud).
    Default: True di lokal, False di Streamlit Community Cloud.
    """
    try:
        import streamlit as st

        app = st.secrets.get("app", None)
        if app is not None:
            if isinstance(app, dict) and "allow_runtime_training" in app:
                return bool(app["allow_runtime_training"])
            if hasattr(app, "allow_runtime_training"):
                return bool(app.allow_runtime_training)
    except Exception:
        pass

    env = os.environ.get("ALLOW_RUNTIME_TRAINING", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False

    return not is_streamlit_cloud()


def is_streamlit_cloud() -> bool:
    """Deteksi Streamlit Community Cloud (/mount/src)."""
    if Path("/mount/src").is_dir():
        return True
    return os.environ.get("STREAMLIT_SERVER_ENVIRONMENT", "").lower() == "cloud"
