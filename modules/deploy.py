"""Pemeriksaan kesiapan deploy & pesan untuk operator."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.config import MODELS_DIR, TRAIN_EPOCHS
from modules.settings import allow_runtime_training, is_streamlit_cloud
from modules.trainer import get_trainable_currencies


def count_trained_models() -> int:
    return len(list(Path(MODELS_DIR).glob("*.keras")))


def models_deployed() -> bool:
    return count_trained_models() > 0


def deploy_status() -> dict:
    total = len(get_trainable_currencies())
    trained = count_trained_models()
    return {
        "cloud": is_streamlit_cloud(),
        "trained": trained,
        "total": total,
        "ready": trained > 0,
        "allow_training": allow_runtime_training(),
    }


def show_deploy_banner():
    """Peringatan singkat di sidebar jika model belum ada (khususnya di Cloud)."""
    s = deploy_status()
    if s["ready"]:
        return

    if s["cloud"] and not s["allow_training"]:
        st.error(
            "Model LSTM belum ada di server. "
            "Latih lokal (`python train_models.py`), commit folder `models/`, lalu deploy ulang."
        )
    elif not s["allow_training"]:
        st.warning(
            "Model belum dilatih. Jalankan `python train_models.py` di mesin lokal, "
            "lalu unggah folder `models/` ke repo."
        )
