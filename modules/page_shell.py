from pathlib import Path

import streamlit as st

from modules.api import get_bulk_live_rates_usd
from modules.config import MODELS_DIR, TRAIN_EPOCHS
from modules.i18n import (
    DEFAULT_LANG,
    available_languages,
    is_rtl,
    lang,
    language_label,
    t,
    tp,
)
from modules.deploy import show_deploy_banner
from modules.trainer import get_trainable_currencies
from modules.ui_theme import footer, hero, inject_theme


def init_session():
    if "lang" not in st.session_state:
        st.session_state.lang = DEFAULT_LANG
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"


def render_language_switcher():
    st.markdown(f'<p class="lang-label">{t("lang_label")}</p>', unsafe_allow_html=True)
    codes = available_languages()
    cur = lang()
    idx = codes.index(cur) if cur in codes else 0
    picked = st.selectbox(
        t("language"),
        codes,
        index=idx,
        format_func=language_label,
        label_visibility="collapsed",
        key="lang_selectbox",
    )
    if picked != cur:
        st.session_state.lang = picked
        st.rerun()


def render_sidebar():
    trained = len(list(Path(MODELS_DIR).glob("*.keras")))
    total = len(get_trainable_currencies())

    with st.sidebar:
        st.markdown(f"### {t('sidebar_title')}")
        render_language_switcher()
        st.session_state.dark_mode = st.toggle(t("dark_mode"), value=st.session_state.dark_mode)
        st.divider()
        st.progress(min(trained / max(total, 1), 1.0))
        st.caption(f"{t('models_ready')}: {trained}/{total}")
        if trained < total:
            st.warning(t("train_warn"))
            st.code("python train_models.py", language="bash")
            show_deploy_banner()
        st.caption(t("nav_hint"))
        bulk = get_bulk_live_rates_usd()
        if len(bulk) > 5:
            st.session_state.api_status = "ok"
            st.success(t("api_ok"))
        else:
            st.session_state.api_status = "cached"
            st.warning(t("api_cached"))


def api_banner():
    if st.session_state.get("api_status") == "cached":
        st.warning(t("api_cached"))


def setup_page(page_key: str):
    init_session()
    inject_theme(st.session_state.dark_mode, rtl=is_rtl())
    api_banner()
    hero(tp(page_key, "title"), tp(page_key, "subtitle"), page_key[:3].upper())


def page_footer():
    footer(
        source=t("footer_data_source"),
        model_info=t("footer_model_info", epochs=TRAIN_EPOCHS),
    )
