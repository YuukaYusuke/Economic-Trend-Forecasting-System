import streamlit as st

from modules.currencies import get_selectable_currencies
from modules.i18n import t


def currency_selector():
    options = get_selectable_currencies()
    if not options:
        st.error("—")
        st.stop()

    labels = [f"{o['iso']} — {o['name']}" for o in options]
    default_iso = st.session_state.get("selected_iso", options[0]["iso"])
    default_idx = next((i for i, o in enumerate(options) if o["iso"] == default_iso), 0)

    picked = st.selectbox(t("select_currency"), labels, index=default_idx, key="currency_selectbox")
    chosen = options[labels.index(picked)]
    st.session_state.selected_iso = chosen["iso"]
    return chosen["iso"], chosen["column"], chosen["name"]


def dual_currency_selector():
    options = get_selectable_currencies()
    if len(options) < 2:
        st.error("—")
        st.stop()
    labels = [f"{o['iso']} — {o['name']}" for o in options]
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox(t("currency_a"), labels, key="dual_a")
    with c2:
        b = st.selectbox(t("currency_b"), labels, index=1, key="dual_b")
    oa = options[labels.index(a)]
    ob = options[labels.index(b)]
    return (oa["iso"], oa["column"], oa["name"]), (ob["iso"], ob["column"], ob["name"])
