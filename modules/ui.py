import streamlit as st

from modules.currencies import get_dataset_currencies, get_selectable_currencies
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


_DATASET_DEFAULT_ISOS = ("IDR", "EUR", "JPY")


def dataset_currencies_selector(max_count: int = 6):
    options = get_dataset_currencies()
    if not options:
        st.error("—")
        st.stop()

    labels = [f"{o['iso']} — {o['name']}" for o in options]
    label_map = dict(zip(labels, options))

    saved = st.session_state.get("dataset_preview_isos")
    if saved:
        default_labels = [lb for lb in labels if label_map[lb]["iso"] in saved]
    else:
        default_labels = []
        for iso in _DATASET_DEFAULT_ISOS:
            for lb in labels:
                if label_map[lb]["iso"] == iso and lb not in default_labels:
                    default_labels.append(lb)
    if not default_labels:
        default_labels = labels[: min(3, len(labels))]

    picked = st.multiselect(
        t("select_currencies"),
        labels,
        default=default_labels,
        max_selections=max_count,
        key="dataset_currency_multiselect",
        placeholder=t("select_currencies_placeholder"),
    )
    if not picked:
        st.info(t("select_currencies_empty"))
        st.stop()

    chosen = [label_map[lb] for lb in picked]
    st.session_state.dataset_preview_isos = [c["iso"] for c in chosen]
    return chosen


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
