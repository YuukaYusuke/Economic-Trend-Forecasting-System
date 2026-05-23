import pandas as pd
import streamlit as st

from modules.api import get_bulk_live_rates_usd, get_currency_names
from modules.config import DATA_PATH
from modules.currency_registry import CURRENCY_COLUMNS


@st.cache_data(ttl=3600)
def get_dataset_columns():
    return set(pd.read_csv(DATA_PATH, nrows=0).columns)


@st.cache_data(ttl=300)
def get_selectable_currencies():
    """
    Mata uang yang ada di dataset DAN tersedia di API live.
    Return: list of dict {iso, name, column}
    """
    bulk = get_bulk_live_rates_usd()
    cols = get_dataset_columns()
    items = []

    for column, (iso, name) in CURRENCY_COLUMNS.items():
        if column not in cols:
            continue
        if iso not in bulk:
            continue
        items.append({"iso": iso, "name": name, "column": column})

    return sorted(items, key=lambda x: x["name"])


@st.cache_data(ttl=3600)
def get_dataset_currencies():
    """Mata uang yang punya kolom di dataset (tanpa syarat API live)."""
    cols = get_dataset_columns()
    items = []
    for column, (iso, name) in CURRENCY_COLUMNS.items():
        if column in cols:
            items.append({"iso": iso, "name": name, "column": column})
    return sorted(items, key=lambda x: x["name"])
