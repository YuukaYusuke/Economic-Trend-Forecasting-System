import pandas as pd
import streamlit as st

from modules.api import get_bulk_live_rates_usd, get_currency_names
from modules.config import DATA_PATH, DISPLAY_NAMES
from modules.currency_registry import CURRENCY_COLUMNS
from modules.load_data import load_dataset
from modules.trend import get_direction


MIN_OBS_5Y = 30


def _avg_rate_5y(raw: pd.DataFrame, column: str, cutoff) -> float | None:
    mask = (raw["date"] >= cutoff) & raw[column].notna() & (raw[column] > 0)
    subset = raw.loc[mask, column]
    if len(subset) < MIN_OBS_5Y:
        return None
    return float((1 / subset).mean())


@st.cache_data(ttl=300, show_spinner=False)
def build_realtime_comparison():
    """
    Tampilkan SEMUA kurs dari API live.
    Jika ada di dataset → hitung rata-rata 5 tahun & trend.
    """
    raw = load_dataset(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    cutoff = raw["date"].max() - pd.DateOffset(years=5)

    bulk = get_bulk_live_rates_usd()
    api_names = get_currency_names()
    iso_to_column = {iso: col for col, (iso, _) in CURRENCY_COLUMNS.items()}

    rows = []
    for iso, live in sorted(bulk.items(), key=lambda x: x[0]):
        if iso == "USD":
            continue

        name = DISPLAY_NAMES.get(iso) or api_names.get(iso, iso)
        column = iso_to_column.get(iso)
        avg_5y = _avg_rate_5y(raw, column, cutoff) if column else None
        in_dataset = column is not None and avg_5y is not None

        spark = None
        if column and column in raw.columns:
            tail = raw[[column]].dropna().tail(30)
            if len(tail) >= 5:
                spark = (1 / tail[column]).tolist()

        if in_dataset:
            pct = ((live - avg_5y) / avg_5y) * 100
            trend = get_direction(live, avg_5y)
        else:
            pct = None
            trend = "—"

        rows.append(
            {
                "Kode": iso,
                "Mata Uang": name,
                "Di Dataset": "Ya" if in_dataset else "Tidak",
                "Rata-rata 5 Tahun": avg_5y,
                "Kurs Live": live,
                "Perubahan (%)": pct,
                "Trend": trend,
                "_spark": spark,
            }
        )

    df = pd.DataFrame(rows)
    return df, cutoff
