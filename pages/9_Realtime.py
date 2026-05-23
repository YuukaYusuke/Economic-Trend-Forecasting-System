import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules.api import get_bulk_live_rates_usd
from modules.charts import sparkline_figure, trend_heatmap
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.realtime_analysis import build_realtime_comparison
from modules.ui_theme import note, section, stat_card

setup_page("realtime")

with st.spinner(tp("realtime", "loading")):
    df, _ = build_realtime_comparison()

if df.empty:
    st.warning(tp("realtime", "no_api"))
    st.stop()

hm = trend_heatmap(df)
if hm:
    section(tp("realtime", "heatmap"))
    st.plotly_chart(hm, width="stretch")

in_ds = int((df["Di Dataset"] == "Ya").sum())
note(f"{len(df)} API · {in_ds} {t('tbl_in_dataset').lower()} · {datetime.datetime.now():%H:%M}")

section(tp("realtime", "summary"))
c1, c2, c3, c4 = st.columns(4)
stat_card(tp("realtime", "total"), str(len(df)), "neutral")
stat_card(tp("realtime", "with_ds"), str(in_ds), "neutral")
stat_card(tp("realtime", "up"), str((df["Trend"] == "NAIK").sum()), "up")
stat_card(tp("realtime", "down"), str((df["Trend"] == "TURUN").sum()), "down")

fc1, fc2, fc3 = st.columns(3)
with fc1:
    ft = st.selectbox(tp("realtime", "filter_trend"), [t("all"), "NAIK", "TURUN", "DATAR", "—"])
with fc2:
    fd = st.selectbox(tp("realtime", "filter_ds"), [t("all"), t("yes"), t("no")])
with fc3:
    q = st.text_input(t("search"))

view = df.copy()
if ft != t("all"):
    view = view[view["Trend"] == ft]
if fd == t("yes"):
    view = view[view["Di Dataset"] == "Ya"]
elif fd == t("no"):
    view = view[view["Di Dataset"] == "Tidak"]
if q.strip():
    ql = q.strip().lower()
    view = view[view["Mata Uang"].str.lower().str.contains(ql) | view["Kode"].str.lower().str.contains(ql)]

section(tp("realtime", "spark"))
spark_rows = view[view["_spark"].notna()].head(12)
if len(spark_rows):
    cols = st.columns(4)
    for idx, (_, row) in enumerate(spark_rows.iterrows()):
        with cols[idx % 4]:
            st.caption(row["Kode"])
            st.plotly_chart(sparkline_figure(row["_spark"]), width="stretch", config={"displayModeBar": False})

section(tp("realtime", "table"))
raw = view.drop(columns=["_spark"])


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.4f}"


display = pd.DataFrame(
    {
        t("tbl_code"): raw["Kode"],
        t("tbl_currency"): raw["Mata Uang"],
        t("tbl_in_dataset"): raw["Di Dataset"].apply(lambda x: t("yes") if x == "Ya" else t("no")),
        t("tbl_avg_5y"): raw["Rata-rata 5 Tahun"].apply(_fmt),
        t("tbl_live"): raw["Kurs Live"].apply(lambda x: f"{x:,.4f}"),
        t("tbl_change"): raw["Perubahan (%)"].apply(
            lambda x: f"{x:+.2f}%"
            if x is not None and not (isinstance(x, float) and np.isnan(x))
            else "—"
        ),
        t("tbl_trend"): raw["Trend"].apply(
            lambda x: format_direction(x) if x not in ("—", None) else "—"
        ),
    }
)

st.dataframe(display, width="stretch", hide_index=True, height=400)

if st.button(t("refresh"), type="primary"):
    build_realtime_comparison.clear()
    get_bulk_live_rates_usd.clear()
    st.rerun()

page_footer()
