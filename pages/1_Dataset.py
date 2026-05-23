import pandas as pd
import streamlit as st

from modules.charts import dataset_preview_chart
from modules.config import DATA_PATH
from modules.i18n import tp
from modules.load_data import load_dataset
from modules.page_shell import page_footer, setup_page
from modules.ui import dataset_currencies_selector
from modules.ui_theme import note, section, stat_card

setup_page("dataset")

df = load_dataset(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())
default_start = max(min_year, max_year - 4)

c1, c2, c3 = st.columns(3)
stat_card(tp("dataset", "rows"), f"{len(df):,}", "neutral")
stat_card(tp("dataset", "cols"), str(len(df.columns)), "neutral")
stat_card(tp("dataset", "start"), str(df["date"].min())[:10], "neutral")

section(tp("dataset", "preview"))
note(tp("dataset", "preview_note"))

chosen = dataset_currencies_selector()
year_start, year_end = st.slider(
    tp("dataset", "year_range"),
    min_value=min_year,
    max_value=max_year,
    value=(default_start, max_year),
    key="dataset_year_range",
)

period = df[(df["date"].dt.year >= year_start) & (df["date"].dt.year <= year_end)].copy()
series_list = []
preview = period[["date"]].copy()
stat_frames = []

for item in chosen:
    col = item["column"]
    iso, name = item["iso"], item["name"]
    series = period[["date", col]].rename(columns={col: "rate"}).dropna(subset=["rate"])
    series_list.append((series, iso, name))
    preview[iso] = period[col]
    stat_frames.append(series["rate"].rename(iso))

if len(chosen) > 1:
    note(tp("dataset", "multi_chart_note"))

st.plotly_chart(dataset_preview_chart(series_list), width="stretch")

note(tp("dataset", "table_caption"))
st.dataframe(preview.dropna(how="all", subset=[c["iso"] for c in chosen]), width="stretch", height=380, hide_index=True)

with st.expander(tp("dataset", "stats")):
    if stat_frames:
        st.dataframe(pd.concat(stat_frames, axis=1).describe(), width="stretch")

page_footer()
