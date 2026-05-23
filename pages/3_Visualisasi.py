import streamlit as st

from modules.charts import history_chart
from modules.i18n import tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_preprocessed_df
from modules.ui import currency_selector
from modules.ui_theme import section, stat_card

setup_page("visualisasi")

iso, column, name = currency_selector()
df = get_preprocessed_df(column)
st.plotly_chart(history_chart(df, iso, name), width="stretch")

section(tp("visualisasi", "stats"))
c1, c2, c3, c4 = st.columns(4)
stat_card(tp("visualisasi", "last_date"), str(df["date"].iloc[-1].date()), "neutral")
stat_card(tp("visualisasi", "last_rate"), f"{df['rate'].iloc[-1]:,.4f}", "neutral")
stat_card(tp("visualisasi", "avg"), f"{df['rate'].mean():,.4f}", "neutral")
stat_card(tp("visualisasi", "vol"), f"{df['rate'].std():,.4f}", "neutral")
page_footer()
