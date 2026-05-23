import streamlit as st

from modules.config import DATA_PATH
from modules.i18n import tp
from modules.load_data import load_dataset
from modules.page_shell import page_footer, setup_page
from modules.ui_theme import note, section, stat_card

setup_page("dataset")

df = load_dataset(DATA_PATH)
c1, c2, c3 = st.columns(3)
stat_card(tp("dataset", "rows"), f"{len(df):,}", "neutral")
stat_card(tp("dataset", "cols"), str(len(df.columns)), "neutral")
stat_card(tp("dataset", "start"), str(df["date"].min())[:10], "neutral")

section(tp("dataset", "preview"))
note(tp("dataset", "preview_note"))
st.dataframe(df.head(20), width="stretch", height=380)
with st.expander(tp("dataset", "stats")):
    st.dataframe(df.describe(), width="stretch")
page_footer()
