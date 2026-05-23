import streamlit as st

from modules.i18n import tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_preprocessed_df
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("preprocessing")

iso, column, name = currency_selector()
df = get_preprocessed_df(column)
note(tp("preprocessing", "note", name=name))

c1, c2, c3 = st.columns(3)
stat_card(tp("preprocessing", "rows"), str(len(df)), "neutral")
stat_card(tp("preprocessing", "min"), f"{df['rate'].min():,.4f}", "down")
stat_card(tp("preprocessing", "max"), f"{df['rate'].max():,.4f}", "up")

section(tp("preprocessing", "sample"))
st.dataframe(df.tail(15), width="stretch")
page_footer()
