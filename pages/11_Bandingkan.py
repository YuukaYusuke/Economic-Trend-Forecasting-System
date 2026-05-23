import streamlit as st

from modules.charts import dual_currency_chart
from modules.i18n import tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_preprocessed_df
from modules.ui import dual_currency_selector
from modules.ui_theme import note, section

setup_page("bandingkan")

note(tp("bandingkan", "note"))
(iso1, col1, name1), (iso2, col2, name2) = dual_currency_selector()
if iso1 == iso2:
    st.warning(tp("bandingkan", "same_warn"))
    st.stop()

section(tp("bandingkan", "chart"))
st.plotly_chart(
    dual_currency_chart(get_preprocessed_df(col1), get_preprocessed_df(col2), iso1, iso2, name1, name2),
    width="stretch",
)
page_footer()
