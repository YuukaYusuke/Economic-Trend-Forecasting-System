import streamlit as st

from modules.i18n import is_rtl
from modules.navigation import build_navigation
from modules.page_shell import init_session, render_sidebar
from modules.ui_theme import inject_theme

st.set_page_config(
    page_title="Economic Trend Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()
inject_theme(st.session_state.dark_mode, rtl=is_rtl())
render_sidebar()

build_navigation().run()
