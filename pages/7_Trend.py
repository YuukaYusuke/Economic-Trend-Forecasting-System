import streamlit as st

from modules.api import get_live_rate
from modules.i18n import format_direction, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.ui import currency_selector
from modules.ui_theme import section, stat_card

setup_page("trend")

iso, column, name = currency_selector()
live = get_live_rate("USD", iso)
r = get_predictions(column, live_rate=live, iso=iso, name=name)

section(tp("trend", "compare"))
c1, c2, c3 = st.columns(3)
stat_card(tp("trend", "hist"), format_direction(r["trend_dataset"]), "neutral")
stat_card(tp("trend", "pred"), format_direction(r["prediksi_arah"]), "up" if r["prediksi_arah"] == "NAIK" else "down")
stat_card(
    tp("trend", "live_hist"),
    format_direction(r["trend_realtime"]) if r["trend_realtime"] else "—",
    "neutral",
)
page_footer()
