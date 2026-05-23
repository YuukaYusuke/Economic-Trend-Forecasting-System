import streamlit as st

from modules.api import get_live_rate
from modules.confidence import prediction_confidence
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import ModelsNotDeployedError, get_trained_model
from modules.predict import predict_next_with_live
from modules.trend import get_direction
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("simulasi")

iso, column, name = currency_selector()
live_real = get_live_rate("USD", iso) or 0.0

note(tp("simulasi", "note"))
pct = st.slider(tp("simulasi", "slider"), -10.0, 10.0, 0.0, 0.1)
sim_live = live_real * (1 + pct / 100) if live_real else 0.0

try:
    model, scaler, scaled, *_ = get_trained_model(column)
except ModelsNotDeployedError as exc:
    st.error(str(exc))
    st.stop()
pred = predict_next_with_live(model, scaled, scaler, sim_live)
arah = get_direction(pred, sim_live)
level, conf_pct, _ = prediction_confidence(pred, sim_live)

section(tp("simulasi", "result"))
c1, c2, c3 = st.columns(3)
stat_card(tp("simulasi", "live_orig"), f"{live_real:,.4f}", "neutral")
stat_card(tp("simulasi", "live_sim"), f"{sim_live:,.4f}", "neutral")
stat_card(tp("simulasi", "pred_val"), format_direction(arah), "up" if arah == "NAIK" else "down")

st.markdown(
    f'<div class="fx-panel">{tp("simulasi", "pred_info", val=pred, conf=t(level), pct=conf_pct)}</div>',
    unsafe_allow_html=True,
)
page_footer()
