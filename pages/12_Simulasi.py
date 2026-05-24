import streamlit as st

from modules.api import get_live_rate
from modules.confidence import prediction_confidence
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("simulasi")

iso, column, name = currency_selector()
live_real = get_live_rate("USD", iso) or 0.0

note(tp("simulasi", "note"))
pct = st.slider(tp("simulasi", "slider"), -10.0, 10.0, 0.0, 0.1)
sim_live = live_real * (1 + pct / 100) if live_real else 0.0

result = get_predictions(column, live_rate=sim_live)
pred = result["next_value"]
arah = result["prediksi_arah"]
level, conf_pct, _ = prediction_confidence(pred, sim_live)
cal = result.get("calibration", {})

section(tp("simulasi", "result"))
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card(tp("simulasi", "live_orig"), f"{live_real:,.4f}", "neutral")
with c2:
    stat_card(tp("simulasi", "live_sim"), f"{sim_live:,.4f}", "neutral")
with c3:
    stat_card(tp("simulasi", "pred_val"), f"{pred:,.4f}", "neutral")
with c4:
    stat_card(t("direction"), format_direction(arah), "up" if arah == "NAIK" else "down")

st.markdown(
    f'<div class="fx-panel">{tp("simulasi", "pred_info", val=pred, conf=t(level), pct=conf_pct)}</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Raw LSTM: {result.get('raw_next_value', pred):,.4f} · "
    f"return final: {cal.get('return_pct', 0):+.3f}% · batas: ±{cal.get('cap_pct', 0):.3f}%"
)
page_footer()
