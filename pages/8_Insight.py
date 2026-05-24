import streamlit as st

from modules.api import get_live_rate
from modules.evaluation import mape
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.prediction_log import load_log
from modules.ui import currency_selector
from modules.ui_theme import section

setup_page("insight")

iso, column, name = currency_selector()
live = get_live_rate("USD", iso)
r = get_predictions(column, live_rate=live, iso=iso, name=name)
err = mape(r["actual"], r["predicted_calibrated"])

if r["uses_live"]:
    conf = r.get("confidence") or {}
    conf_lbl = t(conf.get("level", "weak")) if conf.get("level") else "—"
    st.markdown(
        f'<div class="fx-panel">{tp("insight", "summary", iso=iso, name=name, dir=format_direction(r["prediksi_arah"]), live=live, pred=r["next_value"], conf=conf_lbl, conf_pct=conf.get("pct", 0), mape=err)}</div>',
        unsafe_allow_html=True,
    )
else:
    st.warning(tp("insight", "no_api"))

section(tp("insight", "log"))
log = load_log(50)
if len(log):
    st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)
else:
    st.caption(tp("insight", "log_empty"))

page_footer()
