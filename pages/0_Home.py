import streamlit as st

from modules.currencies import get_selectable_currencies
from modules.dashboard_data import executive_summary
from modules.i18n import t, tp
from modules.page_shell import page_footer, setup_page
from modules.ui_theme import note, section, stat_card

setup_page("home")

currencies = get_selectable_currencies()
fav_opts = [c["iso"] for c in currencies] if currencies else ["IDR"]
fav = st.selectbox(tp("home", "fav_currency"), fav_opts, index=0)

with st.spinner(t("loading")):
    ex = executive_summary(fav)

section(tp("home", "exec_kpi"))
k1, k2, k3, k4 = st.columns(4)
k1.metric(tp("home", "api_count"), ex["api_count"])
k2.metric(t("models_ready"), f"{ex['trained']}/{ex['total_models']}")
if ex["pred"]:
    from modules.i18n import format_direction

    k3.metric(tp("home", "pred_today"), format_direction(ex["pred"]["arah"]))
    k4.metric(t("live"), f"{ex['pred']['live']:,.4f}")
else:
    k3.metric(tp("home", "pred_today"), "—")
    k4.metric(t("live"), "—")

if ex["vs_5y_pct"] is not None:
    stat_card(tp("home", "vs_5y"), f"{ex['vs_5y_pct']:+.2f}%", "up" if ex["vs_5y_pct"] > 0 else "down")

section(tp("home", "about"))
note(tp("home", "about_text"))

c1, c2 = st.columns(2)
with c1:
    st.markdown(f'<div class="fx-panel">{tp("home", "features")}</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="fx-panel">{tp("home", "flow")}</div>', unsafe_allow_html=True)

page_footer()
