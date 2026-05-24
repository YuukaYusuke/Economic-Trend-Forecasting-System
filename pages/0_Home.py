import streamlit as st

from modules.currencies import get_selectable_currencies
from modules.dashboard_data import executive_summary
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.ui_theme import note, section, stat_card

setup_page("home")

currencies = get_selectable_currencies()
fav_opts = [c["iso"] for c in currencies] if currencies else ["IDR"]
fav = st.selectbox(tp("home", "fav_currency"), fav_opts, index=0)

with st.spinner(t("loading")):
    ex = executive_summary(fav)

section(tp("home", "exec_kpi"))
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card(tp("home", "api_count"), str(ex["api_count"]), "neutral")
with c2:
    stat_card(t("models_ready"), f"{ex['trained']}/{ex['total_models']}", "neutral")

if ex["pred"]:
    with c3:
        stat_card(
            tp("home", "pred_today"),
            format_direction(ex["pred"]["arah"]),
            "up" if ex["pred"]["arah"] == "NAIK" else "down",
        )
    with c4:
        stat_card(t("live"), f"{ex['pred']['live']:,.4f}", "neutral")
else:
    with c3:
        stat_card(tp("home", "pred_today"), "-", "neutral")
    with c4:
        stat_card(t("live"), "-", "neutral")

if ex["vs_5y_pct"] is not None:
    stat_card(tp("home", "vs_5y"), f"{ex['vs_5y_pct']:+.2f}%", "up" if ex["vs_5y_pct"] > 0 else "down")

if ex["pred"]:
    section("Forecast Snapshot")
    p1, p2, p3 = st.columns(3)
    with p1:
        stat_card("Prediksi harga", f"{ex['pred']['pred']:,.4f}", "neutral")
    with p2:
        change = ex["pred"].get("change_pct") or 0.0
        stat_card("Return prediksi", f"{change:+.3f}%", "up" if change > 0 else "down")
    with p3:
        stat_card("Guardrail", f"+/-{(ex['pred'].get('cap_pct') or 0):.3f}%", "neutral")
    note(
        "Prediksi di dashboard sudah dikalibrasi supaya lebih kalem: LSTM tetap dipakai, tapi return ekstrem "
        "dipotong dan hasil akhirnya dismoothing ke kurs live."
    )

section(tp("home", "about"))
note(tp("home", "about_text"))

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f'<div class="fx-panel">{tp("home", "features")}</div>', unsafe_allow_html=True)
with p2:
    st.markdown(f'<div class="fx-panel">{tp("home", "flow")}</div>', unsafe_allow_html=True)
with p3:
    st.markdown(
        '<div class="fx-panel"><strong>Global map</strong><br>Choropleth prediksi global sekarang ada di halaman Peta Dunia. '
        'Dipisah biar beranda tetap ringan tapi aksesnya tetap cepat.</div>',
        unsafe_allow_html=True,
    )
    try:
        st.page_link("pages/10_World_Map.py", label="Buka Peta Dunia")
    except Exception:
        st.caption("Buka lewat sidebar: Peta Dunia")

page_footer()
