import pandas as pd
import streamlit as st

from modules.api import get_bulk_live_rates_usd
from modules.page_shell import page_footer, setup_page
from modules.ui_theme import note, section, stat_card
from modules.world_map import build_world_map_forecasts, compare_chart, projection_chart, world_choropleth


def _plotly_selected_country(event):
    if not event:
        return None
    selection = event.get("selection", {}) if isinstance(event, dict) else getattr(event, "selection", {})
    points = selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        return point.get("location") or point.get("customdata", [None])[0]
    return getattr(point, "location", None)


def _fmt_price(value):
    if pd.isna(value):
        return "-"
    return f"{value:,.4f}"


def _meter(label):
    width = {"Low": 32, "Medium": 64, "High": 100}.get(label, 32)
    color = {"Low": "#94a3b8", "Medium": "#f59e0b", "High": "#10b981"}.get(label, "#94a3b8")
    st.markdown(
        f"""
<div class="world-meter">
  <div class="world-meter-head"><span>Confidence meter</span><strong>{label}</strong></div>
  <div class="world-meter-track"><div class="world-meter-fill" style="width:{width}%;background:{color};"></div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


setup_page("world_map")

st.markdown(
    """
<style>
.world-detail {
    animation: worldFade 220ms ease-out;
}
@keyframes worldFade {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.world-meter {
    margin: 0.45rem 0 1rem 0;
}
.world-meter-head {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    color: #cbd5e1;
    font-size: 0.82rem;
    margin-bottom: 0.45rem;
}
.world-meter-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.24);
    overflow: hidden;
}
.world-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 240ms ease;
}
.world-insight {
    border: 1px solid rgba(45, 212, 191, 0.28);
    background: rgba(15, 23, 42, 0.74);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    color: #d1fae5;
    line-height: 1.55;
}
</style>
    """,
    unsafe_allow_html=True,
)

tf_col, refresh_col = st.columns([1, 1])
with tf_col:
    timeframe = st.segmented_control("Timeframe", ["1D", "7D", "30D"], default="1D")
with refresh_col:
    st.write("")
    st.write("")
    if st.button("Refresh map", type="primary"):
        build_world_map_forecasts.clear()
        get_bulk_live_rates_usd.clear()
        st.rerun()

with st.spinner("Nyiapin prediksi global dulu... sabar dikit ya."):
    live_rates = get_bulk_live_rates_usd()
    df = build_world_map_forecasts(live_rates, timeframe)

if df.empty:
    st.warning("Data map belum bisa dibuat. Cek koneksi API atau model yang tersedia dulu.")
    st.stop()

note(
    "Warna map memakai predicted return (%), bukan harga mentah. Prediksi dibatasi maksimal +/-1.5% "
    "dari harga terakhir, lalu dihaluskan dengan final_pred = 0.8 * last_price + 0.2 * pred."
)

selected_from_state = st.session_state.get("world_map_selected_country")
event = st.plotly_chart(
    world_choropleth(df, selected_from_state),
    width="stretch",
    key="world_map_choropleth",
    on_select="rerun",
    selection_mode="points",
    config={"displayModeBar": False, "scrollZoom": False},
)

clicked_country = _plotly_selected_country(event)
if clicked_country and clicked_country != selected_from_state:
    st.session_state.world_map_selected_country = clicked_country
    st.rerun()

options = df.sort_values(["country", "currency"])["row_id"].tolist()
labels = dict(zip(df["row_id"], df["country"] + " (" + df["currency"] + ")"))
manual = st.selectbox(
    "Detail negara",
    [""] + options,
    format_func=lambda key: "Klik negara di map atau pilih manual" if key == "" else labels.get(key, key),
)

selected = None
if manual:
    selected = df[df["row_id"] == manual].iloc[0]
    st.session_state.world_map_selected_country = selected["country_code"]
elif st.session_state.get("world_map_selected_country"):
    match = df[df["country_code"] == st.session_state.world_map_selected_country]
    if len(match):
        selected = match.iloc[0]

if selected is not None:
    st.markdown('<div class="world-detail">', unsafe_allow_html=True)
    section(f"Detail Panel - {selected['country']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stat_card("Negara", selected["country"], "neutral")
    with c2:
        stat_card("Mata uang", f"{selected['currency']} - {selected['currency_name']}", "neutral")
    with c3:
        stat_card("Harga sekarang", _fmt_price(selected["last_price"]), "neutral")
    with c4:
        stat_card("Prediksi", _fmt_price(selected["final_pred"]), "neutral")
    with c5:
        variant = "up" if selected["predicted_return"] > 0 else "down" if selected["predicted_return"] < 0 else "neutral"
        stat_card("% change", f"{selected['predicted_return']:+.3f}%", variant)

    left, right = st.columns([1.55, 1])
    with left:
        st.plotly_chart(projection_chart(selected), width="stretch", config={"displayModeBar": False})
    with right:
        section("Indicator")
        i1, i2, i3 = st.columns(3)
        with i1:
            stat_card("Trend", selected["trend"], "up" if selected["trend"] == "Bullish" else "down")
        with i2:
            stat_card("RSI", "-" if pd.isna(selected["rsi"]) else f"{selected['rsi']:.1f}", "neutral")
        with i3:
            stat_card("Volatility", f"{selected['volatility_label']}", "neutral")
        _meter(selected["confidence"])
        st.markdown(f'<div class="world-insight">{selected["insight"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Klik salah satu negara di map buat buka detail panelnya.")

section("Compare 2 Negara")
compare_pick = st.multiselect(
    "Pilih maksimal 2 negara",
    options,
    max_selections=2,
    format_func=lambda key: labels.get(key, key),
)
if len(compare_pick) == 2:
    compare_rows = df[df["row_id"].isin(compare_pick)]
    st.plotly_chart(compare_chart(compare_rows), width="stretch", config={"displayModeBar": False})
    st.dataframe(
        compare_rows[
            ["country", "currency", "last_price", "final_pred", "predicted_return", "confidence", "trend", "rsi", "volatility"]
        ],
        width="stretch",
        hide_index=True,
    )

page_footer()
