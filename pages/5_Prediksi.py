import streamlit as st

from modules.api import get_live_rate
from modules.charts import prediction_chart
from modules.confidence import prediction_confidence
from modules.export_report import report_dataframe, to_csv_bytes, to_pdf_bytes
from modules.evaluation import mae, mape, rmse
from modules.i18n import format_direction, t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("prediksi")

iso, column, name = currency_selector()
live = get_live_rate("USD", iso)
result = get_predictions(column, live_rate=live, iso=iso, name=name)

if not result["uses_live"]:
    st.error(tp("prediksi", "no_api"))
    st.stop()

live = result["live_rate"]
next_value = result["next_value"]
arah = result["prediksi_arah"]
level, conf_pct, _ = prediction_confidence(next_value, live)
conf_label = t(level)

note(tp("prediksi", "note"))

section(tp("prediksi", "result"))
c1, c2, c3, c4 = st.columns(4)
stat_card(t("live"), f"{live:,.4f}", "neutral")
stat_card(t("prediction"), f"{next_value:,.4f}", "neutral")
stat_card(t("direction"), format_direction(arah), "up" if arah == "NAIK" else "down")
stat_card(t("confidence"), conf_label, level if level != "medium" else "neutral")

st.success(tp("prediksi", "result_ok", iso=iso, name=name, dir=format_direction(arah), pct=conf_pct))

section(tp("prediksi", "chart"))
st.plotly_chart(prediction_chart(result["df"].tail(90), iso, live, next_value, arah), width="stretch")

section(tp("prediksi", "export"))
metrics = {
    "mae": f"{mae(result['actual'], result['predicted']):,.4f}",
    "rmse": f"{rmse(result['actual'], result['predicted']):,.4f}",
    "mape": f"{mape(result['actual'], result['predicted']):.2f}%",
}
rep = report_dataframe(iso, name, live, next_value, arah, conf_label, metrics)
ec1, ec2 = st.columns(2)
ec1.download_button(t("export_csv"), to_csv_bytes(rep), f"forecast_{iso}.csv", "text/csv")
try:
    ec2.download_button(t("export_pdf"), to_pdf_bytes(f"Forecast {iso}", rep), f"forecast_{iso}.pdf", "application/pdf")
except Exception:
    ec2.caption(tp("prediksi", "pdf_err"))

page_footer()
