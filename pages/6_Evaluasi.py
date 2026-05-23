import streamlit as st

from modules.baseline import moving_average_predict
from modules.charts import evaluation_with_baseline, residual_chart
from modules.evaluation import mae, mape, rmse
from modules.i18n import t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("evaluasi")

iso, column, name = currency_selector()
result = get_predictions(column, iso=iso, name=name)
actual = result["actual"]
predicted = result["predicted"]

w = min(30, max(5, len(actual) // 4))
ma_pred = moving_average_predict(actual, window=w)
l_mae, b_mae = mae(actual, predicted), mae(actual, ma_pred)
improve = ((b_mae - l_mae) / b_mae * 100) if b_mae else 0

note(tp("evaluasi", "note", iso=iso, name=name))

section(tp("evaluasi", "metrics"))
c1, c2, c3, c4 = st.columns(4)
c1.metric(t("metric_mae"), f"{l_mae:,.4f}")
c2.metric(t("metric_rmse"), f"{rmse(actual, predicted):,.4f}")
c3.metric(t("metric_mape"), f"{mape(actual, predicted):.2f}%")
c4.metric(tp("evaluasi", "compare"), f"{improve:+.1f}%", tp("evaluasi", "compare_hint") if improve > 0 else tp("evaluasi", "ma_better"))

section(tp("evaluasi", "chart_model"))
st.plotly_chart(evaluation_with_baseline(actual, predicted, ma_pred, iso), width="stretch")

section(tp("evaluasi", "chart_residual"))
st.plotly_chart(residual_chart(actual, predicted), width="stretch")

page_footer()
