import streamlit as st
import pandas as pd

from modules.baseline import moving_average_predict
from modules.charts import evaluation_with_baseline, residual_chart
from modules.evaluation import mae, mape, rmse, walk_forward_metrics
from modules.i18n import t, tp
from modules.page_shell import page_footer, setup_page
from modules.pipeline_service import get_predictions
from modules.ui import currency_selector
from modules.ui_theme import note, section, stat_card

setup_page("evaluasi")

iso, column, name = currency_selector()
result = get_predictions(column, iso=iso, name=name)
actual = result["actual"]
raw_predicted = result["predicted"]
predicted = result["predicted_calibrated"]

w = min(30, max(5, len(actual) // 4))
ma_pred = moving_average_predict(actual, window=w)
l_mae, raw_mae, b_mae = mae(actual, predicted), mae(actual, raw_predicted), mae(actual, ma_pred)
improve = ((b_mae - l_mae) / b_mae * 100) if b_mae else 0
raw_delta = ((raw_mae - l_mae) / raw_mae * 100) if raw_mae else 0

note(tp("evaluasi", "note", iso=iso, name=name))
note(
    "Evaluasi ini pakai prediksi yang sudah dikalibrasi. Clue-nya: untuk kurs, baseline sederhana sering kuat, "
    "jadi model perlu dibandingkan dengan MA dan versi raw-nya juga."
)

section(tp("evaluasi", "metrics"))
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(t("metric_mae"), f"{l_mae:,.4f}")
c2.metric(t("metric_rmse"), f"{rmse(actual, predicted):,.4f}")
c3.metric(t("metric_mape"), f"{mape(actual, predicted):.2f}%")
c4.metric(tp("evaluasi", "compare"), f"{improve:+.1f}%", tp("evaluasi", "compare_hint") if improve > 0 else tp("evaluasi", "ma_better"))
c5.metric("vs Raw LSTM", f"{raw_delta:+.1f}%")
c6.metric("Max error", f"{result['max_error']:,.4f}")

with st.expander("Raw LSTM vs calibrated"):
    st.dataframe(
        {
            "Metrik": ["MAE", "RMSE", "MAPE"],
            "Raw LSTM": [
                f"{raw_mae:,.4f}",
                f"{rmse(actual, raw_predicted):,.4f}",
                f"{mape(actual, raw_predicted):.2f}%",
            ],
            "Calibrated": [
                f"{l_mae:,.4f}",
                f"{rmse(actual, predicted):,.4f}",
                f"{mape(actual, predicted):.2f}%",
            ],
            "Moving Average": [
                f"{b_mae:,.4f}",
                f"{rmse(actual, ma_pred):,.4f}",
                f"{mape(actual, ma_pred):.2f}%",
            ],
        },
        width="stretch",
        hide_index=True,
    )

with st.expander("Walk-forward validation"):
    wf = walk_forward_metrics(actual, predicted, n_splits=5)
    st.dataframe(wf, width="stretch", hide_index=True)

section(tp("evaluasi", "chart_model"))
st.plotly_chart(evaluation_with_baseline(actual, predicted, ma_pred, iso), width="stretch")

section(tp("evaluasi", "chart_residual"))
st.plotly_chart(residual_chart(actual, predicted), width="stretch")

section("Outlier Detection")
outlier_mask = result["outliers"]
outlier_count = int(outlier_mask.sum())
st.caption(f"Outlier terdeteksi: {outlier_count} titik (threshold residual p95: {result['outlier_threshold']:,.4f})")
if outlier_count:
    idx = [i for i, flag in enumerate(outlier_mask) if flag]
    outlier_df = pd.DataFrame(
        {
            "Index": idx,
            "Actual": actual[outlier_mask],
            "Predicted": predicted[outlier_mask],
            "Error": (actual - predicted)[outlier_mask],
        }
    )
    st.dataframe(outlier_df.tail(20), width="stretch", hide_index=True)

page_footer()
