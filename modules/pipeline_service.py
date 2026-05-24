import numpy as np
import streamlit as st

from modules.config import DATA_PATH, TRAIN_EPOCHS, WINDOW
from modules.data_pipeline import (
    inverse_target,
    latest_model_inputs,
    load_dataset,
    prepare_model_data,
    preprocess_dataset,
    returns_to_prices,
)
from modules.forecast_calibration import calibrate_series_predictions
from modules.model_storage import load_model_artifacts, save_model_artifacts
from modules.predict import ensemble_return, postprocess_prediction, predict_tree_return
from modules.settings import allow_runtime_training
from modules.trainer import build_training_bundle
from modules.trend import get_direction


class ModelsNotDeployedError(RuntimeError):
    """Model belum ada di disk dan pelatihan runtime dinonaktifkan (mode deploy)."""


@st.cache_data(show_spinner=False)
def get_preprocessed_df(currency_column: str):
    df = load_dataset(DATA_PATH)
    return preprocess_dataset(df, currency_column)


@st.cache_resource(show_spinner=False)
def get_trained_model(currency_column: str, _epochs: int = TRAIN_EPOCHS):
    df = get_preprocessed_df(currency_column)
    data = prepare_model_data(df, window=WINDOW)

    loaded = load_model_artifacts(currency_column, epochs=_epochs)
    if loaded:
        model, artifacts = loaded
        sequence, latest_features, _ = latest_model_inputs(
            df, artifacts["feature_scaler"], window=WINDOW
        )
        artifacts = {**artifacts, "latest_sequence": sequence, "latest_features": latest_features}
        return model, artifacts, data, df

    if not allow_runtime_training():
        raise ModelsNotDeployedError(
            f"Model untuk {currency_column} belum memakai pipeline return-based terbaru. "
            "Latih ulang dengan `python train_models.py --force` lalu commit folder `models/`."
        )

    model, artifacts, data, df = build_training_bundle(currency_column, epochs=_epochs)
    save_model_artifacts(model, artifacts, currency_column, epochs=_epochs)
    return model, artifacts, data, df


def _ensemble_scaled_prediction(model, artifacts: dict, X):
    lstm_scaled = model.predict(X, verbose=0)
    tree = artifacts.get("tree_model")
    if tree is None:
        return lstm_scaled
    tree_scaled = tree.predict(X[:, -1, :]).reshape(-1, 1)
    return (0.5 * lstm_scaled) + (0.5 * tree_scaled)


def _prediction_prices(model, artifacts: dict, data: dict):
    pred_scaled = _ensemble_scaled_prediction(model, artifacts, data["X_test"])
    pred_returns = inverse_target(artifacts["target_scaler"], pred_scaled)
    actual_returns = inverse_target(artifacts["target_scaler"], data["y_test"])
    predicted_prices = returns_to_prices(data["test_price_refs"], pred_returns)
    actual_prices = data["test_actual_prices"]
    calibrated_prices = calibrate_series_predictions(actual_prices, predicted_prices)

    residual = actual_prices - calibrated_prices
    if len(residual):
        outlier_threshold = float(np.quantile(np.abs(residual), 0.95))
    else:
        outlier_threshold = 0.0
    outliers = np.abs(residual) >= outlier_threshold if outlier_threshold else np.zeros_like(residual, dtype=bool)

    return {
        "actual_prices": actual_prices,
        "actual_returns": actual_returns,
        "predicted_prices": predicted_prices,
        "predicted_returns": pred_returns,
        "calibrated_prices": calibrated_prices,
        "max_error": float(np.max(np.abs(residual))) if len(residual) else 0.0,
        "outliers": outliers,
        "outlier_threshold": outlier_threshold,
    }


def _live_prediction(model, artifacts: dict, df, live_rate: float | None, timeframe: str = "1D"):
    reference = float(live_rate) if live_rate is not None else float(df["rate"].iloc[-1])
    sequence, latest_features, live_features_df = latest_model_inputs(
        df,
        artifacts["feature_scaler"],
        window=WINDOW,
        live_rate=reference if live_rate is not None else None,
    )
    lstm_scaled = model.predict(sequence, verbose=0)
    lstm_return = float(artifacts["target_scaler"].inverse_transform(lstm_scaled)[0][0])
    tree_return = predict_tree_return(
        artifacts.get("tree_model"),
        artifacts["target_scaler"],
        latest_features,
    )
    pred_return = ensemble_return(lstm_return, tree_return)
    processed = postprocess_prediction(
        pred_return,
        reference,
        live_features_df["rate"].tail(90),
        timeframe=timeframe,
        residual_quantiles=artifacts.get("residual_quantiles"),
    )
    return {
        "lstm_return": lstm_return,
        "tree_return": tree_return,
        "ensemble_return": pred_return,
        **processed,
    }


def get_predictions(
    currency_column: str,
    live_rate: float | None = None,
    iso: str = "",
    name: str = "",
    timeframe: str = "1D",
):
    try:
        model, artifacts, data, df = get_trained_model(currency_column)
    except ModelsNotDeployedError as exc:
        st.error(str(exc))
        st.stop()

    eval_pred = _prediction_prices(model, artifacts, data)
    last_hist = float(df["rate"].iloc[-1])
    prev_hist = float(df["rate"].iloc[-2])
    uses_live = live_rate is not None
    live_block = _live_prediction(model, artifacts, df, live_rate, timeframe=timeframe)

    reference = float(live_rate) if uses_live else last_hist
    next_value = live_block["median"]
    prediksi_arah = get_direction(next_value, reference)
    trend_realtime = get_direction(reference, last_hist) if uses_live else None
    trend_dataset = get_direction(last_hist, prev_hist)

    confidence = None
    if uses_live:
        from modules.confidence import prediction_confidence

        level, pct_conf, _ = prediction_confidence(next_value, reference)
        confidence = {"level": level, "pct": pct_conf}
        if iso:
            try:
                from modules.prediction_log import append_prediction

                append_prediction(
                    iso, name or iso, reference, next_value, prediksi_arah, level, pct_conf
                )
            except Exception:
                pass

    calibration = live_block["calibration"]
    quantiles = {
        "lower_bound": live_block["lower_bound"],
        "median": live_block["median"],
        "upper_bound": live_block["upper_bound"],
    }

    return {
        "model": model,
        "artifacts": artifacts,
        "df": df,
        "actual": eval_pred["actual_prices"],
        "actual_returns": eval_pred["actual_returns"],
        "predicted": eval_pred["predicted_prices"],
        "predicted_returns": eval_pred["predicted_returns"],
        "predicted_calibrated": eval_pred["calibrated_prices"],
        "max_error": eval_pred["max_error"],
        "outliers": eval_pred["outliers"],
        "outlier_threshold": eval_pred["outlier_threshold"],
        "raw_next_value": live_block["raw_price"],
        "next_value": next_value,
        "lower_bound": quantiles["lower_bound"],
        "median": quantiles["median"],
        "upper_bound": quantiles["upper_bound"],
        "quantiles": quantiles,
        "lstm_return": live_block["lstm_return"],
        "tree_return": live_block["tree_return"],
        "ensemble_return": live_block["ensemble_return"],
        "calibration": calibration,
        "last_hist": last_hist,
        "prev_hist": prev_hist,
        "live_rate": live_rate,
        "uses_live": uses_live,
        "prediksi_arah": prediksi_arah,
        "trend_dataset": trend_dataset,
        "trend_realtime": trend_realtime,
        "confidence": confidence,
    }
