import numpy as np

from modules.forecast_calibration import calibrate_next_prediction


def _predict_scaled_return(model, sequence):
    pred = model.predict(sequence, verbose=0)
    return np.asarray(pred, dtype=float).reshape(-1, 1)


def predict_return(model, target_scaler, sequence) -> float:
    pred_scaled = _predict_scaled_return(model, sequence)
    return float(target_scaler.inverse_transform(pred_scaled)[0][0])


def predict_tree_return(tree_model, target_scaler, latest_features) -> float | None:
    if tree_model is None:
        return None
    pred_scaled = tree_model.predict(latest_features).reshape(-1, 1)
    return float(target_scaler.inverse_transform(pred_scaled)[0][0])


def ensemble_return(lstm_return: float, tree_return: float | None) -> float:
    if tree_return is None:
        return float(lstm_return)
    return float((0.5 * lstm_return) + (0.5 * tree_return))


def postprocess_prediction(
    pred_return: float,
    last_price: float,
    history,
    *,
    timeframe: str = "1D",
    residual_quantiles: dict | None = None,
):
    raw_price = float(last_price) * (1 + float(pred_return))
    calibration = calibrate_next_prediction(raw_price, float(last_price), history, timeframe=timeframe)
    median = calibration["value"]

    residual_quantiles = residual_quantiles or {}
    low_q = float(residual_quantiles.get("q10", -abs(calibration["cap_pct"]) / 100))
    high_q = float(residual_quantiles.get("q90", abs(calibration["cap_pct"]) / 100))
    lower_raw = float(last_price) * (1 + float(pred_return) + low_q)
    upper_raw = float(last_price) * (1 + float(pred_return) + high_q)

    lower = calibrate_next_prediction(lower_raw, float(last_price), history, timeframe=timeframe)["value"]
    upper = calibrate_next_prediction(upper_raw, float(last_price), history, timeframe=timeframe)["value"]
    if lower > upper:
        lower, upper = upper, lower

    return {
        "raw_price": raw_price,
        "median": median,
        "lower_bound": lower,
        "upper_bound": upper,
        "calibration": calibration,
    }


def predict_next(model, artifacts: dict):
    lstm_return = predict_return(model, artifacts["target_scaler"], artifacts["latest_sequence"])
    tree_return = predict_tree_return(
        artifacts.get("tree_model"),
        artifacts["target_scaler"],
        artifacts["latest_features"],
    )
    return ensemble_return(lstm_return, tree_return)


def predict_next_with_live(model, artifacts: dict, live_rate: float, history):
    pred_return = predict_next(model, artifacts)
    return postprocess_prediction(
        pred_return,
        live_rate,
        history,
        residual_quantiles=artifacts.get("residual_quantiles"),
    )
