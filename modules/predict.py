"""
Prediction using ensemble models (BiLSTM + RF + XGBoost).
"""

import numpy as np
from modules.forecast_calibration import calibrate_next_prediction
from modules.ensemble_models import EnsembleRegressor
from modules.confidence import volatility_based_confidence


def _predict_scaled_return(bilstm_model, X_lstm):
    """Get BiLSTM prediction."""
    pred = bilstm_model.predict(X_lstm, verbose=0)
    return np.asarray(pred, dtype=float).reshape(-1, 1)


def predict_lstm_return(bilstm_model, target_scaler, X_lstm) -> float:
    """Predict return using BiLSTM."""
    pred_scaled = _predict_scaled_return(bilstm_model, X_lstm)
    return float(target_scaler.inverse_transform(pred_scaled)[0][0])


def predict_rf_return(rf_model, target_scaler, X_features) -> float | None:
    """Predict return using Random Forest."""
    if rf_model is None:
        return None
    pred_scaled = rf_model.predict(X_features).reshape(-1, 1)
    return float(target_scaler.inverse_transform(pred_scaled)[0][0])


def predict_xgb_return(xgb_model, target_scaler, X_features) -> float | None:
    """Predict return using XGBoost."""
    if xgb_model is None:
        return None
    pred_scaled = xgb_model.predict(X_features).reshape(-1, 1)
    return float(target_scaler.inverse_transform(pred_scaled)[0][0])


def ensemble_return(
    lstm_return: float,
    rf_return: float | None,
    xgb_return: float | None,
    weights: dict | None = None,
) -> float:
    """
    Combine predictions from all models using ensemble weights.
    
    Args:
        lstm_return: BiLSTM prediction
        rf_return: Random Forest prediction (optional)
        xgb_return: XGBoost prediction (optional)
        weights: dict with keys 'bilstm', 'rf', 'xgb'
    
    Returns:
        Ensemble prediction
    """
    weights = weights or {"bilstm": 0.4, "rf": 0.3, "xgb": 0.3}
    
    total_weight = 0
    weighted_sum = 0
    
    weighted_sum += lstm_return * weights.get("bilstm", 0.4)
    total_weight += weights.get("bilstm", 0.4)
    
    if rf_return is not None:
        weighted_sum += rf_return * weights.get("rf", 0.3)
        total_weight += weights.get("rf", 0.3)
    
    if xgb_return is not None:
        weighted_sum += xgb_return * weights.get("xgb", 0.3)
        total_weight += weights.get("xgb", 0.3)
    
    return float(weighted_sum / total_weight) if total_weight > 0 else float(lstm_return)


def postprocess_prediction(
    pred_return: float,
    last_price: float,
    history,
    *,
    timeframe: str = "1D",
    residual_quantiles: dict | None = None,
):
    """Add calibration and prediction interval to forecast."""
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


def predict_next(bilstm_model, artifacts: dict):
    """
    Make ensemble prediction for next period.
    """
    lstm_return = predict_lstm_return(bilstm_model, artifacts["target_scaler"], artifacts["latest_sequence"])
    rf_return = predict_rf_return(
        artifacts.get("rf_model"),
        artifacts["target_scaler"],
        artifacts["latest_features"],
    )
    xgb_return = predict_xgb_return(
        artifacts.get("xgb_model"),
        artifacts["target_scaler"],
        artifacts["latest_features"],
    )
    
    weights = artifacts.get("ensemble_weights", {"bilstm": 0.4, "rf": 0.3, "xgb": 0.3})
    return ensemble_return(lstm_return, rf_return, xgb_return, weights=weights)


def predict_next_with_live(bilstm_model, artifacts: dict, live_rate: float, history):
    """
    Make ensemble prediction with live rate and return calibrated forecast.
    """
    pred_return = predict_next(bilstm_model, artifacts)
    return postprocess_prediction(
        pred_return,
        live_rate,
        history,
        residual_quantiles=artifacts.get("residual_quantiles"),
    )


def predict_with_confidence(
    bilstm_model,
    artifacts: dict,
    live_rate: float,
    history,
    recent_volatility: float,
    market_stress: float,
):
    """
    Make prediction with volatility-based confidence estimation.
    
    Args:
        bilstm_model: BiLSTM model
        artifacts: Model artifacts dict
        live_rate: Current exchange rate
        history: Historical price data
        recent_volatility: Recent volatility (e.g., 10-day)
        market_stress: Market stress index (0-1)
    
    Returns:
        dict with prediction, confidence level, and prediction interval
    """
    # Get ensemble prediction
    forecast = predict_next_with_live(bilstm_model, artifacts, live_rate, history)
    
    # Add confidence metrics
    confidence = volatility_based_confidence(
        recent_volatility=recent_volatility,
        market_stress=market_stress,
        residual_quantiles=artifacts.get("residual_quantiles", {}),
        prediction=forecast["raw_price"],
    )
    
    return {
        **forecast,
        "confidence": confidence,
    }

