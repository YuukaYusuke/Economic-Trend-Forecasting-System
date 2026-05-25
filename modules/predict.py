"""
Prediction using ensemble models (BiLSTM + RF + XGBoost) - WITH INPUT VALIDATION.
"""

import logging
import numpy as np
from modules.forecast_calibration import calibrate_next_prediction
from modules.ensemble_models import EnsembleRegressor
from modules.confidence import volatility_based_confidence

logger = logging.getLogger(__name__)


def _validate_numeric(value, name, allow_none=False):
    """Validate that value is numeric."""
    if allow_none and value is None:
        return True
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"{name} is NaN or Inf")
    return True


def _predict_scaled_return(bilstm_model, X_lstm):
    """Get BiLSTM prediction."""
    if bilstm_model is None:
        raise ValueError("BiLSTM model is None")
    pred = bilstm_model.predict(X_lstm, verbose=0)
    return np.asarray(pred, dtype=float).reshape(-1, 1)


def predict_lstm_return(bilstm_model, target_scaler, X_lstm) -> float:
    """Predict return using BiLSTM."""
    if bilstm_model is None:
        raise ValueError("BiLSTM model is None")
    if target_scaler is None:
        raise ValueError("Target scaler is None")
    
    pred_scaled = _predict_scaled_return(bilstm_model, X_lstm)
    result = float(target_scaler.inverse_transform(pred_scaled)[0][0])
    _validate_numeric(result, "LSTM return")
    return result


def predict_rf_return(rf_model, target_scaler, X_features) -> float | None:
    """Predict return using Random Forest."""
    if rf_model is None:
        return None
    if target_scaler is None:
        raise ValueError("Target scaler is None")
    if X_features is None:
        raise ValueError("X_features is None")
    
    pred_scaled = rf_model.predict(X_features).reshape(-1, 1)
    result = float(target_scaler.inverse_transform(pred_scaled)[0][0])
    _validate_numeric(result, "RF return")
    return result


def predict_xgb_return(xgb_model, target_scaler, X_features) -> float | None:
    """Predict return using XGBoost."""
    if xgb_model is None:
        return None
    if target_scaler is None:
        raise ValueError("Target scaler is None")
    if X_features is None:
        raise ValueError("X_features is None")
    
    pred_scaled = xgb_model.predict(X_features).reshape(-1, 1)
    result = float(target_scaler.inverse_transform(pred_scaled)[0][0])
    _validate_numeric(result, "XGBoost return")
    return result


def ensemble_return(
    lstm_return: float,
    rf_return: float | None,
    xgb_return: float | None,
    weights: dict | None = None,
) -> float:
    """
    Combine predictions from all models using ensemble weights.
    
    Args:
        lstm_return: BiLSTM prediction (required)
        rf_return: Random Forest prediction (optional)
        xgb_return: XGBoost prediction (optional)
        weights: dict with keys 'bilstm', 'rf', 'xgb' (optional)
    
    Returns:
        Ensemble prediction
        
    Raises:
        TypeError: If inputs are not numeric
        ValueError: If inputs are NaN/Inf
    """
    # Validate inputs
    _validate_numeric(lstm_return, "lstm_return")
    _validate_numeric(rf_return, "rf_return", allow_none=True)
    _validate_numeric(xgb_return, "xgb_return", allow_none=True)
    
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
    
    if total_weight <= 0:
        logger.warning("total_weight is 0, returning LSTM return")
        return float(lstm_return)
    
    result = float(weighted_sum / total_weight)
    _validate_numeric(result, "ensemble_return")
    return result


def postprocess_prediction(
    pred_return: float,
    last_price: float,
    history,
    *,
    timeframe: str = "1D",
    residual_quantiles: dict | None = None,
):
    """
    Add calibration and prediction interval to forecast.
    
    Args:
        pred_return: Predicted return (float)
        last_price: Current/last price (float)
        history: Historical price series
        timeframe: Time period ('1D', '1H', etc)
        residual_quantiles: Dict with q10, q90 (optional)
    
    Returns:
        Dict with raw_price, median, lower_bound, upper_bound, calibration
    """
    _validate_numeric(pred_return, "pred_return")
    _validate_numeric(last_price, "last_price")
    
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
    
    Args:
        bilstm_model: Trained BiLSTM model
        artifacts: Dictionary with scaler, features, and optional tree models
    
    Returns:
        Ensemble return prediction
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
    
    Args:
        bilstm_model: Trained BiLSTM model
        artifacts: Model artifacts dictionary
        live_rate: Current live exchange rate
        history: Historical price data
    
    Returns:
        Dictionary with calibrated forecast and bounds
    """
    _validate_numeric(live_rate, "live_rate")
    
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
    _validate_numeric(live_rate, "live_rate")
    _validate_numeric(recent_volatility, "recent_volatility")
    _validate_numeric(market_stress, "market_stress")
    
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
