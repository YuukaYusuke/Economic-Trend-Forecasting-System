"""
Prediction Service - Handles live predictions and forecasting.
Part of Phase 2 refactoring.
"""

import logging
import streamlit as st
from modules.config import WINDOW
from modules.data_pipeline import latest_model_inputs
from modules.predict import predict_rf_return, predict_xgb_return, ensemble_return, postprocess_prediction
from modules.confidence import prediction_confidence
from modules.prediction_log import append_prediction
from modules.trend import get_direction
from modules.validation import validate_numeric, validate_artifacts_dict, validate_prediction_output

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def make_live_prediction(
    model,
    _artifacts: dict,
    df,
    live_rate: float | None = None,
    timeframe: str = "1D"
) -> dict:
    """
    Make live prediction with current/live rate.
    
    Args:
        model: BiLSTM model
        artifacts: Model artifacts
        df: Full dataframe
        live_rate: Current live rate (optional)
        timeframe: Time period ('1D', '1H', etc)
    
    Returns:
        Dict with predictions, bounds, returns
    """
    logger.info(f"Making live prediction with live_rate={live_rate}")
    
    # Validate inputs
    validate_artifacts_dict(_artifacts)
    if live_rate is not None:
        live_rate = validate_numeric(live_rate, "live_rate", allow_none=False, allow_zero=False)
    
    reference = float(live_rate) if live_rate is not None else float(df["rate"].iloc[-1])
    
    # Get latest sequence and features
    sequence, latest_features, live_features_df = latest_model_inputs(
        df,
        _artifacts["feature_scaler"],
        window=WINDOW,
        live_rate=reference if live_rate is not None else None,
    )
    
    # BiLSTM prediction
    lstm_scaled = model.predict(sequence, verbose=False)
    lstm_return = float(_artifacts["target_scaler"].inverse_transform(lstm_scaled)[0][0])
    
    # RF prediction
    rf_return = predict_rf_return(
        _artifacts.get("rf_model"),
        _artifacts["target_scaler"],
        latest_features,
    )
    
    # XGBoost prediction
    xgb_return = predict_xgb_return(
        _artifacts.get("xgb_model"),
        _artifacts["target_scaler"],
        latest_features,
    )
    
    # Ensemble prediction (3 models)
    pred_return = ensemble_return(lstm_return, rf_return, xgb_return)
    
    # Post-process with calibration
    processed = postprocess_prediction(
        pred_return,
        reference,
        live_features_df["rate"].tail(90),
        timeframe=timeframe,
        residual_quantiles=_artifacts.get("residual_quantiles"),
    )
    
    result = {
        "lstm_return": lstm_return,
        "rf_return": rf_return,
        "xgb_return": xgb_return,
        "ensemble_return": pred_return,
        **processed,
    }
    
    logger.info(f"Prediction made: {pred_return:.6f}")
    return result


def get_full_predictions(
    model,
    _artifacts: dict,
    data: dict,
    df,
    live_rate: float | None = None,
    iso: str = "",
    name: str = "",
    timeframe: str = "1D",
) -> dict:
    """
    Get complete prediction output with all metrics.
    
    Args:
        model: BiLSTM model
        artifacts: Model artifacts
        data: Training data dict
        df: Full dataframe
        live_rate: Live rate (optional)
        iso: Currency ISO code (optional)
        name: Currency name (optional)
        timeframe: Time period
    
    Returns:
        Complete prediction dict
    """
    from modules.evaluator import evaluate_ensemble_model
    
    logger.info(f"Getting full predictions for {iso or 'currency'}")
    
    # Evaluate model
    eval_pred = evaluate_ensemble_model(model, _artifacts, data)
    
    # Get historical data
    last_hist = float(df["rate"].iloc[-1])
    prev_hist = float(df["rate"].iloc[-2])
    
    # Get live predictions
    uses_live = live_rate is not None
    live_block = make_live_prediction(model, _artifacts, df, live_rate, timeframe=timeframe)
    
    # Get direction predictions
    reference = float(live_rate) if uses_live else last_hist
    next_value = live_block["median"]
    prediksi_arah = get_direction(next_value, reference)
    trend_realtime = get_direction(reference, last_hist) if uses_live else None
    trend_dataset = get_direction(last_hist, prev_hist)
    
    # Get confidence if live
    confidence = None
    if uses_live:
        level, pct_conf, _ = prediction_confidence(next_value, reference)
        confidence = {"level": level, "pct": pct_conf}
        
        # Log prediction if ISO provided
        if iso:
            try:
                append_prediction(
                    iso, 
                    name or iso,
                    reference,
                    next_value,
                    prediksi_arah,
                    level,
                    pct_conf
                )
                logger.info(f"Prediction logged for {iso}")
            except Exception as e:
                logger.warning(f"Could not log prediction: {e}")
    
    # Prepare quantiles
    calibration = live_block["calibration"]
    quantiles = {
        "lower_bound": live_block["lower_bound"],
        "median": live_block["median"],
        "upper_bound": live_block["upper_bound"],
    }
    
    # Complete result
    result = {
        "model": model,
        "artifacts": _artifacts,
        "df": df,
        # Evaluation metrics
        "actual": eval_pred["actual_prices"],
        "actual_returns": eval_pred["actual_returns"],
        "predicted": eval_pred["predicted_prices"],
        "predicted_returns": eval_pred["predicted_returns"],
        "predicted_calibrated": eval_pred["calibrated_prices"],
        "max_error": eval_pred["max_error"],
        "outliers": eval_pred["outliers"],
        "outlier_threshold": eval_pred["outlier_threshold"],
        # Prediction values
        "raw_next_value": live_block["raw_price"],
        "next_value": next_value,
        "lower_bound": quantiles["lower_bound"],
        "median": quantiles["median"],
        "upper_bound": quantiles["upper_bound"],
        "quantiles": quantiles,
        # Model returns
        "lstm_return": live_block["lstm_return"],
        "rf_return": live_block["rf_return"],
        "xgb_return": live_block["xgb_return"],
        "ensemble_return": live_block["ensemble_return"],
        # Calibration
        "calibration": calibration,
        # Historical
        "last_hist": last_hist,
        "prev_hist": prev_hist,
        "live_rate": live_rate,
        "uses_live": uses_live,
        # Trends
        "prediksi_arah": prediksi_arah,
        "trend_dataset": trend_dataset,
        "trend_realtime": trend_realtime,
        # Confidence
        "confidence": confidence,
    }
    
    logger.info("Full predictions retrieved successfully")
    return result
