"""
Model Evaluation Service - Handles model evaluation and performance metrics.
Part of Phase 2 refactoring.
"""

import logging
import numpy as np
from modules.data_pipeline import inverse_target, returns_to_prices
from modules.forecast_calibration import calibrate_series_predictions

logger = logging.getLogger(__name__)


def evaluate_ensemble_model(model, artifacts: dict, data: dict) -> dict:
    """
    Evaluate ensemble model on test set.
    
    Args:
        model: BiLSTM model
        artifacts: Model artifacts (scalers, tree models)
        data: Data dict with X_test, y_test, etc
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating ensemble model")
    
    # Get ensemble predictions
    pred_scaled = _ensemble_scaled_prediction(model, artifacts, data["X_test"])
    pred_returns = inverse_target(artifacts["target_scaler"], pred_scaled)
    actual_returns = inverse_target(artifacts["target_scaler"], data["y_test"])
    
    # Convert to prices
    predicted_prices = returns_to_prices(data["test_price_refs"], pred_returns)
    actual_prices = data["test_actual_prices"]
    calibrated_prices = calibrate_series_predictions(actual_prices, predicted_prices)

    # Calculate residuals and errors
    residual = actual_prices - calibrated_prices
    if len(residual):
        outlier_threshold = float(np.quantile(np.abs(residual), 0.95))
        max_error = float(np.max(np.abs(residual)))
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
    else:
        outlier_threshold = 0.0
        max_error = 0.0
        mae = 0.0
        rmse = 0.0
    
    outliers = np.abs(residual) >= outlier_threshold if outlier_threshold else np.zeros_like(residual, dtype=bool)

    result = {
        "actual_prices": actual_prices,
        "actual_returns": actual_returns,
        "predicted_prices": predicted_prices,
        "predicted_returns": pred_returns,
        "calibrated_prices": calibrated_prices,
        "max_error": max_error,
        "mae": mae,
        "rmse": rmse,
        "outliers": outliers,
        "outlier_threshold": outlier_threshold,
    }
    
    logger.info(f"Evaluation complete. RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    return result


def _ensemble_scaled_prediction(model, artifacts: dict, X):
    """
    Get ensemble prediction (average of LSTM + RF + XGBoost).
    
    Args:
        model: BiLSTM model
        artifacts: Model artifacts
        X: Input features (3D array for LSTM)
    
    Returns:
        Ensemble predictions
    """
    lstm_scaled = model.predict(X, verbose=False)
    rf_model = artifacts.get("rf_model")
    xgb_model = artifacts.get("xgb_model")
    
    if rf_model is None and xgb_model is None:
        return lstm_scaled
    
    ensemble_scaled = lstm_scaled.copy()
    count = 1
    
    if rf_model is not None:
        rf_scaled = rf_model.predict(X[:, -1, :]).reshape(-1, 1)
        ensemble_scaled = ensemble_scaled + rf_scaled
        count += 1
    
    if xgb_model is not None:
        xgb_scaled = xgb_model.predict(X[:, -1, :]).reshape(-1, 1)
        ensemble_scaled = ensemble_scaled + xgb_scaled
        count += 1
    
    return ensemble_scaled / count


def calculate_metrics_summary(eval_result: dict) -> dict:
    """
    Calculate summary metrics from evaluation result.
    
    Args:
        eval_result: Result from evaluate_ensemble_model()
    
    Returns:
        Summary metrics dict
    """
    metrics = {
        "total_samples": len(eval_result["actual_prices"]),
        "max_error": eval_result["max_error"],
        "mae": eval_result["mae"],
        "rmse": eval_result["rmse"],
        "outlier_count": int(eval_result["outliers"].sum()),
        "outlier_pct": float((eval_result["outliers"].sum() / len(eval_result["outliers"])) * 100),
    }
    return metrics
