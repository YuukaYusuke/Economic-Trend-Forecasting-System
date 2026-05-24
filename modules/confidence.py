import numpy as np


def prediction_confidence(predicted: float, reference: float) -> tuple[str, float, str]:
    """
    Returns: (level_id: strong|medium|weak, pct_diff, label_id)
    pct_diff = abs(predicted - reference) / reference * 100
    """
    if reference == 0:
        return "weak", 0.0, "weak"

    pct = abs(predicted - reference) / abs(reference) * 100
    if pct >= 1.5:
        level = "strong"
    elif pct >= 0.4:
        level = "medium"
    else:
        level = "weak"
    return level, pct, level


def volatility_based_confidence(
    recent_volatility: float,
    market_stress: float,
    residual_quantiles: dict,
    prediction: float,
) -> dict:
    """
    Compute confidence metrics based on volatility and market stress.
    
    Args:
        recent_volatility: Standard deviation of recent returns (e.g., 10-day vol)
        market_stress: Market stress index (0-1)
        residual_quantiles: Dict with 'q10', 'q50', 'q90' from validation residuals
        prediction: The predicted return value
    
    Returns:
        dict with:
            - confidence_level: 'high', 'medium', 'low'
            - confidence_score: float (0-1)
            - prediction_interval: (lower, upper) - 80% prediction interval
            - volatility_adjusted: bool - True if high volatility detected
    """
    # Base confidence inversely related to volatility
    vol_confidence = 1.0 / (1.0 + recent_volatility)
    
    # Stress-adjusted confidence
    stress_confidence = 1.0 - market_stress
    
    # Combined confidence
    base_confidence = 0.6 * vol_confidence + 0.4 * stress_confidence
    
    # Prediction interval using residual quantiles
    q10 = residual_quantiles.get("q10", -0.01)
    q90 = residual_quantiles.get("q90", 0.01)
    
    lower_bound = prediction + q10
    upper_bound = prediction + q90
    
    # Determine confidence level
    if base_confidence > 0.7:
        confidence_level = "high"
    elif base_confidence > 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Flag if volatility is high
    high_volatility = recent_volatility > 0.02  # ~2% daily volatility threshold
    
    return {
        "confidence_level": confidence_level,
        "confidence_score": float(base_confidence),
        "prediction_interval": (float(lower_bound), float(upper_bound)),
        "volatility_adjusted": bool(high_volatility),
        "market_stress": float(market_stress),
        "volatility": float(recent_volatility),
    }


def compute_prediction_interval(
    residuals: np.ndarray,
    prediction: float,
    confidence_level: float = 0.8,
) -> tuple:
    """
    Compute prediction interval based on residuals from validation.
    
    Args:
        residuals: Residuals from validation (predictions - actual)
        prediction: The predicted value
        confidence_level: Desired confidence level (default 0.8 = 80%)
    
    Returns:
        (lower_bound, upper_bound)
    """
    if len(residuals) == 0:
        return (prediction - 0.01, prediction + 0.01)
    
    # Use quantiles of residuals for prediction interval
    alpha = (1 - confidence_level) / 2
    lower_quantile = np.quantile(residuals, alpha)
    upper_quantile = np.quantile(residuals, 1 - alpha)
    
    return (prediction + lower_quantile, prediction + upper_quantile)

