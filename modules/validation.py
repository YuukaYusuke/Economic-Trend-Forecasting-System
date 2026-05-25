"""
Validation utilities for model inputs and outputs.
Ensures data integrity throughout the prediction pipeline.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def validate_numeric(value, name: str, allow_none: bool = False, allow_zero: bool = True) -> float | None:
    """
    Validate and convert a value to numeric type.
    
    Args:
        value: Value to validate
        name: Field name for error messages
        allow_none: Whether None is acceptable
        allow_zero: Whether zero is acceptable
    
    Returns:
        Validated float value or None
    
    Raises:
        TypeError: If value is not numeric type
        ValueError: If value is NaN, Inf, or invalid
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None")
    
    if isinstance(value, bool):
        raise TypeError(f"{name} cannot be boolean")
    
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    
    numeric_value = float(value)
    
    if np.isnan(numeric_value):
        raise ValueError(f"{name} is NaN")
    
    if np.isinf(numeric_value):
        raise ValueError(f"{name} is Infinity")
    
    if numeric_value == 0 and not allow_zero:
        raise ValueError(f"{name} cannot be zero")
    
    return numeric_value


def validate_array(arr, name: str, expected_shape: tuple | None = None, dtype=np.float32) -> np.ndarray:
    """
    Validate array for model input.
    
    Args:
        arr: Array to validate
        name: Field name for error messages
        expected_shape: Expected shape (None to skip check)
        dtype: Expected data type
    
    Returns:
        Validated numpy array
    
    Raises:
        TypeError: If arr is not array-like
        ValueError: If shape or values are invalid
    """
    try:
        arr_np = np.asarray(arr, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} cannot be converted to array: {e}")
    
    if len(arr_np) == 0:
        raise ValueError(f"{name} is empty")
    
    if np.any(np.isnan(arr_np)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(arr_np)):
        raise ValueError(f"{name} contains Infinity values")
    
    if expected_shape is not None and arr_np.shape != expected_shape:
        raise ValueError(f"{name} has shape {arr_np.shape}, expected {expected_shape}")
    
    return arr_np


def validate_artifacts_dict(artifacts: dict) -> None:
    """
    Validate model artifacts dictionary structure.
    
    Args:
        artifacts: Artifacts dictionary from model storage
    
    Raises:
        KeyError: If required keys are missing
        TypeError: If artifact values have wrong type
    """
    required_keys = ["feature_scaler", "target_scaler"]
    missing_keys = [k for k in required_keys if k not in artifacts]
    
    if missing_keys:
        raise KeyError(f"Missing required artifact keys: {missing_keys}")
    
    if artifacts["feature_scaler"] is None:
        raise ValueError("feature_scaler cannot be None")
    
    if artifacts["target_scaler"] is None:
        raise ValueError("target_scaler cannot be None")


def validate_prediction_output(pred: dict) -> None:
    """
    Validate prediction output dictionary.
    
    Args:
        pred: Prediction dictionary from make_live_prediction()
    
    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    required_keys = ["lstm_return", "rf_return", "xgb_return", "ensemble_return"]
    missing_keys = [k for k in required_keys if k not in pred]
    
    if missing_keys:
        raise KeyError(f"Missing required prediction keys: {missing_keys}")
    
    for key in required_keys:
        try:
            validate_numeric(pred[key], f"prediction['{key}']", allow_none=False)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid {key} in prediction: {e}")
