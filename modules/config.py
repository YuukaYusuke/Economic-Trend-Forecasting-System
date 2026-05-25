"""
Configuration Module - MERGED VERSION
Contains all original settings + Phase 2 caching optimizations.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_DIR = str(MODELS_PATH)  # For compatibility
LOGS_PATH = PROJECT_ROOT / "logs"

# Create necessary directories
LOGS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)


# ============================================================================
# CURRENCIES & DISPLAY NAMES
# ============================================================================

# Currency column mappings with display names
DISPLAY_NAMES = {
    "USDIDR": "USD → IDR (US Dollar to Indonesian Rupiah)",
    "EURUSD": "EUR → USD (Euro to US Dollar)",
    "GBPUSD": "GBP → USD (British Pound to US Dollar)",
    "JPYUSD": "JPY → USD (Japanese Yen to US Dollar)",
    "AUDUSD": "AUD → USD (Australian Dollar to US Dollar)",
    "CADUSD": "CAD → USD (Canadian Dollar to US Dollar)",
    "CHFUSD": "CHF → USD (Swiss Franc to US Dollar)",
    "CNHUSD": "CNH → USD (Chinese Yuan to US Dollar)",
    "SGDUSD": "SGD → USD (Singapore Dollar to US Dollar)",
    "HKDUSD": "HKD → USD (Hong Kong Dollar to US Dollar)",
    "INRUSD": "INR → USD (Indian Rupee to US Dollar)",
    "THBUSD": "THB → USD (Thai Baht to US Dollar)",
}

# Alternative mapping
CURRENCY_COLUMNS = {
    "USDIDR": ("USD", "US Dollar"),
    "EURUSD": ("EUR", "Euro"),
    "GBPUSD": ("GBP", "British Pound"),
    "JPYUSD": ("JPY", "Japanese Yen"),
    "AUDUSD": ("AUD", "Australian Dollar"),
    "CADUSD": ("CAD", "Canadian Dollar"),
    "CHFUSD": ("CHF", "Swiss Franc"),
    "CNHUSD": ("CNH", "Chinese Yuan"),
    "SGDUSD": ("SGD", "Singapore Dollar"),
    "HKDUSD": ("HKD", "Hong Kong Dollar"),
    "INRUSD": ("INR", "Indian Rupee"),
    "THBUSD": ("THB", "Thai Baht"),
}


# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

WINDOW = 60  # Look-back window for LSTM
BATCH_SIZE = 32
TRAIN_EPOCHS = 50
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1

# Hyperparameter ranges (for Optuna)
LSTM_UNITS_RANGE = (32, 256)
DROPOUT_RANGE = (0.1, 0.5)
LEARNING_RATE_RANGE = (1e-4, 1e-2)

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 5

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1


# ============================================================================
# STREAMLIT CACHING STRATEGY (PHASE 2)
# ============================================================================

CACHE_SETTINGS = {
    # Data loading (changes rarely)
    "data_loading_ttl": 86400,  # 24 hours
    
    # Model loading (changes rarely)
    "model_loading_ttl": None,  # Permanent (use cache_resource)
    
    # Predictions (updates frequently)
    "prediction_ttl": 300,  # 5 minutes
    
    # Map data (medium frequency)
    "map_data_ttl": 3600,  # 1 hour
    
    # Evaluation results (changes rarely)
    "evaluation_ttl": 3600,  # 1 hour
}

# Streamlit session state keys
SESSION_KEYS = {
    "selected_currency": "USDIDR",
    "live_rate": None,
    "last_update": None,
    "error_message": None,
}


# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Confidence level thresholds (for prediction_confidence function)
CONFIDENCE_THRESHOLDS = {
    "strong": 1.5,    # >= 1.5% difference
    "medium": 0.4,    # >= 0.4% difference
    "weak": 0.0,      # < 0.4% difference
}

# Volatility thresholds (for risk levels)
VOLATILITY_THRESHOLDS = {
    "low": 0.02,      # < 2% volatility
    "medium": 0.05,   # 2-5% volatility
    "high": 0.10,     # 5-10% volatility
    "extreme": 0.10,  # > 10% volatility
}

# Forecast calibration settings (from forecast_calibration.py)
CALIBRATION_MAX_RETURN = 0.015        # Max allowed return cap (1.5%)
CALIBRATION_SMOOTH_WEIGHT = 0.2       # Smoothing weight for final prediction
CALIBRATION_TIMEFRAME_SCALE = {       # Timeframe volatility scaling
    "1D": 1.0,
    "7D": 1.8,
    "30D": 2.6,
}

# Model prediction blending weights
PREDICTION_BLEND_WEIGHTS = {
    "model": 0.68,          # Model prediction weight
    "momentum": 0.32,       # Momentum historical weight
}

# Recent return signal parameters
RETURN_SIGNAL_TAIL_DAYS = 30          # Look-back days for volatility
RETURN_SIGNAL_SHORT_DAYS = 7          # Look-back days for momentum
RETURN_SIGNAL_EWM_SPAN = 5            # Exponential weighted mean span

# Dynamic cap parameters
DYNAMIC_CAP_MIN_VOL = 0.0015          # Minimum volatility cap
DYNAMIC_CAP_VOL_MULTIPLIER = 1.35     # Volatility multiplier for cap
DYNAMIC_CAP_VOL_THRESHOLD = {
    "high": 1.0,                       # High volatility threshold
    "elevated": 0.5,                   # Elevated volatility threshold
}
DYNAMIC_CAP_AGGRESSIVENESS = {        # Regime aggressiveness factors
    "high": 0.55,
    "elevated": 0.75,
    "normal": 1.0,
}


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_PATH / "app.log"
ERROR_LOG_FILE = LOGS_PATH / "errors.log"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# RUNTIME SETTINGS
# ============================================================================

# Allow runtime training (False for production/deploy)
ALLOW_RUNTIME_TRAINING = os.getenv("ALLOW_RUNTIME_TRAINING", "True").lower() == "true"

# Features to use
FEATURES_TO_USE = [
    "rate",
    "sma_7",
    "sma_30",
    "rsi_14",
    "macd",
    "volatility_10",
]

# Lookback period for volatility calculation
VOLATILITY_LOOKBACK = 10  # days


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

STREAMLIT_CONFIG = {
    "page_title": "Economic Trend Forecasting",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


# ============================================================================
# DEBUG & DEVELOPMENT
# ============================================================================

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SHOW_TIMINGS = os.getenv("SHOW_TIMINGS", "False").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "False").lower() == "true"


# ============================================================================
# HELPER FUNCTIONS (PHASE 2)
# ============================================================================

def get_cache_ttl(cache_type: str) -> int | None:
    """
    Get cache TTL for specific type.
    
    Args:
        cache_type: Type of cache ('data', 'model', 'prediction', etc)
    
    Returns:
        TTL in seconds or None for permanent cache
    """
    return CACHE_SETTINGS.get(f"{cache_type}_ttl", 300)


def get_display_name(currency_column: str) -> str:
    """
    Get display name for currency column.
    
    Args:
        currency_column: Column name (e.g., 'USDIDR')
    
    Returns:
        Display name (e.g., 'USD → IDR (US Dollar to Indonesian Rupiah)')
    """
    return DISPLAY_NAMES.get(currency_column, currency_column)


def get_currency_info(currency_column: str) -> tuple:
    """
    Get currency info (ISO code, full name).
    
    Args:
        currency_column: Column name
    
    Returns:
        (iso_code, currency_name)
    """
    return CURRENCY_COLUMNS.get(currency_column, (currency_column, currency_column))


def validate_config():
    """Validate configuration on startup."""
    # Check data exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    # Check models directory writable
    try:
        MODELS_PATH.mkdir(exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot write to models dir: {e}")
    
    # Check logs directory writable
    try:
        LOGS_PATH.mkdir(exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot write to logs dir: {e}")
    
    print("✓ Configuration validation passed")


if __name__ == "__main__":
    validate_config()