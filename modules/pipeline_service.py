"""
Pipeline Service - Main service that uses model_loader, evaluator, and predictor.
Part of Phase 2 refactoring - CLEANER VERSION.
"""

import logging
import streamlit as st
from modules.model_loader import get_trained_model, ModelsNotDeployedError, get_preprocessed_df

logger = logging.getLogger(__name__)


def get_predictions(
    currency_column: str,
    live_rate: float | None = None,
    iso: str = "",
    name: str = "",
    timeframe: str = "1D",
) -> dict:
    """
    Get complete predictions for a currency.
    
    This is the main entry point. It handles:
    1. Loading/training models
    2. Evaluating on test set
    3. Making live predictions
    4. Logging results
    
    Args:
        currency_column: Column name (e.g., 'USDIDR')
        live_rate: Current rate (optional)
        iso: ISO code (optional, for logging)
        name: Currency name (optional, for logging)
        timeframe: Time period ('1D', '1H', etc)
    
    Returns:
        Dictionary with all prediction data and metrics
    
    Raises:
        ModelsNotDeployedError: If models not available and runtime training disabled
    """
    logger.info(f"Getting predictions for {currency_column}")
    
    # Load models and data
    try:
        model, artifacts, data, df = get_trained_model(currency_column)
    except ModelsNotDeployedError as exc:
        logger.error(f"Model not deployed: {exc}")
        st.error(str(exc))
        st.stop()
    
    # Import predictor here to avoid circular imports
    from modules.predictor import get_full_predictions
    
    # Get all predictions
    predictions = get_full_predictions(
        model,
        artifacts,  # This will be passed as _artifacts parameter
        data,
        df,
        live_rate=live_rate,
        iso=iso,
        name=name,
        timeframe=timeframe,
    )
    
    logger.info(f"Predictions complete for {currency_column}")
    return predictions
