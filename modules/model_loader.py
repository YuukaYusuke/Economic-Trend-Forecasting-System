"""
Model Loading Service - Handles model loading, training, and artifact management.
Part of Phase 2 refactoring.
"""

import logging
import streamlit as st
from modules.config import DATA_PATH, TRAIN_EPOCHS, WINDOW
from modules.data_pipeline import load_dataset, preprocess_dataset, prepare_model_data, latest_model_inputs
from modules.model_storage import load_model_artifacts, save_model_artifacts
from modules.trainer import build_training_bundle
from modules.settings import allow_runtime_training

logger = logging.getLogger(__name__)


class ModelsNotDeployedError(RuntimeError):
    """Model belum ada di disk dan pelatihan runtime dinonaktifkan (mode deploy)."""


@st.cache_data(show_spinner=False, ttl=3600)
def get_preprocessed_df(currency_column: str):
    """
    Load and preprocess dataset.
    
    Args:
        currency_column: Currency column name
    
    Returns:
        Preprocessed dataframe
    """
    logger.info(f"Loading data for {currency_column}")
    df = load_dataset(DATA_PATH)
    return preprocess_dataset(df, currency_column)


@st.cache_resource(show_spinner=False)
def get_trained_model(currency_column: str, _epochs: int = TRAIN_EPOCHS):
    """
    Load trained model or train new one.
    
    Args:
        currency_column: Currency column name
        _epochs: Number of training epochs
    
    Returns:
        (model, artifacts, data, df)
    """
    logger.info(f"Getting trained model for {currency_column}")
    
    df = get_preprocessed_df(currency_column)
    data = prepare_model_data(df, window=WINDOW)

    # Try loading existing model
    loaded = load_model_artifacts(currency_column, epochs=_epochs)
    if loaded:
        logger.info(f"Loaded existing model for {currency_column}")
        model, artifacts = loaded
        sequence, latest_features, _ = latest_model_inputs(
            df, artifacts["feature_scaler"], window=WINDOW
        )
        artifacts = {
            **artifacts,
            "latest_sequence": sequence,
            "latest_features": latest_features
        }
        return model, artifacts, data, df

    # Check if runtime training allowed
    if not allow_runtime_training():
        raise ModelsNotDeployedError(
            f"Model untuk {currency_column} belum tersedia. "
            "Latih ulang dengan: python train_models.py --force"
        )

    # Train new model
    logger.info(f"Training new model for {currency_column}")
    model, ensemble, artifacts, data, df = build_training_bundle(
        currency_column, 
        epochs=_epochs
    )
    save_model_artifacts(model, artifacts, currency_column, epochs=_epochs)
    logger.info(f"Model training complete for {currency_column}")
    
    return model, artifacts, data, df


def reload_model(currency_column: str, epochs: int = TRAIN_EPOCHS):
    """
    Force reload model (clear cache and retrain).
    
    Args:
        currency_column: Currency column name
        epochs: Number of epochs
    """
    logger.warning(f"Force reloading model for {currency_column}")
    get_trained_model.clear()
    get_preprocessed_df.clear()
    return get_trained_model(currency_column, epochs)
