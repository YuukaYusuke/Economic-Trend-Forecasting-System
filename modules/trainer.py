"""
Model training with BiLSTM + Ensemble (RF + XGBoost) and TimeSeriesSplit validation.
"""

import sys

import numpy as np
import pandas as pd

from modules.config import BATCH_SIZE, DATA_PATH, TRAIN_EPOCHS, WINDOW
from modules.currency_registry import CURRENCY_COLUMNS
from modules.data_pipeline import (
    inverse_target,
    load_dataset,
    prepare_model_data,
    preprocess_dataset,
    returns_to_prices,
)
from modules.lstm_model import build_model
from modules.model_storage import load_model_artifacts, save_model_artifacts
from modules.ensemble_models import (
    build_random_forest_model,
    build_xgboost_model,
    EnsembleRegressor,
    TimeSeriesEnsembleValidator,
)


def get_trainable_currencies():
    cols = set(pd.read_csv(DATA_PATH, nrows=0).columns)
    items = []
    for column, (iso, name) in CURRENCY_COLUMNS.items():
        if column in cols:
            items.append({"column": column, "iso": iso, "name": name})
    return sorted(items, key=lambda x: x["name"])


def build_training_bundle(currency_column: str, epochs: int = TRAIN_EPOCHS, use_attention: bool = False):
    """
    Build and train BiLSTM + Ensemble models with TimeSeriesSplit validation.
    
    Args:
        currency_column: Currency column name
        epochs: Number of training epochs
        use_attention: Whether to use attention mechanism
    
    Returns:
        (bilstm_model, ensemble, artifacts, data, df)
    """
    raw = load_dataset(DATA_PATH)
    df = preprocess_dataset(raw, currency_column)
    data = prepare_model_data(df, window=WINDOW)

    # Build and train BiLSTM
    bilstm = build_model((data["X_train"].shape[1], data["X_train"].shape[2]), use_attention=use_attention)
    validation = (data["X_test"], data["y_test"]) if len(data["X_test"]) else None
    fit_kwargs = {}
    if validation is not None:
        fit_kwargs["validation_data"] = validation
    
    bilstm.fit(
        data["X_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False,
        **fit_kwargs,
    )

    # Train Random Forest
    X_train_features = data["X_train"][:, -1, :]  # Last timestep features
    X_test_features = data["X_test"][:, -1, :] if len(data["X_test"]) else np.empty((0, X_train_features.shape[1]))
    
    rf_model = build_random_forest_model(n_estimators=200, max_depth=8, min_samples_leaf=5)
    rf_model.fit(X_train_features, data["y_train"].reshape(-1))
    
    # Train XGBoost
    xgb_model = build_xgboost_model(n_estimators=100, max_depth=6, learning_rate=0.1)
    xgb_model.fit(X_train_features, data["y_train"].reshape(-1))
    
    # Validate ensemble using TimeSeriesSplit
    validator = TimeSeriesEnsembleValidator(n_splits=3)
    validation_result = validator.validate_models(
        X_train_features,
        data["X_train"],
        data["y_train"],
        bilstm_model=bilstm,
        rf_params={"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
        xgb_params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    )
    
    # Create ensemble
    ensemble = EnsembleRegressor(bilstm_model=bilstm, rf_model=rf_model, xgb_model=xgb_model)
    ensemble.set_weights(validation_result["ensemble_weights"])
    
    # Compute residual quantiles for confidence estimation
    residual_quantiles = {"q10": -0.002, "q90": 0.002}
    if len(data["X_test"]):
        ensemble_pred = ensemble.predict(X_test_features, data["X_test"])
        actual_returns = inverse_target(data["target_scaler"], data["y_test"])
        pred_returns = inverse_target(data["target_scaler"], ensemble_pred)
        residuals = actual_returns - pred_returns
        residual_quantiles = {
            "q10": float(np.quantile(residuals, 0.1)),
            "q50": float(np.quantile(residuals, 0.5)),
            "q90": float(np.quantile(residuals, 0.9)),
        }

    artifacts = {
        "feature_scaler": data["feature_scaler"],
        "target_scaler": data["target_scaler"],
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "ensemble_weights": validation_result["ensemble_weights"],
        "latest_sequence": data["latest_sequence"],
        "latest_features": data["latest_features"],
        "feature_columns": data["feature_columns"],
        "residual_quantiles": residual_quantiles,
        "validation_result": {
            "model_errors": validation_result["model_errors"],
            "ensemble_weights": validation_result["ensemble_weights"],
        },
    }
    return bilstm, ensemble, artifacts, data, df


def train_single(currency_column: str, epochs: int = TRAIN_EPOCHS, force: bool = False) -> str:
    """
    Train a single currency model. Return: 'skipped' | 'trained' | 'failed'
    """
    if not force:
        loaded = load_model_artifacts(currency_column, epochs=epochs)
        if loaded:
            return "skipped"

    try:
        bilstm, ensemble, artifacts, _, _ = build_training_bundle(currency_column, epochs=epochs)
        save_model_artifacts(bilstm, artifacts, currency_column, epochs=epochs)
        return "trained"
    except Exception as exc:
        print(f"  [ERROR] {currency_column}: {exc}", file=sys.stderr)
        return "failed"


def train_all(epochs: int = TRAIN_EPOCHS, force: bool = False):
    """Train all available currency models."""
    items = get_trainable_currencies()
    stats = {"trained": 0, "skipped": 0, "failed": 0}

    print(f"Melatih {len(items)} model (epoch={epochs}, force={force})...\n")
    for i, item in enumerate(items, 1):
        col = item["column"]
        label = f"{item['iso']} - {item['name']}"
        status = train_single(col, epochs=epochs, force=force)
        stats[status] += 1
        icon = {"trained": "+", "skipped": "=", "failed": "x"}[status]
        print(f"  [{i}/{len(items)}] ({icon}) {label}: {status}")

    print(f"\nSelesai - trained: {stats['trained']}, skipped: {stats['skipped']}, failed: {stats['failed']}")
    return stats

