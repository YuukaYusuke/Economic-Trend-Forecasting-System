"""Pelatihan model LSTM tanpa Streamlit - untuk skrip batch."""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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


def get_trainable_currencies():
    cols = set(pd.read_csv(DATA_PATH, nrows=0).columns)
    items = []
    for column, (iso, name) in CURRENCY_COLUMNS.items():
        if column in cols:
            items.append({"column": column, "iso": iso, "name": name})
    return sorted(items, key=lambda x: x["name"])


def build_training_bundle(currency_column: str, epochs: int = TRAIN_EPOCHS):
    raw = load_dataset(DATA_PATH)
    df = preprocess_dataset(raw, currency_column)
    data = prepare_model_data(df, window=WINDOW)

    model = build_model((data["X_train"].shape[1], data["X_train"].shape[2]))
    validation = (data["X_test"], data["y_test"]) if len(data["X_test"]) else None
    fit_kwargs = {}
    if validation is not None:
        fit_kwargs["validation_data"] = validation
    model.fit(
        data["X_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False,
        **fit_kwargs,
    )

    tree_model = RandomForestRegressor(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    tree_model.fit(data["X_train"][:, -1, :], data["y_train"].reshape(-1))

    residual_quantiles = {"q10": -0.002, "q90": 0.002}
    if len(data["X_test"]):
        lstm_scaled = model.predict(data["X_test"], verbose=0)
        tree_scaled = tree_model.predict(data["X_test"][:, -1, :]).reshape(-1, 1)
        ensemble_scaled = (0.5 * lstm_scaled) + (0.5 * tree_scaled)
        pred_returns = inverse_target(data["target_scaler"], ensemble_scaled)
        actual_returns = inverse_target(data["target_scaler"], data["y_test"])
        residuals = actual_returns - pred_returns
        residual_quantiles = {
            "q10": float(np.quantile(residuals, 0.1)),
            "q50": float(np.quantile(residuals, 0.5)),
            "q90": float(np.quantile(residuals, 0.9)),
        }

    artifacts = {
        "feature_scaler": data["feature_scaler"],
        "target_scaler": data["target_scaler"],
        "tree_model": tree_model,
        "latest_sequence": data["latest_sequence"],
        "latest_features": data["latest_features"],
        "feature_columns": data["feature_columns"],
        "residual_quantiles": residual_quantiles,
    }
    return model, artifacts, data, df


def train_single(currency_column: str, epochs: int = TRAIN_EPOCHS, force: bool = False) -> str:
    """Return: 'skipped' | 'trained' | 'failed'"""
    if not force:
        loaded = load_model_artifacts(currency_column, epochs=epochs)
        if loaded:
            return "skipped"

    try:
        model, artifacts, _, _ = build_training_bundle(currency_column, epochs=epochs)
        save_model_artifacts(model, artifacts, currency_column, epochs=epochs)
        return "trained"
    except Exception as exc:
        print(f"  [ERROR] {currency_column}: {exc}", file=sys.stderr)
        return "failed"


def train_all(epochs: int = TRAIN_EPOCHS, force: bool = False):
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
