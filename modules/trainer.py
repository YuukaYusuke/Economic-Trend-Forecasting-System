"""Pelatihan model LSTM tanpa Streamlit — untuk skrip batch."""

import sys
from pathlib import Path

import pandas as pd

from modules.config import BATCH_SIZE, DATA_PATH, TRAIN_EPOCHS, WINDOW
from modules.currency_registry import CURRENCY_COLUMNS
from modules.data_pipeline import (
    create_sequences,
    load_dataset,
    preprocess_dataset,
    scale_data,
    split_data,
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


def train_single(currency_column: str, epochs: int = TRAIN_EPOCHS, force: bool = False) -> str:
    """Return: 'skipped' | 'trained' | 'failed'"""
    if not force:
        loaded = load_model_artifacts(currency_column, epochs=epochs)
        if loaded:
            return "skipped"

    try:
        df = load_dataset(DATA_PATH)
        df = preprocess_dataset(df, currency_column)
        if len(df) < WINDOW + 50:
            return "failed"

        scaled, scaler = scale_data(df)
        X, y = create_sequences(scaled, window=WINDOW)
        X_train, y_train, _, _ = split_data(X, y)

        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
        save_model_artifacts(model, scaler, currency_column, epochs=epochs)
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
        label = f"{item['iso']} — {item['name']}"
        status = train_single(col, epochs=epochs, force=force)
        stats[status] += 1
        icon = {"trained": "+", "skipped": "=", "failed": "x"}[status]
        print(f"  [{i}/{len(items)}] ({icon}) {label}: {status}")

    print(f"\nSelesai — trained: {stats['trained']}, skipped: {stats['skipped']}, failed: {stats['failed']}")
    return stats
