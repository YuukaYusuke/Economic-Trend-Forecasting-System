import numpy as np
import pandas as pd


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted):
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def max_error(actual, predicted):
    if len(actual) == 0:
        return 0.0
    return float(np.max(np.abs(actual - predicted)))


def walk_forward_metrics(actual, predicted, n_splits: int = 5):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    if n == 0:
        return pd.DataFrame()

    n_splits = max(1, min(n_splits, n))
    fold_size = max(1, n // n_splits)
    rows = []
    start = 0
    for fold in range(1, n_splits + 1):
        end = n if fold == n_splits else min(n, start + fold_size)
        if start >= end:
            break
        a = actual[start:end]
        p = predicted[start:end]
        residual = np.abs(a - p)
        threshold = np.quantile(residual, 0.95) if len(residual) else 0.0
        rows.append(
            {
                "Fold": fold,
                "Start idx": start,
                "End idx": end - 1,
                "MAE": mae(a, p),
                "MAPE": mape(a, p),
                "Max error": max_error(a, p),
                "Outliers": int((residual >= threshold).sum()) if threshold else 0,
            }
        )
        start = end
    return pd.DataFrame(rows)
