import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "return_1",
    "ma5",
    "ma20",
    "ma50",
    "volatility_10",
    "volatility_20",
    "momentum_10",
    "rsi",
    "lag_1",
    "lag_2",
    "lag_5",
    "lag_10",
]


def load_dataset(path: str):
    return pd.read_csv(path)


def preprocess_dataset(df, currency):
    """Ambil satu mata uang; konversi kolom *_to_usd menjadi kurs USD per unit."""
    if currency not in df.columns:
        raise ValueError(f"Kolom '{currency}' tidak ditemukan di dataset.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["date", currency]].rename(columns={currency: "rate"})
    df = df.dropna(subset=["date", "rate"])
    df = df[df["rate"] > 0]
    df = df.sort_values("date").reset_index(drop=True)

    # Dataset: currency_to_usd -> balik menjadi USD per 1 unit currency.
    df["rate"] = 1 / df["rate"]
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(0, 100)


def engineer_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["return_1"] = out["rate"].pct_change()
    out["ma5"] = out["rate"].rolling(5).mean()
    out["ma20"] = out["rate"].rolling(20).mean()
    out["ma50"] = out["rate"].rolling(50).mean()
    out["volatility_10"] = out["return_1"].rolling(10).std()
    out["volatility_20"] = out["return_1"].rolling(20).std()
    out["momentum_10"] = out["rate"] - out["rate"].shift(10)
    out["rsi"] = _rsi(out["rate"])
    for lag in (1, 2, 5, 10):
        out[f"lag_{lag}"] = out["rate"].shift(lag)

    if include_target:
        # Target: next daily return, bukan harga langsung.
        out["target_return"] = out["return_1"].shift(-1)
    out = out.replace([np.inf, -np.inf], np.nan)
    required = [*FEATURE_COLUMNS, "rate"]
    if include_target:
        required.append("target_return")
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def _windowed(features: np.ndarray, target: np.ndarray, prices: np.ndarray, window: int):
    X, y, price_refs, actual_prices = [], [], [], []
    for i in range(window, len(features)):
        X.append(features[i - window : i])
        y.append(target[i])
        price_refs.append(prices[i - 1])
        actual_prices.append(prices[i])
    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float).reshape(-1, 1),
        np.asarray(price_refs, dtype=float),
        np.asarray(actual_prices, dtype=float),
    )


def prepare_model_data(df: pd.DataFrame, window: int = 60, ratio: float = 0.8):
    """
    Build supervised data tanpa leakage:
    - split time-series dulu,
    - StandardScaler fit hanya dari train rows,
    - target adalah return berikutnya.
    """
    features_df = engineer_features(df, include_target=True)
    if len(features_df) <= window + 5:
        raise ValueError("Data terlalu pendek untuk window dan feature engineering.")

    split_row = int(len(features_df) * ratio)
    split_row = max(window + 1, min(split_row, len(features_df) - 2))

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(features_df.loc[: split_row - 1, FEATURE_COLUMNS])
    target_scaler.fit(features_df.loc[: split_row - 1, ["target_return"]])

    scaled_features = feature_scaler.transform(features_df[FEATURE_COLUMNS])
    scaled_target = target_scaler.transform(features_df[["target_return"]])
    prices = features_df["rate"].to_numpy(dtype=float)

    X, y, price_refs, actual_prices = _windowed(scaled_features, scaled_target.reshape(-1), prices, window)
    row_indices = np.arange(window, len(features_df))
    train_mask = row_indices < split_row

    latest_features = feature_scaler.transform(features_df[FEATURE_COLUMNS].tail(window))
    latest_sequence = latest_features.reshape(1, window, len(FEATURE_COLUMNS))

    return {
        "features_df": features_df,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_test": X[~train_mask],
        "y_test": y[~train_mask],
        "train_price_refs": price_refs[train_mask],
        "test_price_refs": price_refs[~train_mask],
        "train_actual_prices": actual_prices[train_mask],
        "test_actual_prices": actual_prices[~train_mask],
        "latest_sequence": latest_sequence,
        "latest_features": latest_features[-1].reshape(1, -1),
        "feature_columns": FEATURE_COLUMNS,
        "window": window,
    }


def latest_model_inputs(df: pd.DataFrame, feature_scaler: StandardScaler, window: int = 60, live_rate: float | None = None):
    source = df.copy()
    if live_rate is not None and live_rate > 0:
        next_date = source["date"].max() + pd.Timedelta(days=1)
        source = pd.concat(
            [source, pd.DataFrame([{"date": next_date, "rate": float(live_rate)}])],
            ignore_index=True,
        )
    features_df = engineer_features(source, include_target=False)
    if len(features_df) < window:
        raise ValueError("Data terlalu pendek untuk membuat latest sequence.")
    scaled = feature_scaler.transform(features_df[FEATURE_COLUMNS].tail(window))
    return scaled.reshape(1, window, len(FEATURE_COLUMNS)), scaled[-1].reshape(1, -1), features_df


def returns_to_prices(reference_prices, predicted_returns):
    refs = np.asarray(reference_prices, dtype=float)
    returns = np.asarray(predicted_returns, dtype=float).reshape(-1)
    return refs * (1 + returns)


def inverse_target(target_scaler: StandardScaler, values):
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return target_scaler.inverse_transform(arr).reshape(-1)


def scale_data(df):
    """Backward-compatible helper untuk modul lama. Pakai prepare_model_data untuk training baru."""
    scaler = StandardScaler()
    rates = df["rate"].to_numpy(dtype=float).reshape(-1, 1)
    scaled = scaler.fit_transform(rates)
    return scaled, scaler


def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def split_data(X, y, ratio=0.8):
    split = int(len(X) * ratio)
    return X[:split], y[:split], X[split:], y[split:]
