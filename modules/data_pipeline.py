import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

    # Dataset: currency_to_usd → balik menjadi USD per 1 unit currency
    df["rate"] = 1 / df["rate"]
    return df


def scale_data(df):
    scaler = MinMaxScaler()
    rates = df["rate"].to_numpy(dtype=float).reshape(-1, 1)
    scaled = scaler.fit_transform(rates)
    return scaled, scaler


def create_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def split_data(X, y, ratio=0.8):
    split = int(len(X) * ratio)
    return X[:split], y[:split], X[split:], y[split:]