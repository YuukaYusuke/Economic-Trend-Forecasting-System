from __future__ import annotations

import math

import numpy as np
import pandas as pd


MAX_RETURN = 0.015
SMOOTH_WEIGHT = 0.2
TIMEFRAME_SCALE = {"1D": 1.0, "7D": 1.8, "30D": 2.6}


def pct_change(current: float, reference: float) -> float:
    if not reference or math.isnan(reference):
        return 0.0
    return (current - reference) / abs(reference)


def recent_return_signal(history, timeframe: str = "1D") -> tuple[float, float]:
    series = pd.Series(history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0, 0.0

    tail = returns.tail(30)
    short = returns.tail(7)
    volatility = float(tail.std()) if len(tail) > 1 else 0.0
    momentum = float(short.ewm(span=min(5, len(short)), adjust=False).mean().iloc[-1])
    scale = TIMEFRAME_SCALE.get(timeframe, 1.0)
    return momentum * scale, volatility * 100


def _dynamic_cap(volatility_pct: float, timeframe: str = "1D") -> float:
    scale = TIMEFRAME_SCALE.get(timeframe, 1.0)
    vol_cap = max(0.0015, (volatility_pct / 100) * 1.35 * scale)
    return min(MAX_RETURN, vol_cap)


def detect_regime(volatility_pct: float) -> dict:
    if volatility_pct >= 1.0:
        return {"label": "High volatility", "aggressiveness": 0.55}
    if volatility_pct >= 0.5:
        return {"label": "Elevated volatility", "aggressiveness": 0.75}
    return {"label": "Normal volatility", "aggressiveness": 1.0}


def calibrate_next_prediction(
    raw_pred: float,
    reference_price: float,
    history=None,
    *,
    timeframe: str = "1D",
) -> dict:
    """
    Guardrail untuk prediksi kurs next-step:
    1. ubah prediksi menjadi return,
    2. blend dengan momentum historis pendek,
    3. batasi return ekstrem,
    4. smoothing kuat ke harga terakhir.
    """
    if not reference_price:
        return {
            "value": float(raw_pred),
            "return_pct": 0.0,
            "raw_return_pct": 0.0,
            "adjusted_return_pct": 0.0,
            "cap_pct": MAX_RETURN * 100,
            "volatility_pct": 0.0,
        }

    momentum, volatility_pct = recent_return_signal([] if history is None else history, timeframe=timeframe)
    raw_return = pct_change(float(raw_pred), float(reference_price))
    cap = _dynamic_cap(volatility_pct, timeframe=timeframe)
    regime = detect_regime(volatility_pct)

    clipped_model = float(np.clip(raw_return, -cap, cap))
    clipped_momentum = float(np.clip(momentum, -cap, cap))
    blended_return = (0.68 * clipped_model) + (0.32 * clipped_momentum)
    bounded_return = float(np.clip(blended_return * regime["aggressiveness"], -cap, cap))

    bounded_pred = float(reference_price) * (1 + bounded_return)
    final_pred = (1 - SMOOTH_WEIGHT) * float(reference_price) + SMOOTH_WEIGHT * bounded_pred
    final_return = pct_change(final_pred, float(reference_price))

    return {
        "value": float(final_pred),
        "return_pct": float(final_return * 100),
        "raw_return_pct": float(raw_return * 100),
        "adjusted_return_pct": float(bounded_return * 100),
        "cap_pct": float(cap * 100),
        "volatility_pct": float(volatility_pct),
        "regime": regime["label"],
        "aggressiveness": float(regime["aggressiveness"]),
    }


def calibrate_series_predictions(actual, predicted) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return predicted

    calibrated = []
    for i, raw in enumerate(predicted):
        reference = actual[i - 1] if i > 0 else actual[i]
        history_start = max(0, i - 30)
        history = actual[history_start : i + 1]
        adjusted = calibrate_next_prediction(raw, reference, history)
        calibrated.append(adjusted["value"])
    return np.asarray(calibrated, dtype=float)
