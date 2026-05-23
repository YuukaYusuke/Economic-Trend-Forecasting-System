import numpy as np


def moving_average_predict(series: np.ndarray, window: int = 30) -> np.ndarray:
    """Prediksi naive: nilai MA(window) untuk setiap titik setelah window."""
    preds = []
    for i in range(len(series)):
        if i < window:
            preds.append(series[i])
        else:
            preds.append(np.mean(series[i - window : i]))
    return np.array(preds)
