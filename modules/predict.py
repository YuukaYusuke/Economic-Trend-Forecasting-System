import numpy as np


def predict_next(model, data, scaler, window=30):
    """Prediksi dari window historis saja (fallback jika API tidak tersedia)."""
    last_seq = data[-window:]
    last_seq = np.reshape(last_seq, (1, window, 1))
    pred = model.predict(last_seq, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])


def predict_next_with_live(model, scaled, scaler, live_rate, window=30):
    """
    Prediksi berikutnya dengan window = (window-1) historis + kurs live terakhir.
  Acuan arah: bandingkan hasil prediksi dengan live_rate.
    """
    live_scaled = scaler.transform([[live_rate]])[0][0]
    hist_part = scaled[-(window - 1) :].reshape(-1)
    seq = np.concatenate([hist_part, [live_scaled]])
    seq = np.reshape(seq, (1, window, 1))
    pred = model.predict(seq, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])
