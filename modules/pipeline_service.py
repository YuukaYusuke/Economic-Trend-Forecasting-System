import numpy as np
import streamlit as st

from modules.config import BATCH_SIZE, DATA_PATH, TRAIN_EPOCHS, TRAIN_RATIO, WINDOW
from modules.data_pipeline import (
    create_sequences,
    load_dataset,
    preprocess_dataset,
    scale_data,
    split_data,
)
from modules.lstm_model import build_model
from modules.model_storage import load_model_artifacts, save_model_artifacts
from modules.predict import predict_next, predict_next_with_live
from modules.settings import allow_runtime_training
from modules.trend import get_direction


class ModelsNotDeployedError(RuntimeError):
    """Model belum ada di disk dan pelatihan runtime dinonaktifkan (mode deploy)."""


@st.cache_data(show_spinner=False)
def get_preprocessed_df(currency_column: str):
    df = load_dataset(DATA_PATH)
    return preprocess_dataset(df, currency_column)


@st.cache_resource(show_spinner=False)
def get_trained_model(currency_column: str, _epochs: int = TRAIN_EPOCHS):
    df = get_preprocessed_df(currency_column)
    scaled, scaler = scale_data(df)
    X, y = create_sequences(scaled, window=WINDOW)
    X_train, y_train, X_test, y_test = split_data(X, y, ratio=TRAIN_RATIO)

    loaded = load_model_artifacts(currency_column, epochs=_epochs)
    if loaded:
        model, scaler = loaded
        return model, scaler, scaled, X_test, y_test, df

    if not allow_runtime_training():
        raise ModelsNotDeployedError(
            f"Model untuk {currency_column} tidak ditemukan. "
            "Latih dengan `python train_models.py` lalu commit folder `models/`."
        )

    model = build_model((X_train.shape[1], 1))
    model.fit(
        X_train,
        y_train,
        epochs=_epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    save_model_artifacts(model, scaler, currency_column, epochs=_epochs)
    return model, scaler, scaled, X_test, y_test, df


def get_predictions(
    currency_column: str,
    live_rate: float | None = None,
    iso: str = "",
    name: str = "",
):
    try:
        model, scaler, scaled, X_test, y_test, df = get_trained_model(currency_column)
    except ModelsNotDeployedError as exc:
        st.error(str(exc))
        st.stop()

    pred = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred).flatten()
    actual = scaler.inverse_transform(y_test).flatten()
    last_hist = float(df["rate"].iloc[-1])
    prev_hist = float(df["rate"].iloc[-2])
    uses_live = live_rate is not None

    if uses_live:
        next_value = predict_next_with_live(
            model, scaled, scaler, live_rate, window=WINDOW
        )
        prediksi_arah = get_direction(next_value, live_rate)
        trend_realtime = get_direction(live_rate, last_hist)
    else:
        next_value = predict_next(model, scaled, scaler, window=WINDOW)
        prediksi_arah = get_direction(next_value, last_hist)
        trend_realtime = None

    trend_dataset = get_direction(last_hist, prev_hist)

    confidence = None
    if uses_live:
        from modules.confidence import prediction_confidence

        level, pct_conf, _ = prediction_confidence(next_value, live_rate)
        confidence = {"level": level, "pct": pct_conf}
        if iso:
            try:
                from modules.prediction_log import append_prediction

                append_prediction(
                    iso, name or iso, live_rate, next_value, prediksi_arah, level, pct_conf
                )
            except Exception:
                pass

    return {
        "model": model,
        "scaler": scaler,
        "scaled": scaled,
        "df": df,
        "actual": actual,
        "predicted": pred,
        "next_value": next_value,
        "last_hist": last_hist,
        "prev_hist": prev_hist,
        "live_rate": live_rate,
        "uses_live": uses_live,
        "prediksi_arah": prediksi_arah,
        "trend_dataset": trend_dataset,
        "trend_realtime": trend_realtime,
        "confidence": confidence,
    }
