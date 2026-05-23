import pickle
from pathlib import Path

from modules.config import MODELS_DIR, TRAIN_EPOCHS, WINDOW


def _paths(currency_column: str):
    base = Path(MODELS_DIR)
    base.mkdir(exist_ok=True)
    safe = currency_column.replace("/", "_")
    return (
        base / f"{safe}.keras",
        base / f"{safe}_meta.pkl",
    )


def save_model_artifacts(model, scaler, currency_column: str, epochs: int = TRAIN_EPOCHS):
    model_path, meta_path = _paths(currency_column)
    model.save(model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(
            {"scaler": scaler, "epochs": epochs, "window": WINDOW, "column": currency_column},
            f,
        )


def load_model_artifacts(currency_column: str, epochs: int = TRAIN_EPOCHS):
    from tensorflow.keras.models import load_model

    model_path, meta_path = _paths(currency_column)
    if not model_path.exists() or not meta_path.exists():
        return None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if meta.get("epochs") != epochs or meta.get("window") != WINDOW:
        return None

    model = load_model(model_path)
    return model, meta["scaler"]
