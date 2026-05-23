import csv
import logging
from datetime import datetime
from pathlib import Path

from modules.config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "data" / "predictions_log.csv"
logger = logging.getLogger(__name__)
FIELDS = [
    "timestamp",
    "iso",
    "name",
    "live_rate",
    "predicted",
    "direction",
    "confidence",
    "confidence_pct",
]


def append_prediction(
    iso: str,
    name: str,
    live_rate: float,
    predicted: float,
    direction: str,
    confidence: str,
    confidence_pct: float,
):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "iso": iso,
                    "name": name,
                    "live_rate": live_rate,
                    "predicted": predicted,
                    "direction": direction,
                    "confidence": confidence,
                    "confidence_pct": round(confidence_pct, 4),
                }
            )
    except OSError as exc:
        logger.warning("Tidak dapat menulis log prediksi: %s", exc)


def load_log(limit: int = 200):
    if not LOG_PATH.exists():
        return []
    import pandas as pd

    df = pd.read_csv(LOG_PATH)
    return df.tail(limit)
