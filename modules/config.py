from pathlib import Path

from modules.currency_registry import CURRENCY_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "dataset.csv")
MODELS_DIR = str(PROJECT_ROOT / "models")
WINDOW = 30
TRAIN_RATIO = 0.8
TRAIN_EPOCHS = 25
BATCH_SIZE = 32

# Semua mata uang di registry (kolom dataset)
CURRENCY_MAP = {iso: col for col, (iso, name) in CURRENCY_COLUMNS.items()}
DISPLAY_NAMES = {iso: name for col, (iso, name) in CURRENCY_COLUMNS.items()}
API_PAIRS = {iso: ("USD", iso) for iso in CURRENCY_MAP}
