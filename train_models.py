"""
Latih semua model LSTM sekaligus. Jalankan sekali sebelum memakai aplikasi:

    python train_models.py
    python train_models.py --force   # latih ulang semua
"""

import argparse

from modules.trainer import train_all


def main():
    parser = argparse.ArgumentParser(description="Latih semua model LSTM")
    parser.add_argument("--force", action="store_true", help="Latih ulang meski file sudah ada")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    args = parser.parse_args()

    kwargs = {"force": args.force}
    if args.epochs:
        kwargs["epochs"] = args.epochs
    train_all(**kwargs)


if __name__ == "__main__":
    main()
