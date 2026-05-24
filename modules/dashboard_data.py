from pathlib import Path

import streamlit as st

from modules.api import get_bulk_live_rates_usd, get_live_rate
from modules.config import MODELS_DIR
from modules.currencies import get_selectable_currencies
from modules.pipeline_service import get_predictions
from modules.realtime_analysis import build_realtime_comparison
from modules.trainer import get_trainable_currencies


@st.cache_data(ttl=120, show_spinner=False)
def executive_summary(fav_iso: str = "IDR"):
    bulk = get_bulk_live_rates_usd()
    api_ok = len(bulk) > 0
    trained = len(list(Path(MODELS_DIR).glob("*.keras")))
    total_models = len(get_trainable_currencies())
    currencies = get_selectable_currencies()
    fav = next((c for c in currencies if c["iso"] == fav_iso), currencies[0] if currencies else None)

    pred_block = None
    vs_5y = None
    if fav:
        live = bulk.get(fav["iso"]) or get_live_rate("USD", fav["iso"])
        if live:
            try:
                r = get_predictions(fav["column"], live_rate=live)
                pred_block = {
                    "arah": r["prediksi_arah"],
                    "live": live,
                    "pred": r["next_value"],
                    "raw_pred": r.get("raw_next_value"),
                    "change_pct": (r.get("calibration") or {}).get("return_pct"),
                    "cap_pct": (r.get("calibration") or {}).get("cap_pct"),
                }
            except Exception:
                pass

    try:
        rt_df, _ = build_realtime_comparison()
        row = rt_df[rt_df["Kode"] == fav_iso]
        if not row.empty and row.iloc[0]["Perubahan (%)"] is not None:
            vs_5y = float(row.iloc[0]["Perubahan (%)"])
    except Exception:
        pass

    return {
        "api_ok": api_ok,
        "trained": trained,
        "total_models": total_models,
        "fav": fav,
        "pred": pred_block,
        "vs_5y_pct": vs_5y,
        "api_count": len(bulk),
    }
