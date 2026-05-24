from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.chart_theme import CHART, apply_layout
from modules.currencies import get_selectable_currencies
from modules.pipeline_service import get_predictions


CURRENCY_COUNTRIES: dict[str, list[tuple[str, str]]] = {
    "AED": [("ARE", "United Arab Emirates")],
    "AUD": [("AUS", "Australia")],
    "BHD": [("BHR", "Bahrain")],
    "BND": [("BRN", "Brunei")],
    "BRL": [("BRA", "Brazil")],
    "BWP": [("BWA", "Botswana")],
    "CAD": [("CAN", "Canada")],
    "CHF": [("CHE", "Switzerland")],
    "CLP": [("CHL", "Chile")],
    "CNY": [("CHN", "China")],
    "COP": [("COL", "Colombia")],
    "CZK": [("CZE", "Czechia")],
    "DKK": [("DNK", "Denmark")],
    "DZD": [("DZA", "Algeria")],
    "EUR": [
        ("AUT", "Austria"),
        ("BEL", "Belgium"),
        ("DEU", "Germany"),
        ("ESP", "Spain"),
        ("FIN", "Finland"),
        ("FRA", "France"),
        ("GRC", "Greece"),
        ("IRL", "Ireland"),
        ("ITA", "Italy"),
        ("NLD", "Netherlands"),
        ("PRT", "Portugal"),
    ],
    "GBP": [("GBR", "United Kingdom")],
    "HUF": [("HUN", "Hungary")],
    "IDR": [("IDN", "Indonesia")],
    "ILS": [("ISR", "Israel")],
    "INR": [("IND", "India")],
    "IRR": [("IRN", "Iran")],
    "ISK": [("ISL", "Iceland")],
    "JPY": [("JPN", "Japan")],
    "KRW": [("KOR", "South Korea")],
    "KWD": [("KWT", "Kuwait")],
    "KZT": [("KAZ", "Kazakhstan")],
    "LKR": [("LKA", "Sri Lanka")],
    "LYD": [("LBY", "Libya")],
    "MUR": [("MUS", "Mauritius")],
    "MXN": [("MEX", "Mexico")],
    "MYR": [("MYS", "Malaysia")],
    "NOK": [("NOR", "Norway")],
    "NPR": [("NPL", "Nepal")],
    "NZD": [("NZL", "New Zealand")],
    "OMR": [("OMN", "Oman")],
    "PEN": [("PER", "Peru")],
    "PHP": [("PHL", "Philippines")],
    "PKR": [("PAK", "Pakistan")],
    "PLN": [("POL", "Poland")],
    "QAR": [("QAT", "Qatar")],
    "RUB": [("RUS", "Russia")],
    "SAR": [("SAU", "Saudi Arabia")],
    "SEK": [("SWE", "Sweden")],
    "SGD": [("SGP", "Singapore")],
    "THB": [("THA", "Thailand")],
    "TND": [("TUN", "Tunisia")],
    "TTD": [("TTO", "Trinidad and Tobago")],
    "UYU": [("URY", "Uruguay")],
    "VES": [("VEN", "Venezuela")],
    "ZAR": [("ZAF", "South Africa")],
}


def _rsi(values: pd.Series, period: int = 14) -> float:
    delta = values.diff().dropna()
    if len(delta) < period:
        return float("nan")
    gains = delta.clip(lower=0).tail(period).mean()
    losses = (-delta.clip(upper=0)).tail(period).mean()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _volatility(values: pd.Series, window: int = 30) -> float:
    returns = values.pct_change().dropna().tail(window)
    if returns.empty:
        return float("nan")
    return float(returns.std() * 100)


def _confidence_level(abs_return_pct: float, volatility_pct: float) -> str:
    if math.isnan(volatility_pct):
        volatility_pct = 0.0
    score = abs_return_pct / max(volatility_pct, 0.05)
    if score >= 1.6 or abs_return_pct >= 0.22:
        return "High"
    if score >= 0.65 or abs_return_pct >= 0.08:
        return "Medium"
    return "Low"


def _volatility_label(volatility_pct: float) -> str:
    if math.isnan(volatility_pct):
        return "Unknown"
    if volatility_pct >= 1.0:
        return "High"
    if volatility_pct >= 0.35:
        return "Medium"
    return "Low"


def _insight(return_pct: float, volatility_pct: float) -> str:
    direction = "kenaikan" if return_pct > 0 else "penurunan" if return_pct < 0 else "pergerakan datar"
    strength = "moderat" if abs(return_pct) >= 0.12 else "ringan"
    vol = _volatility_label(volatility_pct).lower()
    if direction == "pergerakan datar":
        return f"Prediksi cenderung stabil dengan volatilitas {vol}."
    return f"Prediksi menunjukkan {direction} {strength} dengan volatilitas {vol}."


@st.cache_data(ttl=300, show_spinner=False)
def build_world_map_forecasts(live_rates: dict[str, float], timeframe: str = "1D") -> pd.DataFrame:
    rows = []
    currencies = get_selectable_currencies()

    for item in currencies:
        iso = item["iso"]
        live = live_rates.get(iso)
        if live is None or iso not in CURRENCY_COUNTRIES:
            continue

        try:
            result = get_predictions(item["column"], live_rate=live, timeframe=timeframe)
        except Exception:
            continue

        df = result["df"]
        hist = df["rate"].tail(90)
        final_pred = result["next_value"]
        return_pct = result["calibration"]["return_pct"]
        raw_pred = result["raw_next_value"]
        rsi = _rsi(df["rate"])
        vol = _volatility(df["rate"])
        trend = "Bullish" if return_pct > 0 else "Bearish" if return_pct < 0 else "Neutral"
        confidence = _confidence_level(abs(return_pct), vol)

        for country_code, country in CURRENCY_COUNTRIES[iso]:
            rows.append(
                {
                    "row_id": f"{country_code}:{iso}",
                    "country_code": country_code,
                    "country": country,
                    "currency": iso,
                    "currency_name": item["name"],
                    "column": item["column"],
                    "last_price": float(live),
                    "raw_pred": float(raw_pred),
                    "final_pred": float(final_pred),
                    "predicted_return": float(return_pct),
                    "confidence": confidence,
                    "trend": trend,
                    "rsi": rsi,
                    "volatility": vol,
                    "volatility_label": _volatility_label(vol),
                    "insight": _insight(return_pct, vol),
                    "history": hist.tolist(),
                    "history_dates": df["date"].tail(90).dt.strftime("%Y-%m-%d").tolist(),
                    "timeframe": timeframe,
                }
            )

    return pd.DataFrame(rows)


def world_choropleth(df: pd.DataFrame, selected_country: str | None = None) -> go.Figure:
    custom = np.stack(
        [
            df["row_id"],
            df["country"],
            df["currency"],
            df["currency_name"],
            df["last_price"].map(lambda v: f"{v:,.4f}"),
            df["final_pred"].map(lambda v: f"{v:,.4f}"),
            df["predicted_return"].map(lambda v: f"{v:+.3f}%"),
            df["confidence"],
        ],
        axis=-1,
    )
    selectedpoints = None
    if selected_country:
        matches = df.index[df["country_code"] == selected_country].tolist()
        selectedpoints = matches or None

    fig = go.Figure(
        go.Choropleth(
            locations=df["country_code"],
            z=df["predicted_return"],
            text=df["country"],
            customdata=custom,
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.48, "#9ca3af"],
                [0.5, "#cbd5e1"],
                [0.52, "#9ca3af"],
                [1.0, "#065f46"],
            ],
            zmid=0,
            zmin=-0.35,
            zmax=0.35,
            marker_line_color="rgba(226,232,240,0.72)",
            marker_line_width=0.45,
            colorbar=dict(
                title=dict(text="% change", side="right", font=dict(color="#cbd5e1")),
                ticksuffix="%",
                tickfont=dict(color="#cbd5e1"),
            ),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Mata uang: %{customdata[2]} - %{customdata[3]}<br>"
                "Harga saat ini: %{customdata[4]}<br>"
                "Prediksi harga: %{customdata[5]}<br>"
                "Perubahan: %{customdata[6]}<br>"
                "Confidence: %{customdata[7]}<extra></extra>"
            ),
            selectedpoints=selectedpoints,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.72)),
        )
    )
    if selected_country:
        fig.add_trace(
            go.Choropleth(
                locations=[selected_country],
                z=[0],
                colorscale=[[0, "rgba(248,250,252,0)"], [1, "rgba(248,250,252,0)"]],
                marker_line_color="#f8fafc",
                marker_line_width=2.4,
                showscale=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#e5e7eb"),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.35)",
            projection_type="natural earth",
            landcolor="#111827",
            oceancolor="#020617",
            showocean=True,
            lakecolor="#020617",
        ),
        transition=dict(duration=240, easing="cubic-in-out"),
    )
    return fig


def projection_chart(row: pd.Series) -> go.Figure:
    dates = pd.to_datetime(row["history_dates"])
    values = pd.Series(row["history"], dtype=float)
    horizon_days = {"1D": 1, "7D": 7, "30D": 30}.get(row["timeframe"], 1)
    last_date = dates.iloc[-1] if hasattr(dates, "iloc") else dates[-1]
    future_date = last_date + timedelta(days=horizon_days)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Actual",
            line=dict(color=CHART["primary"], width=2.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[row["last_price"], row["final_pred"]],
            mode="lines+markers",
            name="Predicted",
            line=dict(color=CHART["warn"], width=2.5, dash="dot"),
            marker=dict(size=8),
        )
    )
    return apply_layout(
        fig,
        f"Actual vs Predicted - {row['country']} ({row['currency']})",
        y_title=f"{row['currency']} per 1 USD",
        height=320,
        bottom_margin=72,
    )


def compare_chart(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in rows.iterrows():
        dates = pd.to_datetime(row["history_dates"])
        values = pd.Series(row["history"], dtype=float)
        if values.empty or values.iloc[0] == 0:
            continue
        indexed = values / values.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=indexed,
                mode="lines",
                name=f"{row['country']} ({row['currency']})",
            )
        )
    return apply_layout(fig, "Compare countries - index base 100", y_title="Index", height=330)
