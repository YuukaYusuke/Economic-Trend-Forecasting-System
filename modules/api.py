import requests
import streamlit as st

from modules.settings import get_alphavantage_api_key

FRANKFURTER_LATEST = "https://api.frankfurter.app/latest?from=USD"
FRANKFURTER_CURRENCIES = "https://api.frankfurter.app/currencies"


def _live_rate_frankfurter(from_curr: str, to_curr: str) -> float | None:
    """Fallback gratis: USD → mata uang lain dari Frankfurter (sama dengan bulk)."""
    if from_curr != "USD":
        return None
    bulk = get_bulk_live_rates_usd()
    rate = bulk.get(to_curr)
    return float(rate) if rate is not None else None


def _live_rate_alphavantage(from_curr: str, to_curr: str) -> float | None:
    api_key = get_alphavantage_api_key()
    if not api_key:
        return None

    url = (
        "https://www.alphavantage.co/query"
        f"?function=CURRENCY_EXCHANGE_RATE"
        f"&from_currency={from_curr}"
        f"&to_currency={to_curr}"
        f"&apikey={api_key}"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "Note" in data or "Information" in data:
            return None
        rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return float(rate)
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


@st.cache_data(ttl=60)
def get_live_rate(from_curr: str, to_curr: str):
    """
    Kurs live satu pasangan (USD → ISO).
    Urutan: Alpha Vantage (jika ada API key) → Frankfurter (gratis, tanpa key).
    """
    rate = _live_rate_alphavantage(from_curr, to_curr)
    if rate is not None:
        return rate
    return _live_rate_frankfurter(from_curr, to_curr)


@st.cache_data(ttl=300)
def get_currency_names():
    """Nama resmi mata uang dari API Frankfurter."""
    try:
        res = requests.get(FRANKFURTER_CURRENCIES, timeout=15)
        res.raise_for_status()
        return res.json()
    except (requests.RequestException, ValueError):
        return {}


@st.cache_data(ttl=300)
def get_bulk_live_rates_usd():
    """Semua kurs live: 1 USD = X unit mata uang."""
    try:
        res = requests.get(FRANKFURTER_LATEST, timeout=15)
        res.raise_for_status()
        return {k: float(v) for k, v in res.json().get("rates", {}).items()}
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return {}
