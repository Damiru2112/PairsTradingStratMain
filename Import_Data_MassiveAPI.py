import os
import requests
from datetime import date

API_KEY = os.getenv("Uo5696bmc67zfyX23fbjBqfEk7nPYkWu")
BASE_URL = "https://api.polygon.io"

def get_aggs(ticker: str, multiplier: int, timespan: str, start: str, end: str, adjusted=True, limit=50000):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": limit,
        "apiKey": API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

data = get_aggs("AAPL", 15, "minute", "2025-12-01", "2026-02-01")
bars = data.get("results", [])
print("bars:", len(bars), "first:", bars[0] if bars else None)
