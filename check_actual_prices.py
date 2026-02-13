
import requests
import os
import json
import pandas as pd


API_KEY = os.getenv("POLYGON_API_KEY", "Uo5696bmc67zfyX23fbjBqfEk7nPYkWu")

def check_price(ticker):
    # Get last 1 minute bar
    # Use v2/aggs directly
    import time
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (24 * 60 * 60 * 1000) # last 24h
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_ms}/{end_ms}?adjusted=true&sort=desc&limit=1&apiKey={API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        print(f"--- {ticker} ---")
        if "results" in data and data["results"]:
            res = data["results"][0]
            print(f"Time: {pd.to_datetime(res['t'], unit='ms')}")
            print(f"Close: {res['c']}")
            print(json.dumps(res, indent=2))
        else:
            print("No data found")
            print(data)
    except Exception as e:
        print(f"Error {ticker}: {e}")

check_price("ENS")
check_price("THR")
