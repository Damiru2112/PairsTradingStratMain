import sys
import json
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure basic logging to stderr so stdout corresponds only to JSON output
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

def fetch_earnings(ticker):
    try:
        import yfinance as yf
        import pandas as pd
        
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        
        if ed is None or ed.empty:
            print(json.dumps([]))
            return

        results = []
        for ts_idx in ed.index:
            # ts_idx is a timezone-aware Timestamp (US/Eastern) or naive
            d = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            results.append({"ticker": ticker, "report_date": d.isoformat()})
            
        print(json.dumps(results))
        
    except Exception as e:
        logging.error(f"Fetch failed for {ticker}: {e}")
        # Print empty list on failure so parent process can parse valid JSON
        print(json.dumps([]))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps([]))
        sys.exit(1)
        
    ticker = sys.argv[1]
    fetch_earnings(ticker)
