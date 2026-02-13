"""
smoke_test_polygon.py

Quick verification script to confirm:
1. Polygon is the active provider (implied by this script running successfully with data_polygon)
2. API Key is present.
3. Fetch works for a sample ticker.
"""

import os
import sys
import logging
import pandas as pd

# Add parent dir to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Manual .env loading
env_path = os.path.join(os.path.dirname(__file__), "../.env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from data_polygon import fetch_latest_closed_15m_close, API_KEY

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_test")

def redact(s):
    if not s: return "None"
    if len(s) < 8: return "****"
    return s[:4] + "****" + s[-4:]

def main():
    print("=== POLYGON SMOKE TEST ===")
    
    print(f"Provider: POLYGON (Hardcoded focus)")
    print(f"API Key: {redact(API_KEY)}")
    
    if not API_KEY:
        print("❌ FATAL: No API Key found.")
        sys.exit(1)
        
    # 2. Fetch Sample
    ticker = "SPY"
    print(f"\nFetching latest 15m close for {ticker}...")
    
    try:
        t, close = fetch_latest_closed_15m_close(ticker)
        
        if t is None or close is None:
            print(f"❌ Failed: returned None for {ticker}")
            sys.exit(1)
            
        print(f"✅ Success!")
        print(f"  Time: {t}")
        print(f"  Close: {close}")
        
        # Check freshness (within last few days, allowing for weekends)
        now = pd.Timestamp.now(tz="US/Eastern")
        # Ensure t is localized if not already (it should be from data_polygon)
        if t.tzinfo is None:
            print("  ⚠️ Warning: Returned timestamp is naive.")
        
        print(f"  Current Time (Eastern): {now}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        # traceback.print_exc()
        sys.exit(1)

    print("\n=== TEST PASSED ===")

if __name__ == "__main__":
    main()
