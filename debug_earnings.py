
import os
import sys
import logging
import pandas as pd
import yfinance as yf
from db import connect_db, get_pair_params
from utils.pairs import parse_pairs, unique_symbols_from_pairs

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def test_earnings_fetch():
    db_path = "data/live.db"
    con = connect_db(db_path)
    
    # Get enabled symbols
    df = get_pair_params(con)
    if df is None or df.empty:
        log.error("No pairs found in DB")
        return

    enabled = df[df["enabled"] == 1]
    raw_pairs = enabled["pair"].tolist()
    pairs_tuples = parse_pairs(raw_pairs)
    symbols = unique_symbols_from_pairs(pairs_tuples)
    
    log.info(f"Testing earnings fetch for {len(symbols)} symbols...")
    
    for i, ticker in enumerate(symbols):
        log.info(f"[{i+1}/{len(symbols)}] Fetching {ticker}...")
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            count = len(ed) if ed is not None else 0
            log.info(f"  > Success: {count} dates found")
        except Exception as e:
            log.error(f"  > Failed {ticker}: {e}")
            
    log.info("Test complete.")

if __name__ == "__main__":
    test_earnings_fetch()
