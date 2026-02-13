
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import connect_db
from data_polygon import get_aggs

DB_PATH = "data/live.db"

def migrate_schema():
    con = connect_db(DB_PATH)
    try:
        # Check if engine_id exists
        cursor = con.execute("PRAGMA table_info(open_positions)")
        cols = [row[1] for row in cursor.fetchall()]
        
        if "engine_id" not in cols:
            print("Migrating schema: Adding engine_id to open_positions...")
            con.execute("ALTER TABLE open_positions ADD COLUMN engine_id TEXT")
            con.commit()
            print("Migration complete.")
        else:
            print("Schema already has engine_id.")
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        con.close()

def restore_trade():
    print("Restoring WEC-AEE trade...")
    
    # Params
    pair = "WEC-AEE"
    sym1, sym2 = "WEC", "AEE"
    entry_time_et = "2026-02-03 16:00:00"
    entry_time_ts = pd.Timestamp(entry_time_et).tz_localize("US/Eastern")
    entry_time_iso = entry_time_ts.isoformat()
    
    # Fetch Prices from Polygon
    # We need Close price at 16:00 ET (21:00 UTC)
    # Using get_aggs
    print(f"Fetching prices for {entry_time_iso}...")
    
    # Aggs expects YYYY-MM-DD
    day_str_start = "2026-02-02"
    day_str_end = "2026-02-04"
    
    df1 = get_aggs(sym1, 15, "minute", day_str_start, day_str_end)
    df2 = get_aggs(sym2, 15, "minute", day_str_start, day_str_end)
    
    # Filter to exact time
    # Index is Eastern Time
    try:
        p1 = float(df1.loc[entry_time_ts]["close"])
        p2 = float(df2.loc[entry_time_ts]["close"])
        print(f"Found prices: {sym1}={p1}, {sym2}={p2}")
    except KeyError:
        print("Could not find exact bar. Dumping nearby:")
        print(df1.tail())
        return

    # Fetch Metrics from DB (Z, Beta)
    con = connect_db(DB_PATH)
    row = con.execute("""
        SELECT z, direct_beta, beta_30_weekly 
        FROM pair_series 
        WHERE pair=? AND time LIKE '2026-02-03%21:00:%'
    """, (pair,)).fetchone()
    
    if not row:
        print("Metric not found in DB!")
        z_entry = 2.5 # Approximate from user screenshot/logs if needed
        beta = 1.0 # fallback
        print(f"Using fallback Z={z_entry}, Beta={beta}")
    else:
        z_entry = row[0]
        # direct_beta is usually what we trade on? No, log said beta_drift_pct.
        # Strategy uses beta_30 for sizing.
        beta = row[2] 
        print(f"Found Metrics: Z={z_entry}, Beta={beta}")

    # Calculate Qty
    # Alloc Pct?
    # Fetch params
    p_row = con.execute("SELECT alloc_pct FROM pair_params WHERE pair=?", (pair,)).fetchone()
    alloc_pct = p_row[0] if p_row else 0.15      
    
    # Equity = 100k
    equity = 100000.0
    capital = equity * alloc_pct
    
    # Sizing logic from compute_pair_quantities
    # qty2 = capital / (price2 * (1 + 1/beta)) ... roughly
    # Let's import the actual function to be precise
    from strategy.sizing import compute_pair_quantities
    qty1, qty2 = compute_pair_quantities(p1, p2, beta, capital_per_trade=capital)
    
    print(f"Calculated Size: {qty1} x {qty2}")
    
    # Insert
    # We set engine_id = 'POLYGON_MAIN'
    print("Inserting into DB...")
    
    con.execute("""
        INSERT OR REPLACE INTO open_positions 
        (pair, sym1, sym2, direction, qty1, qty2, beta_entry, entry_time, entry_price1, entry_price2, entry_z, pnl_unrealized, updated_at, engine_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pair, sym1, sym2, "SHORT_SPREAD", 
        int(qty1), int(qty2), float(beta), 
        entry_time_iso, p1, p2, float(z_entry), 
        0.0, # PnL will update next tick
        datetime.now(timezone.utc).isoformat(),
        "POLYGON_MAIN"
    ))
    con.commit()
    con.close()
    print("Restoration Done.")

if __name__ == "__main__":
    migrate_schema()
    restore_trade()
