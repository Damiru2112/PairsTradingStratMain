
import sqlite3
import pandas as pd
from datetime import datetime, timezone

con = sqlite3.connect('/home/trader/PairsTrading/data/live.db')

# Values for KW-PDM
pair = "KW-PDM"
sym1 = "KW"
sym2 = "PDM"
direction = "SHORT_SPREAD"
qty1 = 311
qty2 = 461
entry_price1 = 9.75
entry_price2 = 7.705
beta_entry = 461.0 / 311.0
entry_time = "2026-02-13T09:38:00-05:00"
updated_at = datetime.now(timezone.utc).isoformat()
engine_id = "US_1M"
beta_drift_limit = 10.0
entry_z = 0.0 
pnl_unrealized = 171.58 
last_price1 = 9.895 
last_price2 = 8.175 

sql = """
INSERT INTO open_positions 
(pair, sym1, sym2, direction, qty1, qty2, beta_entry, entry_time, entry_price1, entry_price2, entry_z, pnl_unrealized, last_price1, last_price2, updated_at, engine_id, beta_drift_limit)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

try:
    con.execute(sql, (pair, sym1, sym2, direction, qty1, qty2, beta_entry, entry_time, entry_price1, entry_price2, entry_z, pnl_unrealized, last_price1, last_price2, updated_at, engine_id, beta_drift_limit))
    con.commit()
    print("Inserted KW-PDM successfully.")
except Exception as e:
    print(f"Error inserting: {e}")

# Verify
con.row_factory = sqlite3.Row
cur = con.execute("SELECT * FROM open_positions WHERE pair = ?", (pair,))
row = cur.fetchone()
if row:
    print(" Verified row:", dict(row))
else:
    print(" Failed verification: Row not found.")

con.close()
