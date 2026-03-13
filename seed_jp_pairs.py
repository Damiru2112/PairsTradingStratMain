"""
seed_jp_pairs.py

Populate jp_pair_params table with the 7 JP pairs and default parameters.

Usage:
    python seed_jp_pairs.py
"""
import os
import pandas as pd
from db import connect_db, init_db, init_jp_db, jp_upsert_pair_params

DB_PATH = os.getenv("DB_PATH", "data/live.db")

JP_PAIRS = [
    "8267.T-9020.T",
    "9301.T-9735.T",
    "8309.T-8316.T",
    "7751.T-7752.T",
    "1928.T-4452.T",
    "2002.T-2269.T",
    "6103.T-6113.T",
]

# Default parameters
DEFAULTS = {
    "z_entry": 3.5,
    "z_exit": 1.0,
    "max_drift_pct": 0.50,
    "max_drift_delta": 0.30,
    "alloc_pct": 0.05,
    "enabled": 1,
}


def main():
    con = connect_db(DB_PATH)
    init_db(con)
    init_jp_db(con)

    rows = [{**{"pair": pair}, **DEFAULTS} for pair in JP_PAIRS]
    df = pd.DataFrame(rows)
    jp_upsert_pair_params(con, df)

    for pair in JP_PAIRS:
        print(f"  Seeded: {pair}")

    print(f"\nDone. {len(JP_PAIRS)} JP pairs seeded into jp_pair_params.")
    con.close()


if __name__ == "__main__":
    main()
