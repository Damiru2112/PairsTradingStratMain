import sqlite3
import pandas as pd
import sys
import os

# --------------------------------------------------------
# EDIT THIS LIST TO ADD NEW PAIRS
# Format: "SYM1-SYM2"
# --------------------------------------------------------
NEW_PAIRS = [
    # Example: "AAPL-MSFT",
    # Add your pairs here:
    "AAT-GNL",
    "AKR-CTO",
    "MAC-SKT",
    "ALEX-FPI",
    "AMH-ELS",
    "INVH-SUI",
    "CPT-IRT",
    "KW-PDM",
    "CUBE-NSA",
    "CNI-CP",
    "BOC-CBU",
    "FCF-PFS",
    "QGEN-SNN",
    "LYG-NOK",
    "BCH-BSAC",
    "ABG-AN",
    "ARI-BXMT",
    "CHMI-EFC",
    "ENS-THR",
    "BV-EPAC",
    "ACM-BAH",
    "HXL-TKR",
    "AGCO-ALSN",
    "TTC-VNT",
    "APLE-CLDT",
    "FFC-FPF",
    "BST-NIE",
    "DSL-HYT",
    "FRA-JFR",
    "CTRE-LTC",
    "SII-TFPM",
    "HMN-THG",
    "MTG-RDN",
    "FAF-FNF",
    "EPD-ET",
    "KYN-NML",
    "BBDC-BCSF",
    "PCN-PDI",
    "PDO-PFN",
    "BBN-BOND",
    "NAD-NEA"
]

# Default Parameters for new pairs
DEFAULT_PARAMS = {
    "z_entry": 3.5,
    "z_exit": 1.0,
    "max_drift_pct": 6.5,
    "max_drift_delta": 0.0, # Default $0 allow drift? Or set high? 0 works if we check >0.
    "alloc_pct": 0.1,
    "enabled": 1
}

def add_pairs():
    db_path = "data/live.db"
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print("Connecting to database...")
    con = sqlite3.connect(db_path)
    
    try:
        # Check existing
        existing = pd.read_sql("SELECT pair FROM pair_params", con)["pair"].tolist()
        existing_set = set(existing)
        
        to_add = []
        for p in NEW_PAIRS:
            if p not in existing_set:
                to_add.append(p)
            else:
                print(f"Skipping {p} (already exists)")
        
        if not to_add:
            print("No new pairs to add.")
            return

        print(f"Adding {len(to_add)} new pairs: {to_add}")
        
        # Prepare data
        data = []
        for p in to_add:
            data.append((
                p, 
                DEFAULT_PARAMS["z_entry"],
                DEFAULT_PARAMS["z_exit"],
                DEFAULT_PARAMS["max_drift_pct"],
                DEFAULT_PARAMS["max_drift_delta"],
                DEFAULT_PARAMS["alloc_pct"],
                DEFAULT_PARAMS["enabled"]
            ))
            
        con.executemany("""
            INSERT INTO pair_params (pair, z_entry, z_exit, max_drift_pct, max_drift_delta, alloc_pct, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        con.commit()
        print("Success! Pairs added to database.")
        print("IMPORTANT: Restart the engine (run_engine.py or run_engine_1m.py) to pick up changes.")
        
    except Exception as e:
        print(f"Error adding pairs: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    add_pairs()
