
import pandas as pd
import sqlite3
import os

DB_PATH = "data/live.db"

def inspect_db():
    if not os.path.exists(DB_PATH):
        print("DB not found")
        return

    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT pair, entry_time FROM open_positions", con)
        if df.empty:
            print("No open positions.")
        else:
            print(df)
            print("\nRow 0 Raw Entry Time:", repr(df.iloc[0]["entry_time"]))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    inspect_db()
