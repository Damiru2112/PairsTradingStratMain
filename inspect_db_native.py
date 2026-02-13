
import sqlite3
import os

DB_PATH = "data/live.db"

def inspect_db():
    if not os.path.exists(DB_PATH):
        print("DB not found")
        return

    con = sqlite3.connect(DB_PATH)
    try:
        cursor = con.execute("SELECT pair, entry_time FROM open_positions")
        rows = cursor.fetchall()
        if not rows:
            print("No open positions.")
        else:
            for row in rows:
                print(f"Pair: {row[0]}, Entry Time: {repr(row[1])}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    inspect_db()
