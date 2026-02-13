
import sqlite3
import os

DB_PATH = "data/live.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    con = sqlite3.connect(DB_PATH)
    try:
        # Check if column exists
        cursor = con.execute("PRAGMA table_info(open_positions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "beta_drift_limit" not in columns:
            print("Adding beta_drift_limit column to open_positions...")
            # Default to 10.0 (10%)
            con.execute("ALTER TABLE open_positions ADD COLUMN beta_drift_limit REAL DEFAULT 10.0")
            con.commit()
            print("Migration successful.")
        else:
            print("Column beta_drift_limit already exists.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    migrate()
