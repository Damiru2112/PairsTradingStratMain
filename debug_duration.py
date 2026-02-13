
import pandas as pd
from datetime import datetime, timezone
import pytz

# Simulate the data
data = {
    "entry_time": ["2026-02-12T20:00:00Z", "2026-02-11T10:00:00"] # One UTC string, one naive string
}
positions = pd.DataFrame(data)

print("Original Data:")
print(positions)

try:
    # Reproduce logic from app.py
    positions["entry_time"] = pd.to_datetime(positions["entry_time"], format='mixed', utc=True)
    positions["entry_time"] = positions["entry_time"].dt.tz_convert('US/Eastern')
    
    print("\nConverted to US/Eastern:")
    print(positions["entry_time"])
    
    now_utc = datetime.now(timezone.utc)
    print(f"\nNow UTC: {now_utc}")
    print(f"Now UTC tzinfo: {now_utc.tzinfo}")
    
    entry_val = positions["entry_time"].iloc[0]
    print(f"\nEntry Value (Index 0): {entry_val}")
    print(f"Entry Value tzinfo: {entry_val.tzinfo}")
    
    # Try subtraction
    diff = now_utc - entry_val
    print(f"\nSubtraction Result: {diff}")
    
except Exception as e:
    print(f"\nERROR: {e}") 

# Proposed Fix: Use UTC for calculation
print("\n--- Testing Fix ---")
positions = pd.DataFrame(data)
positions["entry_time_utc"] = pd.to_datetime(positions["entry_time"], format='mixed', utc=True)

now_utc = datetime.now(timezone.utc)
diff_fix = now_utc - positions["entry_time_utc"].iloc[0]
print(f"Fix Subtraction Result: {diff_fix}")
