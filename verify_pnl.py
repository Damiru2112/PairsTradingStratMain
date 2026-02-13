
import sqlite3
import pandas as pd

def calculate_pnl(row):
    p1 = row['last_price1']
    p2 = row['last_price2']
    ep1 = row['entry_price1']
    ep2 = row['entry_price2']
    q1 = row['qty1']
    q2 = row['qty2']
    direction = row['direction']
    
    if direction == "SHORT_SPREAD":
        # Short 1, Long 2
        leg1 = (ep1 - p1) * q1
        leg2 = (p2 - ep2) * q2
    else:
        # Long 1, Short 2
        leg1 = (p1 - ep1) * q1
        leg2 = (ep2 - p2) * q2
        
    pnl = leg1 + leg2
    return pnl, leg1, leg2

def verify():
    con = sqlite3.connect('data/live.db')
    df = pd.read_sql('SELECT * FROM open_positions', con)
    con.close()
    
    print(f"{'Pair':<10} | {'Dir':<12} | {'Calc PnL':<10} | {'DB PnL':<10} | {'Ent1':<10} | {'Last1':<10} | {'Q1':<5} | {'Ent2':<10} | {'Last2':<10} | {'Q2':<5}")
    print("-" * 130)
    
    for _, row in df.iterrows():
        calc, l1, l2 = calculate_pnl(row)
        db_pnl = row['pnl_unrealized']
        diff = calc - db_pnl
        print(f"{row['pair']:<10} | {row['direction']:<12} | {calc:10.2f} | {db_pnl:10.2f} | {row['entry_price1']:10.2f} | {row['last_price1']:10.2f} | {row['qty1']:<5} | {row['entry_price2']:10.2f} | {row['last_price2']:10.2f} | {row['qty2']:<5}")

if __name__ == "__main__":
    verify()
