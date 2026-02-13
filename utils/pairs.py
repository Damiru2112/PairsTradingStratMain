from __future__ import annotations
from typing import Dict, List
import pandas as pd

def parse_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for s in raw_pairs:
        s = s.replace("–", "-").replace("—", "-").replace(" ", "")
        a, b = s.split("-")
        pairs.append((a, b))
    return pairs

def unique_symbols_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    """Extract unique symbols from list of (s1, s2) tuples."""
    s = set()
    for s1, s2 in pairs:
        s.add(s1)
        s.add(s2)
    return list(s)

def build_pair_frames(symbol_frames: Dict[str, pd.DataFrame], 
                     pairs: List[tuple], 
                     how: str = "inner") -> Dict[tuple, pd.DataFrame]:
    """
    Combine individual symbol frames into pair frames.
    dict key: (sym1, sym2)
    dict value: DataFrame with cols [sym1, sym2]
    """
    results = {}
    for s1, s2 in pairs:
        if s1 in symbol_frames and s2 in symbol_frames:
            df1 = symbol_frames[s1]
            df2 = symbol_frames[s2]
            
            # Align
            # Ensure "close" col is renamed to symbol if not already
            if "close" in df1.columns:
                c1 = df1[["close"]].rename(columns={"close": s1})
            else:
                c1 = df1[[s1]] if s1 in df1.columns else df1.iloc[:, 0].to_frame(name=s1)
                
            if "close" in df2.columns:
                c2 = df2[["close"]].rename(columns={"close": s2})
            else:
                c2 = df2[[s2]] if s2 in df2.columns else df2.iloc[:, 0].to_frame(name=s2)
                
            # Merge
            joined = c1.join(c2, how=how).dropna()
            
            if not joined.empty:
                results[(s1, s2)] = joined
                
    return results
