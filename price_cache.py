# price_cache.py
from __future__ import annotations
import pandas as pd

class PriceCache:
    def __init__(self):
        self.data: dict[str, pd.Series] = {}

    def seed(self, symbol: str, df_one_col: pd.DataFrame):
        # df_one_col: index datetime, one column = symbol
        s = df_one_col.iloc[:, 0].copy()
        s.name = symbol
        self.data[symbol] = s.sort_index()

    def update_close(self, symbol: str, t: pd.Timestamp, close: float):
        s = self.data.get(symbol)
        if s is None:
            self.data[symbol] = pd.Series([close], index=[t], name=symbol).sort_index()
            return

        # overwrite if same timestamp exists
        s.loc[t] = close
        self.data[symbol] = s.sort_index()

    def get_frame(self, symbols: list[str]) -> pd.DataFrame:
        # join closes for multiple symbols
        dfs = [self.data[s].to_frame() for s in symbols]
        return pd.concat(dfs, axis=1).sort_index()

    def get_last(self, symbol: str) -> tuple[pd.Timestamp | None, float | None]:
        if symbol not in self.data or self.data[symbol].empty:
            return None, None
        
        s = self.data[symbol]
        last_t = s.index[-1]
        last_val = s.iloc[-1]
        return last_t, last_val

    def get_latest_prices(self) -> dict[str, float]:
        """Return {symbol: latest_close} for all cached symbols."""
        out = {}
        for sym, s in self.data.items():
            if not s.empty:
                out[sym] = float(s.iloc[-1])
        return out
