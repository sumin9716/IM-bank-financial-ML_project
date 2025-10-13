import numpy as np
import pandas as pd

def historical_var(pnl_series: pd.Series, alpha=0.99, window=252):
    s = pnl_series.dropna()
    s = s.iloc[-window:] if len(s) > window else s
    if len(s)==0: return float("nan")
    return float(np.quantile(s, 1 - alpha))

def expected_shortfall(pnl_series: pd.Series, alpha=0.99, window=252):
    s = pnl_series.dropna()
    s = s.iloc[-window:] if len(s) > window else s
    if len(s)==0: return float("nan")
    cutoff = np.quantile(s, 1 - alpha)
    tail = s[s <= cutoff]
    return float(tail.mean()) if len(tail)>0 else float("nan")
