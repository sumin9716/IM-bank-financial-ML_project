import numpy as np
import pandas as pd

def cip_forward_theoretical(spot: pd.Series, r_us: pd.Series, r_kr: pd.Series, days=30, day_count=365.0) -> pd.Series:
    s, us = spot.align(r_us, join="inner")
    s, kr = s.align(r_kr, join="inner")
    return s * (1 + us * (days/day_count)) / (1 + kr * (days/day_count))

def apply_spread(fwd_theo: pd.Series, spread_bps: float = 0.0, side: str = "sell_usd") -> pd.Series:
    pct = spread_bps / 10000.0
    if side == "sell_usd":
        return fwd_theo * (1 + pct)
    elif side == "buy_usd":
        return fwd_theo * (1 - pct)
    else:
        raise ValueError("side must be 'sell_usd' or 'buy_usd'")

def ndf_settlement_pnl(fixing_spot: pd.Series, agreed_forward: pd.Series, notional_usd: float) -> pd.Series:
    fx, fwd = fixing_spot.align(agreed_forward, join="inner")
    return (fx - fwd) * float(notional_usd)
