
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pandas as pd
from scipy.stats import norm

# -----------------------
# Garman–Kohlhagen Pricer
# -----------------------

OptionType = Literal["call", "put"]

@dataclass
class GKInputs:
    S: float        # spot, KRW per USD
    K: float        # strike
    T: float        # year fraction (ACT/365 or similar)
    sigma: float    # annualized vol (decimal)
    r_dom: float    # domestic rate (KRW) as decimal
    r_for: float    # foreign rate (USD) as decimal
    opt_type: OptionType = "call"

@dataclass
class GKResult:
    price: float
    d1: float
    d2: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho_dom: float  # ∂Price/∂r_dom (KRW rate)
    rho_for: float  # ∂Price/∂r_for (USD rate)

def _validate_inputs(x: GKInputs):
    if x.S <= 0 or x.K <= 0:
        raise ValueError("S and K must be positive.")
    if x.T <= 0:
        raise ValueError("T must be positive.")
    if x.sigma <= 0:
        raise ValueError("sigma must be positive.")
    if x.opt_type not in ("call","put"):
        raise ValueError("opt_type must be 'call' or 'put'.")

def garman_kohlhagen(x: GKInputs) -> GKResult:
    """
    Price an FX option under Garman–Kohlhagen.
    Treat foreign rate as a 'dividend yield'. All rates are annual, decimal.
    """
    _validate_inputs(x)
    S, K, T, v = x.S, x.K, x.T, x.sigma
    rd, rf = x.r_dom, x.r_for

    # Forward under GK: F = S * exp((rd - rf) * T)
    F = S * math.exp((rd - rf) * T)
    if v * math.sqrt(T) == 0:
        raise ValueError("sigma * sqrt(T) must be > 0")
    d1 = (math.log(F / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)

    df_dom = math.exp(-rd * T)
    df_for = math.exp(-rf * T)

    if x.opt_type == "call":
        price = df_dom * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = df_dom * norm.cdf(d1) * (F / S)  # dPrice/dS
    else:
        price = df_dom * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -df_dom * norm.cdf(-d1) * (F / S)

    # Greeks
    gamma = df_dom * norm.pdf(d1) * (F / S) / (S * v * math.sqrt(T))
    vega  = df_dom * F * norm.pdf(d1) * math.sqrt(T)            # ∂Price/∂σ
    # Theta (per year): differentiate price w.r.t T (simplified closed-form)
    # We'll compute numerical theta for robustness:
    eps = 1e-5
    price_eps = garman_kohlhagen(GKInputs(S,K,T+eps,v,rd,rf,x.opt_type)).price
    theta = (price_eps - price) / eps

    # Rhos
    # ∂Price/∂rd and ∂Price/∂rf using analytic forms
    if x.opt_type == "call":
        rho_dom = -T * price + T * df_dom * F * norm.cdf(d1)    # chain rule via df_dom and F
        rho_for =  T * df_dom * (-F) * norm.cdf(d1)             # foreign rate enters with minus on forward
    else:
        rho_dom = -T * price - T * df_dom * F * norm.cdf(-d1)
        rho_for =  T * df_dom * F * norm.cdf(-d1)

    return GKResult(price, d1, d2, delta, gamma, vega, theta, rho_dom, rho_for)

# -----------------------
# Utilities
# -----------------------

def realized_vol_from_spot(spot_df: pd.DataFrame, window: int = 63, trading_days: int = 252,
                           col_date: str = "date", col_value: str = "value") -> pd.DataFrame:
    """
    Compute rolling realized volatility from spot series.
    Returns DataFrame[date, rv_{window}d]
    """
    s = spot_df.copy()
    s[col_date] = pd.to_datetime(s[col_date]).dt.tz_localize(None)
    s = s.sort_values(col_date).reset_index(drop=True)
    s["ret"] = (s[col_value].pct_change()).apply(lambda x: math.log(1+x) if pd.notna(x) else x)
    s[f"rv_{window}d"] = s["ret"].rolling(window).std() * math.sqrt(trading_days)
    return s[[col_date, f"rv_{window}d"]]

def implied_vol(target_price: float, x: GKInputs,
                lo: float = 1e-4, hi: float = 3.0, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Solve σ from price using bisection (robust).
    """
    x0 = GKInputs(x.S, x.K, x.T, lo, x.r_dom, x.r_for, x.opt_type)
    x1 = GKInputs(x.S, x.K, x.T, hi, x.r_dom, x.r_for, x.opt_type)
    p0 = garman_kohlhagen(x0).price
    p1 = garman_kohlhagen(x1).price
    if (p0 - target_price) * (p1 - target_price) > 0:
        # expand bounds
        for _ in range(10):
            hi *= 2.0
            x1 = GKInputs(x.S, x.K, x.T, hi, x.r_dom, x.r_for, x.opt_type)
            p1 = garman_kohlhagen(x1).price
            if (p0 - target_price) * (p1 - target_price) <= 0:
                break
        else:
            raise RuntimeError("Failed to bracket implied vol.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        xm = GKInputs(x.S, x.K, x.T, mid, x.r_dom, x.r_for, x.opt_type)
        pm = garman_kohlhagen(xm).price
        if abs(pm - target_price) < tol:
            return mid
        if (p0 - target_price) * (pm - target_price) <= 0:
            hi = mid
            p1 = pm
        else:
            lo = mid
            p0 = pm
    return 0.5 * (lo + hi)

def put_call_parity_gk(S: float, K: float, T: float, r_dom: float, r_for: float) -> float:
    """
    Return forward parity difference: C - P - df_dom*(F-K)
    Should be ~0 under GK (diagnostic).
    """
    df_dom = math.exp(-r_dom * T)
    F = S * math.exp((r_dom - r_for) * T)
    return df_dom * (F - K)
