
from __future__ import annotations
import math
from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd

RetType = Literal["log","simple"]

def _prep_spot(spot_df: pd.DataFrame, col_date: str="date", col_value: str="value") -> pd.DataFrame:
    s = spot_df.copy()
    s[col_date] = pd.to_datetime(s[col_date]).dt.tz_localize(None)
    s = s.sort_values(col_date).reset_index(drop=True)
    s = s.dropna(subset=[col_value])
    s[col_value] = pd.to_numeric(s[col_value], errors="coerce")
    s = s.dropna(subset=[col_value])
    return s

def _compute_returns(spot_df: pd.DataFrame, ret: RetType="log",
                     col_date: str="date", col_value: str="value") -> pd.DataFrame:
    s = _prep_spot(spot_df, col_date, col_value)
    if ret == "log":
        s["ret1"] = np.log(s[col_value] / s[col_value].shift(1))
    else:
        s["ret1"] = s[col_value].pct_change()
    return s

def _horizon_aggregate(s: pd.DataFrame, horizon_days: int, ret: RetType="log",
                       col_date: str="date", col_value: str="value") -> pd.DataFrame:
    """
    Returns DataFrame with horizon returns r_h and base spot S_{t-h}
    """
    s = s.copy()
    if horizon_days <= 1:
        s["r_h"] = s["ret1"]
        s["S_base"] = s[col_value].shift(1)
        s[col_date] = s[col_date]
        return s.dropna(subset=["r_h","S_base"])
    # rolling sum (log) or compounded (simple)
    if ret == "log":
        s["r_h"] = s["ret1"].rolling(horizon_days).sum()
        s["S_base"] = s[col_value].shift(horizon_days)
    else:
        s["r_h"] = (1.0 + s["ret1"]).rolling(horizon_days).apply(lambda x: np.prod(x) - 1.0, raw=True)
        s["S_base"] = s[col_value].shift(horizon_days)
    return s.dropna(subset=["r_h","S_base"])

def _pnl_from_returns(s: pd.DataFrame, ret: RetType="log",
                      notional_usd: float=1.0) -> pd.Series:
    """
    KRW PnL for long USD position of size 'notional_usd': ΔV = N * (S_t - S_{t-h}) = N * S_base * (exp(r_h)-1)
    """
    if ret == "log":
        pnl = notional_usd * s["S_base"] * (np.exp(s["r_h"]) - 1.0)
    else:
        pnl = notional_usd * s["S_base"] * s["r_h"]
    return pnl

def hist_var_es(spot_df: pd.DataFrame,
                horizon_days: int=1,
                alpha: float=0.99,
                ret: RetType="log",
                notional_usd: float=1.0,
                col_date: str="date",
                col_value: str="value") -> dict:
    """
    Historical VaR/ES (KRW) for a long USD position valued in KRW using USD/KRW spot series.
    Returns a dict with summary metrics and the underlying distribution (loss series).
    - VaR/ES are positive KRW amounts (loss units).
    """
    s = _compute_returns(spot_df, ret=ret, col_date=col_date, col_value=col_value)
    s = _horizon_aggregate(s, horizon_days=horizon_days, ret=ret, col_date=col_date, col_value=col_value)
    pnl = _pnl_from_returns(s, ret=ret, notional_usd=notional_usd)
    loss = -pnl  # define loss = -PnL (so loss>0 is a loss for long USD)
    loss = loss.dropna()
    if len(loss) < 10:
        raise ValueError("Not enough observations to compute VaR/ES.")

    # quantile (alpha) e.g. 0.99
    var = np.quantile(loss.values, alpha)
    es = loss[loss >= var].mean()

    # window stats on returns (diagnostic)
    stats = {
        "ret_mean": float(s["r_h"].mean()),
        "ret_std": float(s["r_h"].std(ddof=1)),
        "ret_skew": float(s["r_h"].skew()),
        "ret_kurt": float(s["r_h"].kurt()),  # Fisher by pandas
    }

    result = {
        "alpha": alpha,
        "horizon_days": horizon_days,
        "notional_usd": notional_usd,
        "VaR": float(var),
        "ES": float(es),
        "n_obs": int(loss.shape[0]),
        "loss_series": pd.DataFrame({
            "date": s["date"],
            "loss": loss.values
        }).dropna()
    }
    result.update(stats)
    return result

def worst_events(loss_df: pd.DataFrame, top_n: int=10, col_date: str="date", col_loss: str="loss") -> pd.DataFrame:
    """
    Return top N worst loss events (largest losses).
    """
    df = loss_df[[col_date, col_loss]].dropna().copy()
    df = df.sort_values(col_loss, ascending=False).head(top_n).reset_index(drop=True)
    return df

def rolling_var_series(spot_df: pd.DataFrame,
                       window: int=252,
                       horizon_days: int=1,
                       alpha: float=0.99,
                       ret: RetType="log",
                       notional_usd: float=1.0,
                       col_date: str="date",
                       col_value: str="value") -> pd.DataFrame:
    """
    Compute rolling Historical VaR over a moving window (in observations of horizon PnL).
    Returns: DataFrame[date, VaR, ES]
    """
    s = _compute_returns(spot_df, ret=ret, col_date=col_date, col_value=col_value)
    s = _horizon_aggregate(s, horizon_days=horizon_days, ret=ret, col_date=col_date, col_value=col_value).reset_index(drop=True)
    pnl = _pnl_from_returns(s, ret=ret, notional_usd=notional_usd).reset_index(drop=True)
    loss = -pnl

    out_dates = []
    vars_ = []
    es_ = []
    for i in range(window, len(loss)+1):
        sub = loss.iloc[i-window:i]
        var = float(np.quantile(sub.values, alpha))
        es = float(sub[sub >= var].mean())
        out_dates.append(s["date"].iloc[i-1])
        vars_.append(var); es_.append(es)
    return pd.DataFrame({"date": out_dates, "VaR": vars_, "ES": es_})
