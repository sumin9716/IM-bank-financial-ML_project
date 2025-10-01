
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Iterable
import pandas as pd
from pandas.tseries.offsets import BusinessDay

Direction = Literal["long_usd","short_usd"]

@dataclass
class NDFTrade:
    trade_date: pd.Timestamp
    maturity: pd.Timestamp              # fixing/settlement date (v0: 동일 처리)
    notional_usd: float
    fwd_rate: float                     # KRW per USD
    direction: Direction = "long_usd"   # long_usd: +1 * (fix - fwd); short_usd: -1 * (fix - fwd)
    fix_lag_days: int = 0               # e.g., if fixing happens T-2, set 2 (v0 default 0)

def _normalize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c] = pd.to_datetime(d[c]).dt.tz_localize(None)
    return d

def _closest_available_fix(spot_df: pd.DataFrame, target_date: pd.Timestamp,
                           method: Literal["preceding","following"]="preceding") -> tuple[pd.Timestamp,float]:
    """
    Pick the closest available fixing given daily spot data.
    If exact date is missing (weekend/holiday), choose preceding (default) or following business day.
    """
    s = spot_df[["date","value"]].dropna().copy()
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
    s = s.sort_values("date").reset_index(drop=True)

    # exact
    row = s[s["date"] == target_date]
    if not row.empty:
        return target_date, float(row.iloc[0]["value"])

    if method == "preceding":
        sub = s[s["date"] < target_date]
        if sub.empty:
            raise ValueError("No preceding fixing available before %s" % target_date.date())
        r = sub.iloc[-1]
        return pd.Timestamp(r["date"]), float(r["value"])
    else:
        sub = s[s["date"] > target_date]
        if sub.empty:
            raise ValueError("No following fixing available after %s" % target_date.date())
        r = sub.iloc[0]
        return pd.Timestamp(r["date"]), float(r["value"])

def simulate_ndf_pnl(trades: pd.DataFrame,
                     spot_df: pd.DataFrame,
                     fixing_pick: Literal["preceding","following"]="preceding") -> pd.DataFrame:
    """
    Compute KRW PnL for a list of NDF trades using fixing from spot_df (KRW per USD).
    Sign convention:
      - direction == 'long_usd'  → PnL = + notional * (fix - fwd)
      - direction == 'short_usd' → PnL = - notional * (fix - fwd)
    Inputs
    ------
    trades: DataFrame[trade_date, maturity, notional_usd, fwd_rate, direction?]
    spot_df: DataFrame[date, value]
    Returns
    -------
    DataFrame with columns:
      trade_date, maturity, notional_usd, fwd_rate, direction, fix_date, fix_rate, pnl_krw
    """
    req_cols = ["trade_date","maturity","notional_usd","fwd_rate"]
    for c in req_cols:
        if c not in trades.columns:
            raise KeyError(f"trades missing column: {c}")
    t = trades.copy()
    t = _normalize_dates(t, ["trade_date","maturity"])
    t["direction"] = t.get("direction","long_usd").fillna("long_usd")

    out = []
    for _, row in t.iterrows():
        td = pd.Timestamp(row["trade_date"])
        mat = pd.Timestamp(row["maturity"]) - pd.Timedelta(days=int(row.get("fix_lag_days", 0) or 0))
        notional = float(row["notional_usd"])
        fwd = float(row["fwd_rate"])
        dir_sign = 1.0 if str(row["direction"]) == "long_usd" else -1.0

        fix_date, fix_rate = _closest_available_fix(spot_df, mat, method=fixing_pick)
        pnl = dir_sign * notional * (fix_rate - fwd)
        out.append({
            "trade_date": td, "maturity": pd.Timestamp(row["maturity"]),
            "notional_usd": notional, "fwd_rate": fwd, "direction": row["direction"],
            "fix_date": fix_date, "fix_rate": fix_rate, "pnl_krw": pnl
        })
    return pd.DataFrame(out).sort_values(["maturity","trade_date"]).reset_index(drop=True)

def make_trades_from_fair_values(fair_df: pd.DataFrame,
                                 tenor: str="1M",
                                 notional_usd: float=1_000_000.0,
                                 direction: Direction="long_usd") -> pd.DataFrame:
    """
    Create simple trade tickets from CIP fair value outputs.
    Expects fair_df columns: date, tenor, fair_fwd, maturity
    Returns DataFrame[trade_date, maturity, notional_usd, fwd_rate, direction]
    """
    cols = {"date","tenor","fair_fwd","maturity"}
    missing = cols - set(fair_df.columns)
    if missing:
        raise KeyError(f"fair_df missing columns: {missing}")
    d = fair_df.copy()
    d = d[d["tenor"] == tenor].dropna(subset=["fair_fwd","maturity"])
    d = d.sort_values("date")
    trades = pd.DataFrame({
        "trade_date": pd.to_datetime(d["date"]).dt.tz_localize(None),
        "maturity": pd.to_datetime(d["maturity"]).dt.tz_localize(None),
        "notional_usd": notional_usd,
        "fwd_rate": d["fair_fwd"].astype(float),
        "direction": direction
    })
    return trades.reset_index(drop=True)
