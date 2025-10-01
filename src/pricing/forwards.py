
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd
from pandas.tseries.offsets import DateOffset, BusinessDay

DayCount = Literal["act/360","act/365","cont"]
BizConv = Literal["following","modified_following","preceding"]

@dataclass
class CIPConfig:
    method: Literal["cont","simple"] = "cont"
    dcc_dom: DayCount = "act/365"    # KRW side (domestic)
    dcc_for: DayCount = "act/360"    # USD side (foreign)
    biz_conv: BizConv = "following"  # business day roll
    # If holidays is provided, business adjustment will consider them.
    # Expected schema: DataFrame[date]
    holidays: Optional[pd.DataFrame] = None

def _parse_tenor(tenor: str) -> dict:
    tenor = tenor.strip().upper()
    if tenor.endswith("D"):
        return {"days": int(tenor[:-1])}
    if tenor.endswith("W"):
        return {"days": 7 * int(tenor[:-1])}
    if tenor.endswith("M"):
        return {"months": int(tenor[:-1])}
    if tenor.endswith("Y"):
        return {"years": int(tenor[:-1])}
    raise ValueError("Unsupported tenor format (use Nd/Nw/Nm/Ny), got: %s" % tenor)

def _is_business_day(ts: pd.Timestamp, holidays: Optional[pd.DataFrame]) -> bool:
    if ts.weekday() >= 5:  # 5=Sat,6=Sun
        return False
    if holidays is not None and not holidays.empty:
        hset = set(pd.to_datetime(holidays["date"]).dt.normalize())
        return ts.normalize() not in hset
    return True

def _biz_adjust(dt: pd.Timestamp, conv: BizConv, holidays: Optional[pd.DataFrame]) -> pd.Timestamp:
    if _is_business_day(dt, holidays):
        return dt
    if conv == "following":
        x = dt
        while not _is_business_day(x, holidays):
            x += BusinessDay(1)
        return x
    if conv == "preceding":
        x = dt
        while not _is_business_day(x, holidays):
            x -= BusinessDay(1)
        return x
    if conv == "modified_following":
        # move forward; if month changes, move preceding
        x = dt
        while not _is_business_day(x, holidays):
            x += BusinessDay(1)
        if x.month != dt.month:
            # go back instead
            x = dt
            while not _is_business_day(x, holidays):
                x -= BusinessDay(1)
        return x
    raise ValueError(f"Unsupported business convention: {conv}")

def add_tenor(start: pd.Timestamp, tenor: str, conv: BizConv="following",
              holidays: Optional[pd.DataFrame]=None) -> pd.Timestamp:
    parts = _parse_tenor(tenor)
    dt = pd.Timestamp(start) + DateOffset(**parts)
    return _biz_adjust(dt, conv, holidays)

def year_fraction(start: pd.Timestamp, end: pd.Timestamp, dcc: DayCount) -> float:
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    if dcc == "cont":
        # "cont" means T is computed but later used in exp(r*T) where r is cont. comp. rate.
        # We still need a base; default to ACT/365 for T.
        days = (end - start).days
        return days / 365.0
    if dcc == "act/360":
        return (end - start).days / 360.0
    if dcc == "act/365":
        return (end - start).days / 365.0
    raise ValueError(f"Unknown day count: {dcc}")

def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    raise KeyError(f"Missing column '{col}' in DataFrame with columns={list(df.columns)}")

def _prepare_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Expect at least ['date','value']; allow ['series'] for multi-series
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date")
    return out

def _get_rate_on(date: pd.Timestamp, rates: pd.DataFrame, series: Optional[str]) -> float:
    if series is not None and "series" in rates.columns:
        sub = rates[rates["series"] == series]
    else:
        sub = rates
    sub = sub[sub["date"] <= date]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["value"])  # last available up to 'date'

def price_forward_cip_row(spot: float, r_dom: float, r_for: float, T_dom: float, T_for: float,
                          method: Literal["cont","simple"]="cont") -> float:
    if any([pd.isna(spot), pd.isna(r_dom), pd.isna(r_for), pd.isna(T_dom), pd.isna(T_for)]):
        return float("nan")
    if method == "cont":
        return float(spot) * math.exp((float(r_dom)*T_dom) - (float(r_for)*T_for))
    elif method == "simple":
        return float(spot) * (1.0 + float(r_dom)*T_dom) / (1.0 + float(r_for)*T_for)
    else:
        raise ValueError("method must be 'cont' or 'simple'")

def price_forward_cip(spot_df: pd.DataFrame,
                      kr_rate_df: pd.DataFrame,
                      us_rate_df: pd.DataFrame,
                      tenor: str="3M",
                      cfg: Optional[CIPConfig]=None,
                      kr_rate_series: Optional[str]=None,
                      us_rate_series: Optional[str]=None,
                      value_date_col: str="date",
                      spot_col: str="value") -> pd.DataFrame:
    """
    Compute USD/KRW forward fair values under CIP (v0: no CCY basis).
    Parameters
    ----------
    spot_df : DataFrame[date, value]    # KRW per USD
    kr_rate_df : DataFrame[date, value, (series)]
    us_rate_df : DataFrame[date, value, (series)]
    tenor : e.g., "1M","3M","6M","1Y"
    cfg : CIPConfig (day counts, business convention, holidays)
    kr_rate_series/us_rate_series : pick a 'series' if rate DFs contain multiple series
    Returns
    -------
    DataFrame[date, tenor, spot, r_dom, r_for, T_dom, T_for, fair_fwd, method]
    """
    cfg = cfg or CIPConfig()
    s = spot_df.copy()
    s[value_date_col] = pd.to_datetime(s[value_date_col]).dt.tz_localize(None)
    s = s.sort_values(value_date_col).reset_index(drop=True)
    kr = _prepare_rates(kr_rate_df)
    us = _prepare_rates(us_rate_df)

    out_rows = []
    for _, row in s.iterrows():
        dt = row[value_date_col]
        spot = float(row[spot_col])
        mat = add_tenor(dt, tenor, cfg.biz_conv, cfg.holidays)
        T_dom = year_fraction(dt, mat, cfg.dcc_dom if cfg.method=="simple" else "act/365")
        T_for = year_fraction(dt, mat, cfg.dcc_for if cfg.method=="simple" else "act/365")
        r_dom = _get_rate_on(dt, kr, kr_rate_series)
        r_for = _get_rate_on(dt, us, us_rate_series)
        fair = price_forward_cip_row(spot, r_dom/100.0, r_for/100.0, T_dom, T_for, cfg.method)
        out_rows.append({
            "date": dt, "tenor": tenor, "spot": spot,
            "r_dom": r_dom, "r_for": r_for,
            "T_dom": T_dom, "T_for": T_for,
            "fair_fwd": fair, "method": cfg.method,
            "maturity": mat
        })
    out = pd.DataFrame(out_rows)
    return out
