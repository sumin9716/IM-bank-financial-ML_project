
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd

Risk = Literal["conservative","standard","aggressive"]

@dataclass
class RuleThresholds:
    # Volatility bands (annualized, decimals), default bands tuned for FX:
    high_vol_20d: float = 0.12
    med_vol_20d: float  = 0.07
    high_vol_60d: float = 0.10
    med_vol_60d: float  = 0.06
    # Carry adjustment (r_dom - r_for) in absolute decimal terms
    carry_pos: float = 0.01   # +1% or more → plus one step
    carry_neg: float = -0.01  # -1% or less → minus one step

def _rolling_vol(spot: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    ret = np.log(spot / spot.shift(1))
    return ret.rolling(window).std() * np.sqrt(trading_days)

def _moving_avg(spot: pd.Series, window: int) -> pd.Series:
    return spot.rolling(window).mean()

def compute_market_features(spot_df: pd.DataFrame,
                            kr_rate_df: Optional[pd.DataFrame] = None,
                            us_rate_df: Optional[pd.DataFrame] = None,
                            kr_series: Optional[str] = None,
                            us_series: Optional[str] = None,
                            col_date: str = "date",
                            col_value: str = "value") -> pd.DataFrame:
    """
    Returns DataFrame[date, rv_20d, rv_60d, ma20, ma60, trend_up, carry]
      - trend_up = 1 if MA20 > MA60 else 0
      - carry = (r_dom - r_for) in decimal (%/100). If rates missing, carry=NaN.
    """
    s = spot_df[[col_date, col_value]].copy()
    s[col_date] = pd.to_datetime(s[col_date]).dt.tz_localize(None)
    s = s.sort_values(col_date).reset_index(drop=True)
    s["rv_20d"] = _rolling_vol(s[col_value], 20)
    s["rv_60d"] = _rolling_vol(s[col_value], 60)
    s["ma20"] = _moving_avg(s[col_value], 20)
    s["ma60"] = _moving_avg(s[col_value], 60)
    s["trend_up"] = (s["ma20"] > s["ma60"]).astype(int)

    carry = pd.Series(index=s.index, dtype=float)
    if kr_rate_df is not None and us_rate_df is not None:
        kr = kr_rate_df.copy()
        us = us_rate_df.copy()
        kr["date"] = pd.to_datetime(kr["date"]).dt.tz_localize(None)
        us["date"] = pd.to_datetime(us["date"]).dt.tz_localize(None)
        if "series" in kr.columns and kr_series is not None:
            kr = kr[kr["series"] == kr_series]
        if "series" in us.columns and us_series is not None:
            us = us[us["series"] == us_series]
        kr = kr.sort_values("date").set_index("date")["value"].astype(float)
        us = us.sort_values("date").set_index("date")["value"].astype(float)
        # align by last available up to date
        kr_aligned = []
        us_aligned = []
        for dt in s[col_date]:
            k = kr.loc[:dt].iloc[-1] if not kr.loc[:dt].empty else np.nan
            u = us.loc[:dt].iloc[-1] if not us.loc[:dt].empty else np.nan
            kr_aligned.append(k); us_aligned.append(u)
        carry = (pd.Series(kr_aligned) - pd.Series(us_aligned)) / 100.0
    s["carry"] = carry.values
    return s[[col_date, "rv_20d", "rv_60d", "ma20", "ma60", "trend_up", "carry"]]

def _step_from_ratio(r: float) -> int:
    # Map ratio to discrete step 0,1,2 representing 0%,50%,80%
    if r <= 0.01: return 0
    if r < 0.65:  return 1
    return 2

def _ratio_from_step(step: int) -> float:
    return [0.0, 0.5, 0.8][max(0, min(2, step))]

def _apply_risk(step: int, risk: Risk) -> int:
    # conservative → +1 step, aggressive → -1 step
    if risk == "conservative": return min(2, step + 1)
    if risk == "aggressive":   return max(0, step - 1)
    return step

def recommend_hedge_ratio(features_df: pd.DataFrame,
                          risk: Risk = "standard",
                          thresholds: RuleThresholds = RuleThresholds()) -> pd.DataFrame:
    """
    Rule-based hedge ratio recommendations.
    Input features: DataFrame[date, rv_20d, rv_60d, trend_up, carry]
    Returns: DataFrame[date, base_step, adj_step, hedge_ratio, reason]
    Rules:
      1) Volatility base step:
         - if rv_20d >= high_20 or rv_60d >= high_60 → step=2 (80%)
         - elif rv_20d >= med_20 or rv_60d >= med_60 → step=1 (50%)
         - else step=0 (0%)
      2) Carry adjustment:
         - if carry >= +carry_pos → +1 step
         - if carry <= carry_neg  → -1 step
      3) Trend adjustment (optional, weak):
         - if trend_up==1 → +0 (neutral)  # exporters/importers sign unknown in v0
      4) Risk appetite:
         - conservative → +1 step
         - aggressive   → -1 step
    """
    f = features_df.copy()
    f = f.sort_values("date").reset_index(drop=True)
    reasons = []
    base_steps = []
    adj_steps = []

    for _, row in f.iterrows():
        rv20 = float(row.get("rv_20d", np.nan))
        rv60 = float(row.get("rv_60d", np.nan))
        carry = row.get("carry", np.nan)

        # 1) Volatility base
        if (not np.isnan(rv20) and rv20 >= thresholds.high_vol_20d) or (not np.isnan(rv60) and rv60 >= thresholds.high_vol_60d):
            step = 2; why = ["high vol → 80%"]
        elif (not np.isnan(rv20) and rv20 >= thresholds.med_vol_20d) or (not np.isnan(rv60) and rv60 >= thresholds.med_vol_60d):
            step = 1; why = ["medium vol → 50%"]
        else:
            step = 0; why = ["low vol → 0%"]

        base_steps.append(step)

        # 2) Carry adjustment
        if not np.isnan(carry):
            if carry >= thresholds.carry_pos:
                step = min(2, step + 1); why.append(f"carry {carry:.2%} ≥ +{thresholds.carry_pos:.0%} → +1")
            elif carry <= thresholds.carry_neg:
                step = max(0, step - 1); why.append(f"carry {carry:.2%} ≤ {thresholds.carry_neg:.0%} → -1")

        # 3) Trend adjustment (neutral in v0)
        # Optionally enable in v1 if exposure direction known.

        # 4) Risk appetite
        step_r = _apply_risk(step, risk)
        if step_r != step:
            why.append(f"risk:{risk} → {'+1' if step_r>step else '-1'}")
        step = step_r

        adj_steps.append(step)
        reasons.append("; ".join(why))

    out = pd.DataFrame({
        "date": f["date"],
        "base_step": base_steps,
        "adj_step": adj_steps,
        "hedge_ratio": [_ratio_from_step(s) for s in adj_steps],
        "reason": reasons
    })
    return out

