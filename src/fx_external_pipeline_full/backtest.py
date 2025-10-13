import pandas as pd
from .holiday_calendar import is_business_day, previous_business_day

def _adjust_prev_bday(date, countries, cache_dir):
    d = pd.Timestamp(date).normalize()
    if is_business_day(d, countries=countries, cache_dir=cache_dir):
        return d
    return previous_business_day(d, countries=countries, cache_dir=cache_dir)

def monthly_forward_strategy(exposure_df: pd.DataFrame,
                             spot_eom: pd.Series,
                             forward_eom: pd.Series,
                             company_col: str = "company_id",
                             date_col: str = "month",
                             countries = ("KR",),
                             holiday_cache_dir: str = "data/reference/holidays_cache") -> pd.DataFrame:
    """
    월말 체결 → 익월 만기 전략 (영업일 보정 적용).
    - trade_date_bd: 체결월의 '직전 영업일'
    - fix_date_bd:   익월말의 '직전 영업일'
    가격평가(스팟/선도)는 월말(EOM) 시계열을 사용하고, 일정만 영업일로 보정합니다.
    """
    pnl_rows = []
    s = spot_eom.copy(); f = forward_eom.copy()
    if not isinstance(s.index, pd.DatetimeIndex): s.index = pd.to_datetime(s.index)
    if not isinstance(f.index, pd.DatetimeIndex): f.index = pd.to_datetime(f.index)

    for cid, grp in exposure_df.groupby(company_col):
        g = grp.sort_values(date_col).reset_index(drop=True)
        for i in range(len(g)-1):
            t_eom = pd.to_datetime(g.loc[i, date_col]).normalize()
            t1_eom = pd.to_datetime(g.loc[i+1, date_col]).normalize()

            # 영업일 보정 (직전 영업일)
            trade_bd = _adjust_prev_bday(t_eom, countries, holiday_cache_dir)
            fix_bd   = _adjust_prev_bday(t1_eom, countries, holiday_cache_dir)

            # 월말 기준 가격 평가 (EOM 인덱스 필요)
            if t_eom not in f.index or t1_eom not in s.index or t_eom not in s.index:
                continue

            spot_trade = float(s.loc[t_eom])
            spot_fix   = float(s.loc[t1_eom])
            fwd_trade  = float(f.loc[t_eom])

            hedge = float(g.loc[i, "hedge_ratio"])
            ne = float(g.loc[i, "net_exposure"])
            notional_usd = abs(ne) * hedge / max(spot_trade, 1e-8)

            pnl = (spot_fix - fwd_trade) * notional_usd

            pnl_rows.append({
                company_col: cid,
                "trade_month": t_eom,
                "fix_month": t1_eom,
                "trade_date_bd": trade_bd,
                "fix_date_bd": fix_bd,
                "notional_usd": notional_usd,
                "pnl_krw": pnl
            })

    return pd.DataFrame(pnl_rows)
