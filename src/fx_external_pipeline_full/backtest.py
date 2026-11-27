import pandas as pd

from .holiday_calendar import DEFAULT_CACHE_DIR, previous_business_day


def monthly_forward_strategy(exposure_df: pd.DataFrame,
                             spot_eom: pd.Series,
                             forward_eom: pd.Series,
                             company_col: str = "company_id",
                             date_col: str = "month",
                             countries = ("KR",),
                             holiday_cache_dir: str = DEFAULT_CACHE_DIR,
                             min_notional_threshold: float = 1e-6,
                             include_open_positions: bool = True) -> pd.DataFrame:
    """월말에 체결한 1개월 선도 계약을 익월말 환율로 정산한 거래별 손익.

    - pnl_krw = (정산월말 환율 - 만기 조정 선도환율) × |순노출| × 헤지 비율 / 체결월말 환율
    - trade_date_bd / fix_date_bd: 체결월말·정산월말이 휴일이면 직전 영업일
    - 가격은 월말(EOM) 환율·선도환율을 쓰고, 날짜만 영업일로 보정한다.
    - include_open_positions=True이면 기업별 마지막 노출월을 status='open' 행으로 덧붙인다.
    """
    pnl_rows = []
    s = spot_eom.copy()
    f = forward_eom.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if not isinstance(f.index, pd.DatetimeIndex):
        f.index = pd.to_datetime(f.index)

    for cid, grp in exposure_df.groupby(company_col):
        g = grp.sort_values(date_col).reset_index(drop=True)
        for i in range(len(g)-1):
            t_eom = pd.to_datetime(g.loc[i, date_col]).normalize()
            t1_eom = pd.to_datetime(g.loc[i+1, date_col]).normalize()

            trade_bd = previous_business_day(t_eom, countries, holiday_cache_dir)
            fix_bd   = previous_business_day(t1_eom, countries, holiday_cache_dir)

            if t_eom not in f.index or t1_eom not in s.index or t_eom not in s.index:
                continue

            spot_trade = float(s.loc[t_eom])
            spot_fix   = float(s.loc[t1_eom])
            actual_days = (t1_eom - t_eom).days
            fwd_trade_30d = float(f.loc[t_eom])

            # 입력 선도환율은 30일 만기 기준이다. 월마다 28~31일인 실제 만기에 맞추려고
            # 선도 프리미엄을 일수 비율로 선형 조정한다(금리 커브를 쓰지 않는 근사).
            if actual_days != 30:
                adjustment_factor = actual_days / 30.0
                fwd_premium = fwd_trade_30d - spot_trade
                fwd_trade_adjusted = spot_trade + (fwd_premium * adjustment_factor)
            else:
                fwd_trade_adjusted = fwd_trade_30d

            hedge = float(g.loc[i, "hedge_ratio"])
            ne = float(g.loc[i, "net_exposure"])
            notional_usd = abs(ne) * hedge / max(spot_trade, 1e-8)

            pnl = (spot_fix - fwd_trade_adjusted) * notional_usd
            if pd.isna(pnl) or pd.isna(notional_usd):
                continue
            if abs(notional_usd) < min_notional_threshold:
                continue

            pnl_rows.append({
                company_col: cid,
                "trade_month": t_eom,
                "fix_month": t1_eom,
                "trade_date_bd": trade_bd,
                "fix_date_bd": fix_bd,
                "notional_usd": float(notional_usd),
                "pnl_krw": float(pnl),
                "actual_maturity_days": actual_days,
                "forward_price_30d": float(fwd_trade_30d),
                "forward_price_adjusted": float(fwd_trade_adjusted),
                "maturity_adjustment": "adjusted" if actual_days != 30 else "standard"
            })

    df_trades = pd.DataFrame(pnl_rows)
    
    if include_open_positions:
        open_positions = []
        
        for cid, grp in exposure_df.groupby(company_col):
            g = grp.sort_values(date_col).reset_index(drop=True)
            if len(g) > 0:
                last_row = g.iloc[-1]
                last_month = pd.to_datetime(last_row[date_col]).normalize()
                
                hedge = float(last_row["hedge_ratio"])
                ne = float(last_row["net_exposure"])
                
                if abs(ne) * hedge >= min_notional_threshold and last_month in spot_eom.index:
                    spot_last = float(spot_eom.loc[last_month])
                    notional_usd = abs(ne) * hedge / max(spot_last, 1e-8)
                    
                    trade_bd = previous_business_day(last_month, countries, holiday_cache_dir)
                    
                    open_positions.append({
                        company_col: cid,
                        "trade_month": last_month,
                        "fix_month": pd.NaT,
                        "trade_date_bd": trade_bd,
                        "fix_date_bd": pd.NaT,
                        "notional_usd": float(notional_usd),
                        "pnl_krw": 0.0,
                        "status": "open"
                    })
        
        if open_positions:
            df_open = pd.DataFrame(open_positions)
            df_trades['status'] = 'closed'
            df_trades = pd.concat([df_trades, df_open], ignore_index=True)
    
    return df_trades
