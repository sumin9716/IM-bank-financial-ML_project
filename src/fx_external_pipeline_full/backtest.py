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
                             holiday_cache_dir: str = "data/reference/holidays_cache",
                             min_notional_threshold: float = 1e-6,
                             include_open_positions: bool = True) -> pd.DataFrame:
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
            
            # 실제 만기 일수 계산
            actual_days = (t1_eom - t_eom).days
            
            # 원래 선도가격 (30일 가정)
            fwd_trade_30d = float(f.loc[t_eom])
            
            # 실제 만기에 맞는 선도가격 조정 (간단한 선형 보간)
            # 실제로는 이자율 커브를 사용해야 하지만, 여기서는 근사치로 처리
            if actual_days != 30:
                # 30일 대비 실제 일수 비율로 선도 프리미엄 조정
                adjustment_factor = actual_days / 30.0
                fwd_premium = fwd_trade_30d - spot_trade
                fwd_trade_adjusted = spot_trade + (fwd_premium * adjustment_factor)
            else:
                fwd_trade_adjusted = fwd_trade_30d

            hedge = float(g.loc[i, "hedge_ratio"])
            ne = float(g.loc[i, "net_exposure"])
            notional_usd = abs(ne) * hedge / max(spot_trade, 1e-8)

            pnl = (spot_fix - fwd_trade_adjusted) * notional_usd
            
            # 데이터 품질 검증
            if pd.isna(pnl) or pd.isna(notional_usd):
                continue
                
            # 명목금액이 임계값 미만인 무의미한 트레이드 제외
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
    
    # 미체결 포지션 처리 (마지막 노출월에 대한 열린 포지션 정보)
    if include_open_positions:
        open_positions = []
        
        for cid, grp in exposure_df.groupby(company_col):
            g = grp.sort_values(date_col).reset_index(drop=True)
            if len(g) > 0:
                # 마지막 노출월 (미체결 포지션)
                last_row = g.iloc[-1]
                last_month = pd.to_datetime(last_row[date_col]).normalize()
                
                hedge = float(last_row["hedge_ratio"])
                ne = float(last_row["net_exposure"])
                
                # 명목금액이 임계값 이상인 경우만 포함
                if abs(ne) * hedge >= min_notional_threshold and last_month in spot_eom.index:
                    spot_last = float(spot_eom.loc[last_month])
                    notional_usd = abs(ne) * hedge / max(spot_last, 1e-8)
                    
                    trade_bd = _adjust_prev_bday(last_month, countries, holiday_cache_dir)
                    
                    open_positions.append({
                        company_col: cid,
                        "trade_month": last_month,
                        "fix_month": None,  # 미체결
                        "trade_date_bd": trade_bd,
                        "fix_date_bd": None,  # 미체결
                        "notional_usd": float(notional_usd),
                        "pnl_krw": 0.0,  # 미실현
                        "status": "open"
                    })
        
        # 열린 포지션 정보를 별도 처리 (메인 거래 데이터와 구분)
        if open_positions:
            df_open = pd.DataFrame(open_positions)
            # 실제 거래와 열린 포지션을 구분하기 위해 status 컬럼 추가
            df_trades['status'] = 'closed'
            df_trades = pd.concat([df_trades, df_open], ignore_index=True)
    
    return df_trades
