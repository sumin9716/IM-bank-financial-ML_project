import pandas as pd
import numpy as np
from .pricing import cip_forward_theoretical

def _realized_vol_daily(spot_daily: pd.Series, window: int):
    if spot_daily is None or spot_daily.empty:
        return pd.Series(dtype=float)
    s = spot_daily.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    r = np.log(s).diff()
    rv = r.rolling(window).std() * np.sqrt(252)
    return rv

def _sma(series: pd.Series, window: int):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.rolling(window).mean()

def _sample_eom(series: pd.Series):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    idx = s.index.to_period('M').to_timestamp('M')
    s.index = idx
    return s.groupby(level=0).last()

def compute_features(spot_daily: pd.Series|None,
                     spot_eom: pd.Series,
                     r_us_eom: pd.Series,
                     r_kr_eom: pd.Series,
                     cfg: dict) -> pd.DataFrame:
    """월말 인덱스의 v1 정책 신호 표(rv20, rv60, ma20, ma60, carry)를 만든다.

    - rv/ma: 일별 환율이 있으면 일별 값으로 계산해 월말에 표본 추출하고,
      없으면 월말 시계열에 같은 창 길이를 적용한다(월 단위라 훨씬 거칠다).
    - carry: CIP 이론 선도환율 기준 선도 프리미엄(fwd/spot - 1)
    """
    pol = (cfg.get('policy') or {})
    fcfg = (pol.get('features') or {})
    rv_w = fcfg.get('rv_windows', [20,60])
    ma_w = fcfg.get('ma_windows', [20,60])
    tenor_days = int((cfg.get('pricing') or {}).get('tenor_days', 30))

    if spot_daily is not None and not spot_daily.empty:
        rv20_d = _realized_vol_daily(spot_daily, int(rv_w[0]))
        rv60_d = _realized_vol_daily(spot_daily, int(rv_w[1]))
        ma20_d = _sma(spot_daily, int(ma_w[0]))
        ma60_d = _sma(spot_daily, int(ma_w[1]))
        rv20 = _sample_eom(rv20_d); rv60 = _sample_eom(rv60_d)
        ma20 = _sample_eom(ma20_d); ma60 = _sample_eom(ma60_d)
    else:
        r = np.log(spot_eom).diff()
        rv20 = r.rolling(int(rv_w[0])).std() * np.sqrt(12)
        rv60 = r.rolling(int(rv_w[1])).std() * np.sqrt(12)
        ma20 = spot_eom.rolling(int(ma_w[0])).mean()
        ma60 = spot_eom.rolling(int(ma_w[1])).mean()

    fwd = cip_forward_theoretical(spot_eom, r_us_eom, r_kr_eom, days=tenor_days)
    carry = (fwd / spot_eom) - 1.0

    feats = pd.concat([rv20.rename('rv20'), rv60.rename('rv60'),
                       ma20.rename('ma20'), ma60.rename('ma60'),
                       carry.rename('carry')], axis=1)
    feats.index.name = 'month'
    feats = feats.reset_index()
    return feats
