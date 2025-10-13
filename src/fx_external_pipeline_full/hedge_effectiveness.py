import pandas as pd
import numpy as np
from scipy import stats
from .holiday_calendar import previous_business_day, is_business_day

def _align_xy(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    if y is None or x is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    yy = y.copy(); xx = x.copy()
    if not isinstance(yy.index, pd.DatetimeIndex): yy.index = pd.to_datetime(yy.index)
    if not isinstance(xx.index, pd.DatetimeIndex): xx.index = pd.to_datetime(xx.index)
    df = pd.concat([yy.rename("y"), xx.rename("x")], axis=1).dropna()
    return df["y"], df["x"]

def dollar_offset_ratio(y: pd.Series, x: pd.Series) -> float:
    """Prospective Dollar-Offset ratio: sum(|Δy|)/sum(|Δx|) using consecutive period ends."""
    y, x = _align_xy(y, x)
    if len(y) < 2 or len(x) < 2:
        return float("nan")
    dy = y.diff().abs().dropna()
    dx = x.diff().abs().dropna()
    if len(dy)==0 or len(dx)==0 or dx.sum()==0:
        return float("nan")
    return float(dy.sum() / dx.sum())

def regression_r2(y: pd.Series, x: pd.Series) -> float:
    y, x = _align_xy(y, x)
    if len(y) < 2:
        return float("nan")
    X = np.vstack([np.ones(len(x)), x.values]).T
    beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    yhat = X @ beta
    ss_res = np.sum((y.values - yhat)**2)
    ss_tot = np.sum((y.values - y.values.mean())**2) or np.nan
    if np.isnan(ss_tot) or ss_tot==0:
        return float("nan")
    r2 = 1 - ss_res/ss_tot
    return float(r2)

def regression_stats(y: pd.Series, x: pd.Series) -> dict:
    """Return beta(slope), alpha(intercept), R2, stderr_beta, t_beta, p_beta, n."""
    y, x = _align_xy(y, x)
    n = len(y)
    if n < 3:
        return {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "stderr_beta": np.nan, "t_beta": np.nan, "p_beta": np.nan, "n": n}
    X = np.vstack([np.ones(n), x.values]).T
    beta_hat = np.linalg.lstsq(X, y.values, rcond=None)[0]
    yhat = X @ beta_hat
    resid = y.values - yhat
    s2 = (resid @ resid) / (n - 2)  # residual variance
    cov = s2 * np.linalg.inv(X.T @ X)
    stderr_beta = np.sqrt(cov[1,1])
    t_beta = beta_hat[1] / (stderr_beta if stderr_beta>0 else np.nan)
    p_beta = 2 * stats.t.sf(np.abs(t_beta), df=n-2) if np.isfinite(t_beta) else np.nan
    r2 = regression_r2(y, x)
    return {"alpha": float(beta_hat[0]), "beta": float(beta_hat[1]), "r2": float(r2),
            "stderr_beta": float(stderr_beta), "t_beta": float(t_beta), "p_beta": float(p_beta), "n": int(n)}

def judge_effective(do_ratio: float, r2: float, bounds=(0.8,1.25), r2_th=0.8) -> str:
    if not np.isfinite(do_ratio) or not np.isfinite(r2):
        return "Insufficient data"
    lo, hi = bounds
    if (do_ratio>=lo and do_ratio<=hi) and (r2>=r2_th):
        return "Effective"
    return "Not effective"

def _last_valid(series: pd.Series, dt: pd.Timestamp):
    if series is None or series.empty:
        return np.nan
    s = series[series.index <= dt]
    if len(s)==0: return np.nan
    return float(s.iloc[-1])

def sample_series(spot_daily: pd.Series|None, eom_series: pd.Series, period='M', rule='month_end', countries=('KR',), cache_dir='data/reference/holidays_cache') -> pd.Series:
    """Sample price series by period with rule:
    - period: 'M' monthly, 'Q' quarterly
    - rule:
        'month_end' : use EOM values (from eom_series)
        'prev_business_day' : take last business day of period; prefer daily spot if available, else EOM
        'avg' : period average (daily if available else monthly EOM mean)
    Returns a DatetimeIndex series at period ends.
    """
    if eom_series is None or eom_series.empty:
        return pd.Series(dtype=float)
    if not isinstance(eom_series.index, pd.DatetimeIndex):
        eom = pd.to_datetime(eom_series.index)
        eom_series = pd.Series(eom_series.values, index=eom)
    # determine period ends
    if period.upper()=='Q':
        p_ends = eom_series.index.to_period('Q').to_timestamp('Q')
    else:
        p_ends = eom_series.index.to_period('M').to_timestamp('M')

    out = []
    for pend in sorted(pd.unique(p_ends)):
        if rule == 'avg':
            if spot_daily is not None and not spot_daily.empty:
                start = (pd.Timestamp(pend).to_period(period.upper()).start_time)
                end = pd.Timestamp(pend)
                sd = spot_daily.copy()
                if not isinstance(sd.index, pd.DatetimeIndex): sd.index = pd.to_datetime(sd.index)
                mask = (sd.index>=start) & (sd.index<=end)
                val = float(sd.loc[mask].mean()) if mask.any() else _last_valid(eom_series, pend)
            else:
                # average of EOM in the period (effectively the EOM of that period)
                val = _last_valid(eom_series, pend)
        elif rule == 'prev_business_day':
            dt = pd.Timestamp(pend)
            # ensure prev business day (for KR by default)
            bday = previous_business_day(dt, countries=countries, cache_dir=cache_dir)
            if spot_daily is not None and not spot_daily.empty:
                val = _last_valid(spot_daily, bday)
                if not np.isfinite(val):
                    val = _last_valid(eom_series, pend)
            else:
                val = _last_valid(eom_series, pend)
        else:  # 'month_end'
            val = _last_valid(eom_series, pend)
        out.append((pend, val))
    return pd.Series({d:v for d,v in out}).dropna()
