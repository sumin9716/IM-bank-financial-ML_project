import os, requests, pandas as pd
from .utils import to_month_end

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def _fred_get(series_id, api_key, start=None, end=None):
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start: params["observation_start"] = start
    if end: params["observation_end"] = end
    r = requests.get(FRED_BASE, params=params, timeout=20); r.raise_for_status()
    js = r.json(); rows = js.get("observations", [])
    if not rows: return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(rows)[["date","value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace({".": None}), errors="coerce")
    return df

def load_fred_series(series_id, start=None, end=None, value_to_decimal=False):
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key: raise EnvironmentError("FRED_API_KEY is not set")
    df = _fred_get(series_id, api_key, start, end)
    if df.empty: return pd.Series(dtype=float)
    s = df.set_index("date")["value"].sort_index()
    if value_to_decimal: s = s/100.0
    return to_month_end(s).groupby(level=0).last()

def load_usdkrw_spot_eom_from_fred(start=None, end=None): return load_fred_series("DEXKOUS", start, end, False)
def load_us_1y_yield_eom_from_fred(start=None, end=None): return load_fred_series("DGS1", start, end, True)
def load_sofr_eom_from_fred(start=None, end=None): return load_fred_series("SOFR", start, end, True)
def load_usdkrw_spot_daily_from_fred(start=None, end=None): return load_fred_series("DEXKOUS", start, end, False)
