import os, requests, pandas as pd
from .utils import to_month_end

ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"

def _ecos_get(stat_code, cycle, start, end, item_code, api_key, lang="json", kr="kr"):
    url = f"{ECOS_BASE}/{api_key}/{lang}/{kr}/1/10000/{stat_code}/{cycle}/{start}/{end}/{item_code}"
    r = requests.get(url, timeout=20); r.raise_for_status()
    js = r.json(); rows = (js.get("StatisticSearch") or {}).get("row") or []
    if not rows: return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(rows)
    time_col = "TIME" if "TIME" in df.columns else next((c for c in df.columns if c.lower()=="time"), None)
    val_col = "DATA_VALUE" if "DATA_VALUE" in df.columns else next((c for c in df.columns if c.lower()=="data_value"), None)
    df["date"] = pd.to_datetime(df[time_col].astype(str).str.replace(r"[^0-9]","", regex=True).apply(lambda t: t[:6] + "01"))
    df["value"] = pd.to_numeric(df[val_col].replace({"": None, ".": None}), errors="coerce")
    return df[["date","value"]].dropna()

def load_ecos_series(stat_code, cycle, start, end, item_code=""):
    api_key = os.getenv("ECOS_API_KEY", "")
    if not api_key: raise EnvironmentError("ECOS_API_KEY is not set")
    df = _ecos_get(stat_code, cycle, start, end, item_code, api_key)
    if df.empty: return pd.Series(dtype=float)
    s = df.set_index("date")["value"].sort_index()
    return to_month_end(s).groupby(level=0).last()
