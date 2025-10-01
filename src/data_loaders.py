
from __future__ import annotations
import os, json
from typing import Optional, List, Dict
from datetime import datetime, date
import pandas as pd
import requests, yaml

from .utils.cache import cache_path, read_csv_if_fresh, write_csv

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY_ENV = "dce9672c424568675b65637d7328f911"
ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"

def _to_dt(x):
    if x is None: 
        return None
    if isinstance(x, (datetime, pd.Timestamp, date)):
        return pd.to_datetime(x).date().isoformat()
    return pd.to_datetime(x).date().isoformat()

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _requests_get(url: str, params: dict | None=None, timeout: int=30, retries: int=3):
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last_exc = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_exc = e
    raise last_exc

def _fred_observations(series_id: str, start: str|None, end: str|None, api_key: str) -> pd.DataFrame:
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start: params["observation_start"] = start
    if end: params["observation_end"] = end
    r = _requests_get(FRED_BASE, params=params)
    data = r.json()
    if "observations" not in data:
        raise RuntimeError(f"Unexpected FRED payload for {series_id}: {str(data)[:200]}")
    rows = []
    for obs in data["observations"]:
        val = obs.get("value", None)
        try: v = float(val)
        except Exception: v = None
        rows.append({"date": obs["date"], "value": v})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

def _ecos_series(api_key: str, stat_code: str, cycle: str, start: str, end: str,
                 item_code1: str="", item_code2: str="", item_code3: str="") -> pd.DataFrame:
    url = "/".join([ECOS_BASE, api_key, "json", "kr", "1", "100000", stat_code, cycle, start, end, item_code1, item_code2, item_code3])
    r = _requests_get(url)
    js = r.json()
    rows = None
    for k, v in js.items():
        if isinstance(v, dict) and "row" in v:
            rows = v["row"]; break
    if rows is None:
        rows = js.get("StatisticSearch", {}).get("row", None)
    if rows is None:
        raise RuntimeError(f"Unexpected ECOS payload: {str(js)[:200]}")
    out = []
    for rr in rows:
        out.append({"date": rr.get("TIME"), "value": rr.get("DATA_VALUE")})
    df = pd.DataFrame(out)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

def load_us_rates(tenors: List[str] = ["SOFR","DGS1","DGS2","DGS10"],
                  start: str|None="2000-01-01", end: str|None=None,
                  fred_api_key: Optional[str]=None,
                  fred_config_path: str|None=None,
                  cache_hours: int=24) -> pd.DataFrame:
    start, end = _to_dt(start), _to_dt(end) if end else (start, None)[1]
    fred_api_key = fred_api_key or os.getenv(FRED_API_KEY_ENV, "")
    if not fred_api_key:
        raise EnvironmentError(f"Set {FRED_API_KEY_ENV} or pass fred_api_key.")
    cfg_path = fred_config_path or os.path.join(os.path.dirname(__file__), "configs", "fred_series.yml")
    cfg = _load_yaml(cfg_path)
    mapping = cfg.get("us_rates", {})
    rows = []
    for t in tenors:
        sid = mapping.get(t)
        if not sid:
            raise KeyError(f"Tenor '{t}' not found in fred_series.yml under 'us_rates'.")
        cache_file = cache_path("fred", f"{sid}_{start or 'start'}_{end or 'end'}.csv")
        cached = read_csv_if_fresh(cache_file, max_age_hours=cache_hours)
        if cached is None:
            df = _fred_observations(sid, start, end, fred_api_key)
            df["series"] = t
            write_csv(df, cache_file)
        else:
            df = cached; df["series"] = t
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["source"] = "FRED"
    return out[["date","series","value","source"]].sort_values(["series","date"]).reset_index(drop=True)

def load_usdkrw_spot(source: str="FRED",
                     start: str|None="2000-01-01", end: str|None=None,
                     fred_api_key: Optional[str]=None,
                     fred_config_path: str|None=None,
                     ecos_api_key: Optional[str]=None,
                     ecos_config_path: str|None=None,
                     cache_hours: int=24) -> pd.DataFrame:
    start, end = _to_dt(start), _to_dt(end) if end else (start, None)[1]
    if source.upper() == "FRED":
        fred_api_key = fred_api_key or os.getenv(FRED_API_KEY_ENV, "")
        if not fred_api_key:
            raise EnvironmentError(f"Set {FRED_API_KEY_ENV} or pass fred_api_key.")
        cfg_path = fred_config_path or os.path.join(os.path.dirname(__file__), "configs", "fred_series.yml")
        cfg = _load_yaml(cfg_path)
        sid = cfg.get("fx", {}).get("USDKRW")
        if not sid:
            raise KeyError("Missing fx.USDKRW in fred_series.yml")
        cache_file = cache_path("fred", f"{sid}_{start or 'start'}_{end or 'end'}.csv")
        cached = read_csv_if_fresh(cache_file, max_age_hours=cache_hours)
        if cached is None:
            df = _fred_observations(sid, start, end, fred_api_key)
            write_csv(df, cache_file)
        else:
            df = cached
        df["source"] = "FRED"
        return df[["date","value","source"]].sort_values("date").reset_index(drop=True)
    elif source.upper() == "ECOS":
        ecos_api_key = ecos_api_key or os.getenv("ECOS_API_KEY", "")
        if not ecos_api_key:
            raise EnvironmentError("Set ECOS_API_KEY or pass ecos_api_key.")
        cfg_path = ecos_config_path or os.path.join(os.path.dirname(__file__), "configs", "ecos_series.yml")
        cfg = _load_yaml(cfg_path)
        fxmap = cfg.get("fx", {}).get("USDKRW")
        if not fxmap:
            raise KeyError("Missing fx.USDKRW mapping in ecos_series.yml")
        stat_code = fxmap["stat_code"]; cycle = fxmap["cycle"]
        item_code1 = fxmap.get("item_code1",""); item_code2 = fxmap.get("item_code2",""); item_code3 = fxmap.get("item_code3","")
        s = (start or "2000-01-01").replace("-","")
        e = (end or datetime.today().date().isoformat()).replace("-","")
        cache_file = cache_path("ecos", f"{stat_code}_{cycle}_{s}_{e}_{item_code1}_{item_code2}_{item_code3}.csv")
        cached = read_csv_if_fresh(cache_file, max_age_hours=cache_hours)
        if cached is None:
            df = _ecos_series(ecos_api_key, stat_code, cycle, s, e, item_code1, item_code2, item_code3)
            write_csv(df, cache_file)
        else:
            df = cached
        df["source"] = "ECOS"
        return df[["date","value","source"]].sort_values("date").reset_index(drop=True)
    else:
        raise ValueError("source must be 'FRED' or 'ECOS'")

def load_kr_rates(series_keys: list[str] = ["CD91","CALL_OVN"],
                  start: str|None="2000-01-01", end: str|None=None,
                  ecos_api_key: Optional[str]=None,
                  ecos_config_path: str|None=None,
                  cache_hours: int=24) -> pd.DataFrame:
    start, end = _to_dt(start), _to_dt(end) if end else (start, None)[1]
    ecos_api_key = ecos_api_key or os.getenv("ECOS_API_KEY", "")
    if not ecos_api_key:
        raise EnvironmentError("Set ECOS_API_KEY or pass ecos_api_key.")
    cfg_path = ecos_config_path or os.path.join(os.path.dirname(__file__), "configs", "ecos_series.yml")
    cfg = _load_yaml(cfg_path)
    rows = []
    for k in series_keys:
        meta = cfg.get("kr_rates", {}).get(k)
        if not meta:
            raise KeyError(f"Series '{k}' not found in ecos_series.yml under 'kr_rates'.")
        stat_code = meta["stat_code"]; cycle = meta["cycle"]
        item_code1 = meta.get("item_code1",""); item_code2 = meta.get("item_code2",""); item_code3 = meta.get("item_code3","")
        s = (start or "2000-01-01").replace("-","")
        e = (end or datetime.today().date().isoformat()).replace("-","")
        cache_file = cache_path("ecos", f"{k}_{stat_code}_{cycle}_{s}_{e}_{item_code1}_{item_code2}_{item_code3}.csv")
        cached = read_csv_if_fresh(cache_file, max_age_hours=cache_hours)
        if cached is None:
            df = _ecos_series(ecos_api_key, stat_code, cycle, s, e, item_code1, item_code2, item_code3)
            df["series"] = k
            write_csv(df, cache_file)
        else:
            df = cached; df["series"] = k
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["source"] = "ECOS"
    return out[["date","series","value","source"]].sort_values(["series","date"]).reset_index(drop=True)

def load_holidays(country: str="KR",
                  years: list[int]|None=None,
                  language: str="KO",
                  cache_hours: int=168) -> pd.DataFrame:
    if years is None:
        years = [datetime.today().year]
    base = "https://openholidaysapi.org/PublicHolidays"
    required_cols = ["date", "localName", "name", "country"]
    all_rows = []
    for y in years:
        params = {
            "countryIsoCode": country,
            "languageIsoCode": language,
            "validFrom": f"{y}-01-01",
            "validTo": f"{y}-12-31",
        }
        cache_file = cache_path("holidays", f"{country}_{language}_{y}.csv")
        cached = read_csv_if_fresh(cache_file, max_age_hours=cache_hours)
        if cached is not None:
            df = cached.copy()
        else:
            r = _requests_get(base, params=params)
            js = r.json()
            rows = [{
                "date": it.get("startDate"),
                "localName": it.get("localName"),
                "name": it.get("name"),
                "country": country,
            } for it in js]
            df = pd.DataFrame(rows, columns=required_cols)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            write_csv(df, cache_file)
        missing = [col for col in required_cols if col not in df.columns]
        for col in missing:
            df[col] = pd.Series(dtype="object")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df[required_cols]
        all_rows.append(df)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=required_cols)
    out["source"] = "OpenHolidays"
    return out[required_cols + ["source"]].sort_values(["date"]).reset_index(drop=True)
