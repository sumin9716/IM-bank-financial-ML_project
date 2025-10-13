import pandas as pd
from pathlib import Path
import holidays as pyhol

def _build_country_year(country: str, year: int) -> pd.DataFrame:
    if country.upper()=='KR':
        hol = pyhol.KR(years=year)
    elif country.upper()=='US':
        hol = pyhol.US(years=year)
    else:
        hol = pyhol.CountryHoliday(country.upper(), years=year)
    return pd.DataFrame([{"date": pd.Timestamp(d), "name": name, "country": country} for d, name in hol.items()])

def cache_holidays(countries, years, cache_dir: str):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for c in countries:
        for y in years:
            fp = Path(cache_dir) / f"holidays_{c}_{y}.csv"
            if not fp.exists():
                _build_country_year(c, y).to_csv(fp, index=False, encoding="utf-8-sig")

def load_holidays(countries, years, cache_dir: str) -> pd.DataFrame:
    cache_holidays(countries, years, cache_dir)
    dfs=[]
    for c in countries:
        for y in years:
            fp = Path(cache_dir) / f"holidays_{c}_{y}.csv"
            if fp.exists():
                dfs.append(pd.read_csv(fp, parse_dates=["date"]))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["date","name","country"])

def is_business_day(date, countries=['KR'], cache_dir="data/reference/holidays_cache"):
    y = int(pd.Timestamp(date).year)
    hol = load_holidays(countries, [y], cache_dir)
    d = pd.Timestamp(date).normalize()
    return (d.weekday() < 5) and (not (hol["date"]==d).any())
