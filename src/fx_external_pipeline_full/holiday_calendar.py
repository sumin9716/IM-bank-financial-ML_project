from functools import lru_cache
from pathlib import Path

import holidays as pyhol
import pandas as pd

DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "data" / "reference" / "holidays_cache")


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


@lru_cache(maxsize=None)
def _holiday_dates(countries: tuple, year: int, cache_dir: str) -> frozenset:
    # 백테스트는 거래마다 영업일을 확인하므로 연도별 휴일 CSV를 한 번만 읽어 둔다.
    hol = load_holidays(countries, [year], cache_dir)
    return frozenset(pd.to_datetime(hol["date"]))


def is_business_day(date, countries=("KR",), cache_dir=DEFAULT_CACHE_DIR):
    d = pd.Timestamp(date).normalize()
    return (d.weekday() < 5) and (d not in _holiday_dates(tuple(countries), d.year, str(cache_dir)))

def previous_business_day(date, countries=("KR",), cache_dir=DEFAULT_CACHE_DIR):
    """주어진 날짜가 영업일이면 그대로, 아니면 직전 영업일을 돌려준다."""
    d = pd.Timestamp(date).normalize()
    while True:
        if is_business_day(d, countries, cache_dir):
            return d
        d = d - pd.Timedelta(days=1)
