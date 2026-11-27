#!/usr/bin/env python
"""패널의 월 범위에 맞춰 합성 시장 데이터(USD/KRW 월말 환율, 한국·미국 금리)를 만든다.

data/external/의 CSV를 덮어쓴다. 실제 시장 데이터가 아니라 파이프라인 실행용 예시 값이다
(난수 시드 42의 누적 정규 난수).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_PATH = PROJECT_ROOT / "data" / "panel_base.csv"
OUT_DIR = PROJECT_ROOT / "data" / "external"


def panel_months() -> list:
    """패널에 있는 월(월말 날짜). 패널을 읽을 수 없으면 2024년 12개월."""
    default = pd.period_range("2024-01", "2024-12", freq="M").to_timestamp("M").tolist()
    try:
        m = pd.read_csv(PANEL_PATH, parse_dates=["month"])["month"].dropna()
    except (FileNotFoundError, ValueError):
        return default
    if len(m) == 0:
        return default
    return pd.Series(m.dt.to_period("M").dt.to_timestamp("M")).drop_duplicates().sort_values().tolist()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months = panel_months()
    rng = np.random.default_rng(42)
    n = len(months)
    spot = 1100 + rng.normal(0, 30, size=n).cumsum() / 5
    kr = 0.035 + rng.normal(0, 0.001, size=n).cumsum() / 10
    us = 0.045 + rng.normal(0, 0.001, size=n).cumsum() / 10
    pd.DataFrame({"month": months, "spot": spot}).to_csv(OUT_DIR / "spot_usdkrw_eom.csv", index=False)
    pd.DataFrame({"month": months, "rate": kr}).to_csv(OUT_DIR / "kr_rates_month.csv", index=False)
    pd.DataFrame({"month": months, "rate": us}).to_csv(OUT_DIR / "us_rates_month.csv", index=False)
    print("Templates written to", OUT_DIR)


if __name__ == "__main__":
    main()
