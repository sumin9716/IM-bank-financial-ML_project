import pandas as pd

from fx_external_pipeline_full.holiday_calendar import (
    DEFAULT_CACHE_DIR,
    cache_holidays,
    is_business_day,
    previous_business_day,
)


def test_weekend_and_cached_holiday_are_not_business_days():
    assert not is_business_day("2024-02-10", ("KR",), DEFAULT_CACHE_DIR)  # 토요일·설날
    assert not is_business_day("2024-02-09", ("KR",), DEFAULT_CACHE_DIR)  # 설날 전날(금)
    assert is_business_day("2024-02-08", ("KR",), DEFAULT_CACHE_DIR)


def test_previous_business_day_skips_lunar_new_year():
    assert previous_business_day("2024-02-11", ("KR",), DEFAULT_CACHE_DIR) == pd.Timestamp("2024-02-08")
    assert previous_business_day("2024-02-08", ("KR",), DEFAULT_CACHE_DIR) == pd.Timestamp("2024-02-08")


def test_cache_holidays_writes_missing_year(tmp_path):
    cache_holidays(["US"], [2030], str(tmp_path))
    cached = pd.read_csv(tmp_path / "holidays_US_2030.csv", parse_dates=["date"])
    assert pd.Timestamp("2030-07-04") in set(cached["date"])
