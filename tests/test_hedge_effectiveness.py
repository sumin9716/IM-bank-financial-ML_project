import pandas as pd
import pytest

from fx_external_pipeline_full.hedge_effectiveness import (
    dollar_offset_ratio,
    judge_effective,
    regression_r2,
    regression_stats,
    sample_series,
)

def test_ifrs9_effectiveness_positive():
    idx = pd.to_datetime(["2024-01-31","2024-02-29","2024-03-31","2024-04-30"])
    # target ~ 1.0 * hedge + noise (very small)
    hedge = pd.Series([100, 105, 102, 108], index=idx)
    target = hedge + pd.Series([0.1, -0.1, 0.05, -0.05], index=idx)
    do = dollar_offset_ratio(target, hedge)
    r2 = regression_r2(target, hedge)
    verdict = judge_effective(do, r2, bounds=(0.8,1.25), r2_th=0.8)
    assert verdict == "Effective"


def test_dollar_offset_ratio_and_regression_on_exact_line():
    idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    hedge = pd.Series([100.0, 102.0, 101.0, 105.0, 104.0], index=idx)
    target = 2.0 * hedge + 3.0
    # |Δtarget| = 2·|Δhedge|
    assert dollar_offset_ratio(target, hedge) == pytest.approx(2.0)
    stats = regression_stats(target, hedge)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["alpha"] == pytest.approx(3.0)
    assert stats["r2"] == pytest.approx(1.0)
    assert stats["n"] == 5


def test_judge_effective_bounds_and_missing_values():
    assert judge_effective(1.30, 0.95, bounds=(0.8, 1.25), r2_th=0.8) == "Not effective"
    assert judge_effective(1.00, 0.50, bounds=(0.8, 1.25), r2_th=0.8) == "Not effective"
    assert judge_effective(float("nan"), 0.9) == "Insufficient data"


def test_sample_series_quarterly_rules():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    eom = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    q_end = sample_series(None, eom, period="Q", rule="month_end")
    q_avg = sample_series(None, eom, period="Q", rule="avg")
    assert list(q_end) == [3.0, 6.0]
    assert list(q_avg) == [2.0, 5.0]
    assert list(q_end.index) == list(pd.to_datetime(["2024-03-31", "2024-06-30"]))


def test_sample_series_prev_business_day_uses_daily_spot():
    eom = pd.Series([10.0], index=pd.to_datetime(["2024-03-31"]))
    daily = pd.Series([7.0, 8.0, 9.0], index=pd.to_datetime(["2024-03-27", "2024-03-28", "2024-03-29"]))
    # 2024-03-31은 일요일이므로 직전 영업일은 3월 29일(금)
    sampled = sample_series(daily, eom, period="Q", rule="prev_business_day")
    assert list(sampled) == [9.0]
