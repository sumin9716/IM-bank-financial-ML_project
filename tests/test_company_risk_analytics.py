import numpy as np
import pandas as pd
import pytest

from fx_external_pipeline_full.company_risk_analytics import (
    classify_exposure_size,
    classify_risk_rating,
    create_company_risk_analytics,
)


@pytest.mark.parametrize("amount, expected", [
    (1_000_000, "Very High"), (999_999, "High"), (500_000, "High"),
    (100_000, "Medium"), (50_000, "Low"), (49_999.99, "Very Low"), (0, "Very Low"),
])
def test_classify_risk_rating_bands(amount, expected):
    assert classify_risk_rating(amount) == expected


@pytest.mark.parametrize("amount, expected", [
    (10_000_000, "XL"), (5_000_000, "L"), (1_000_000, "M"), (500_000, "S"), (499_999, "XS"),
])
def test_classify_exposure_size_bands(amount, expected):
    assert classify_exposure_size(amount) == expected


def test_create_company_risk_analytics_filters_small_and_scales_var(tmp_path):
    spot = pd.DataFrame({"month": pd.date_range("2024-01-31", periods=6, freq="ME"),
                         "spot": [1300, 1310, 1290, 1320, 1305, 1330]})
    exposure = pd.DataFrame({
        "company_id": [1, 1, 2],
        "month": ["2024-01-31", "2024-02-29", "2024-01-31"],
        "net_exposure": [2_000_000.0, -400_000.0, 1_500.0],
        "hedge_ratio": [0.5, 0.0, 0.5],
    })
    spot_file, exposure_file, out_file = tmp_path / "spot.csv", tmp_path / "exp.csv", tmp_path / "out.csv"
    spot.to_csv(spot_file, index=False)
    exposure.to_csv(exposure_file, index=False)

    report, summary = create_company_risk_analytics(exposure_file, out_file, spot_file=spot_file)

    # 기업 2의 미헤지 노출(750)은 1,000 미만이라 빠진다
    assert summary["total_records"] == 2
    assert set(report["company_id"]) == {1}
    returns = spot["spot"].pct_change().dropna()
    first = report.iloc[0]
    assert first["unhedged_exposure"] == 1_000_000.0
    assert first["var_99_amount"] == round(abs(1_000_000 * np.percentile(returns, 1)), 2)
    assert out_file.exists()
