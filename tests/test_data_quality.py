import pandas as pd
import pytest

from fx_external_pipeline_full.data_quality import fill_monthly_gaps, validate_monthly_continuity


def _exposure_with_gap():
    return pd.DataFrame({
        "company_id": ["A", "A", "B", "B"],
        "month": pd.to_datetime(["2024-01-31", "2024-03-31", "2024-01-31", "2024-02-29"]),
        "net_exposure": [100.0, 300.0, 10.0, 20.0],
    })


def test_validate_monthly_continuity_finds_missing_month():
    result = validate_monthly_continuity(_exposure_with_gap())
    assert result["has_gaps"]
    assert result["companies_with_gaps"] == ["A"]
    assert result["gap_details"] == [{"company_id": "A", "missing_month": "2024-02", "gap_type": "missing_month"}]


def test_fill_monthly_gaps_zero_fills_numeric_columns():
    filled = fill_monthly_gaps(_exposure_with_gap(), fill_method="zero")
    a = filled[filled["company_id"] == "A"].set_index("month")["net_exposure"]
    assert list(a.index) == list(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]))
    assert a.loc["2024-02-29"] == 0.0
    assert len(filled) == 5


@pytest.mark.filterwarnings("error")
def test_fill_monthly_gaps_marks_text_columns():
    df = _exposure_with_gap().assign(industry=["mfg", "mfg", "svc", "svc"])
    filled = fill_monthly_gaps(df, fill_method="zero")
    added = filled[(filled["company_id"] == "A") & (filled["month"] == pd.Timestamp("2024-02-29"))]
    assert added["industry"].tolist() == ["filled"]
    assert added["company_id"].tolist() == ["A"]
