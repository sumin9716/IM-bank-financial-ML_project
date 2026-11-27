import pandas as pd

from fx_external_pipeline_full import fred_loader


def _fake_observations(series_id, api_key, start=None, end=None):
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"]),
        "value": [1320.0, 1335.0, 1330.0, None],
    })


def test_daily_loader_keeps_daily_observations(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    monkeypatch.setattr(fred_loader, "_fred_get", _fake_observations)
    daily = fred_loader.load_usdkrw_spot_daily_from_fred()
    assert list(daily.index) == list(pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28"]))


def test_eom_loader_keeps_last_value_per_month(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    monkeypatch.setattr(fred_loader, "_fred_get", _fake_observations)
    eom = fred_loader.load_usdkrw_spot_eom_from_fred()
    assert list(eom.index) == list(pd.to_datetime(["2024-01-31", "2024-02-29"]))
    assert eom.iloc[0] == 1335.0
