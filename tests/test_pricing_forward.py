import pandas as pd
from fx_external_pipeline_full.pricing import apply_spread, cip_forward_theoretical, ndf_settlement_pnl

def test_cip_forward_and_spread():
    idx = pd.to_datetime(["2024-01-31","2024-02-29"])
    spot = pd.Series([1300, 1310], index=idx)
    r_us = pd.Series([0.05, 0.05], index=idx)
    r_kr = pd.Series([0.03, 0.03], index=idx)
    theo = cip_forward_theoretical(spot, r_us, r_kr, days=30)
    adj  = apply_spread(theo, spread_bps=10, side="sell_usd")
    assert (adj > theo).all()


def test_cip_forward_matches_formula():
    idx = pd.to_datetime(["2024-01-31"])
    theo = cip_forward_theoretical(pd.Series([1300.0], index=idx), pd.Series([0.05], index=idx),
                                   pd.Series([0.03], index=idx), days=90)
    expected = 1300.0 * (1 + 0.05 * 90 / 365) / (1 + 0.03 * 90 / 365)
    assert abs(theo.iloc[0] - expected) < 1e-9


def test_ndf_settlement_pnl_aligns_dates():
    fixing = pd.Series([1310.0, 1290.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    agreed = pd.Series([1300.0], index=pd.to_datetime(["2024-01-31"]))
    pnl = ndf_settlement_pnl(fixing, agreed, notional_usd=1000)
    assert pnl.to_dict() == {pd.Timestamp("2024-01-31"): 10_000.0}
