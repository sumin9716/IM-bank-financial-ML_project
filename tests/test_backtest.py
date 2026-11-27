import pandas as pd
from fx_external_pipeline_full.backtest import monthly_forward_strategy

def test_monthly_forward_strategy_shape():
    exposure = pd.DataFrame({
        "company_id":["A","A"],
        "month": pd.to_datetime(["2024-01-31","2024-02-29"]),
        "net_exposure":[1_000_000, 1_000_000],
        "hedge_ratio":[0.5, 0.5],
    })
    idx = pd.to_datetime(["2024-01-31","2024-02-29"])
    spot = pd.Series([1300, 1320], index=idx)
    fwd  = pd.Series([1310, 1330], index=idx)
    pnl = monthly_forward_strategy(exposure, spot, fwd)
    assert {"company_id","trade_month","fix_month","notional_usd","pnl_krw","status"}.issubset(set(pnl.columns))
    # 마지막 노출월은 만기가 없으므로 체결 1건과 미체결 포지션 1건이 나온다
    assert (pnl["status"] == "closed").sum() == 1
    assert (pnl["status"] == "open").sum() == 1


def test_monthly_forward_strategy_without_open_positions():
    exposure = pd.DataFrame({
        "company_id":["A","A"],
        "month": pd.to_datetime(["2024-01-31","2024-02-29"]),
        "net_exposure":[1_000_000, 1_000_000],
        "hedge_ratio":[0.5, 0.5],
    })
    idx = pd.to_datetime(["2024-01-31","2024-02-29"])
    spot = pd.Series([1300, 1320], index=idx)
    fwd  = pd.Series([1310, 1330], index=idx)
    pnl = monthly_forward_strategy(exposure, spot, fwd, include_open_positions=False)
    assert len(pnl) == 1
    assert "status" not in pnl.columns
