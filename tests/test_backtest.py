import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    assert {"company_id","trade_month","fix_month","notional_usd","pnl_krw"}.issubset(set(pnl.columns))
    assert len(pnl) == 1
