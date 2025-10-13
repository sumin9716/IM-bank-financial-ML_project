import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fx_external_pipeline_full.risk import historical_var, expected_shortfall

def test_risk_metrics_return_numbers():
    pnl = pd.Series([ -10, 5, -3, 7, -2, 1, -8, 4 ], index=pd.date_range("2024-01-01", periods=8, freq="D"))
    var = historical_var(pnl, alpha=0.9, window=8)
    es  = expected_shortfall(pnl, alpha=0.9, window=8)
    assert isinstance(var, float)
    assert isinstance(es, float)
