import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import numpy as np
import pytest

from fx_external_pipeline_full.risk import expected_shortfall, historical_var

def test_risk_metrics_return_numbers():
    pnl = pd.Series([ -10, 5, -3, 7, -2, 1, -8, 4 ], index=pd.date_range("2024-01-01", periods=8, freq="D"))
    var = historical_var(pnl, alpha=0.9, window=8)
    es  = expected_shortfall(pnl, alpha=0.9, window=8)
    assert isinstance(var, float)
    assert isinstance(es, float)


def test_var_and_es_use_lower_tail_of_recent_window():
    pnl = pd.Series([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -50.0])
    # 마지막 10개(-100 제외)만 본다
    var = historical_var(pnl, alpha=0.9, window=10)
    es = expected_shortfall(pnl, alpha=0.9, window=10)
    assert var == pytest.approx(float(np.quantile(pnl.iloc[-10:], 0.1)))
    assert es == -50.0
