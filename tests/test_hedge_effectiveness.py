import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, numpy as np
from fx_external_pipeline_full.hedge_effectiveness import dollar_offset_ratio, regression_r2, judge_effective

def test_ifrs9_effectiveness_positive():
    idx = pd.to_datetime(["2024-01-31","2024-02-29","2024-03-31","2024-04-30"])
    # target ~ 1.0 * hedge + noise (very small)
    hedge = pd.Series([100, 105, 102, 108], index=idx)
    target = hedge + pd.Series([0.1, -0.1, 0.05, -0.05], index=idx)
    do = dollar_offset_ratio(target, hedge)
    r2 = regression_r2(target, hedge)
    verdict = judge_effective(do, r2, bounds=(0.8,1.25), r2_th=0.8)
    assert verdict == "Effective"
