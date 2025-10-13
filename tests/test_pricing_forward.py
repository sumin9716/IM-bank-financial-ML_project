import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fx_external_pipeline_full.pricing import cip_forward_theoretical, apply_spread

def test_cip_forward_and_spread():
    idx = pd.to_datetime(["2024-01-31","2024-02-29"])
    spot = pd.Series([1300, 1310], index=idx)
    r_us = pd.Series([0.05, 0.05], index=idx)
    r_kr = pd.Series([0.03, 0.03], index=idx)
    theo = cip_forward_theoretical(spot, r_us, r_kr, days=30)
    adj  = apply_spread(theo, spread_bps=10, side="sell_usd")
    assert (adj > theo).all()
