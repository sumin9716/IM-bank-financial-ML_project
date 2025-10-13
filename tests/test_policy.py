import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fx_external_pipeline_full.policy import apply_policy

def test_apply_policy_thresholds():
    df = pd.DataFrame({
        "company_id": ["s","m","l"],
        "month": pd.to_datetime(["2024-01-31"]*3),
        "export_amt": [0,0,0],
        "import_amt": [0,0,0],
        "net_exposure": [5e5, 2e6, 10e6], # small, medium, large
    })
    out = apply_policy(df)
    assert list(out["hedge_ratio"]) == [0.0, 0.5, 0.8]
