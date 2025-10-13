import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fx_external_pipeline_full.exposure import compute_monthly_exposure

def test_compute_monthly_exposure_basic():
    df = pd.DataFrame({
        "company_id": ["A","A","B"],
        "month": pd.to_datetime(["2024-01-31","2024-01-31","2024-02-29"]),
        "export_amt": [100, 50, 200],
        "import_amt": [30, 10, 120],
    })
    out = compute_monthly_exposure(df)
    assert {"company_id","month","export_amt","import_amt","net_exposure"}.issubset(set(out.columns))
    # A@2024-01: export=150 import=40 net=110
    net_a = float(out.loc[(out["company_id"]=="A") & (out["month"]==pd.Timestamp("2024-01-31")),"net_exposure"])
    assert abs(net_a - 110) < 1e-6
