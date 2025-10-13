#!/usr/bin/env python
import sys, pandas as pd
from pathlib import Path
def check_file(path, cols):
    p=Path(path)
    if not p.exists(): return False, f"Missing file: {path}"
    try: df=pd.read_csv(path, nrows=5)
    except Exception as e: return False, f"Failed to read {path}: {e}"
    miss=[c for c in cols if c not in df.columns]
    return (len(miss)==0), (f"OK: {path}" if len(miss)==0 else f"{path} missing columns: {miss}")
def main():
    checks=[
        ("data/panel_base.csv", ["company_id","month","export_amt","import_amt"]),
        ("data/external/spot_usdkrw_eom.csv", ["month","spot"]),
        ("data/external/kr_rates_month.csv", ["month","rate"]),
        ("data/external/us_rates_month.csv", ["month","rate"]),
    ]
    ok_all=True
    for path, cols in checks:
        ok, msg=check_file(path, cols); print(("✅ " if ok else "❌ ")+msg); ok_all&=ok
    sys.exit(0 if ok_all else 1)
if __name__=="__main__": main()
