#!/usr/bin/env python
"""파이프라인 입력 CSV가 있고 필요한 열을 갖췄는지 확인한다. 하나라도 빠지면 종료 코드 1."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ("data/panel_base.csv", ["company_id", "month", "export_amt", "import_amt"]),
    ("data/external/spot_usdkrw_eom.csv", ["month", "spot"]),
    ("data/external/kr_rates_month.csv", ["month", "rate"]),
    ("data/external/us_rates_month.csv", ["month", "rate"]),
]


def check_file(path: Path, cols: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing file: {path}"
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:  # 인코딩·구분자 오류 등 읽기 실패를 같은 형식으로 보고한다
        return False, f"Failed to read {path}: {e}"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return False, f"{path} missing columns: {missing}"
    return True, str(path)


def main() -> int:
    ok_all = True
    for rel_path, cols in REQUIRED:
        ok, msg = check_file(PROJECT_ROOT / rel_path, cols)
        print(("[OK] " if ok else "[FAIL] ") + msg)
        ok_all &= ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
