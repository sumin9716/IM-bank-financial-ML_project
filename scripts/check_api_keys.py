#!/usr/bin/env python
"""FRED·ECOS API 키가 설정되어 있는지, 실제로 짧은 구간을 받아올 수 있는지 확인한다.

키 값은 출력하지 않고 설정 여부만 보여 준다.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fx_external_pipeline_full.ecos_loader import load_ecos_series  # noqa: E402
from fx_external_pipeline_full.fred_loader import load_usdkrw_spot_eom_from_fred  # noqa: E402


def _report_key(name: str) -> None:
    print(f"{name}: {'set' if os.getenv(name) else 'not set'}")


def main() -> int:
    _report_key("FRED_API_KEY")
    _report_key("ECOS_API_KEY")

    failed = False
    try:
        spot = load_usdkrw_spot_eom_from_fred("2024-01-01", "2024-03-31")
        print(f"[OK] FRED: {len(spot)} month-end values")
    except Exception as e:  # 네트워크·인증 오류를 모두 같은 형식으로 보고한다
        print(f"[FAIL] FRED: {e}")
        failed = True

    try:
        kr = load_ecos_series("722Y001", "M", "2024-01", "2024-03", "")
        print(f"[OK] ECOS: {len(kr)} month-end values")
    except Exception as e:
        print(f"[FAIL] ECOS: {e}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
