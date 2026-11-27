"""정책 설정과 IFRS9 효과성 결과를 시점별 스냅샷과 변경 이력 CSV로 남긴다."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _append_row(csv_path: str, row: dict) -> str:
    """CSV가 없으면 머리글과 함께 만들고, 있으면 한 줄을 덧붙인다."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if Path(csv_path).exists():
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(csv_path)


def snapshot_policy(cfg: dict, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"policy_snapshot_{_utc_now():%Y%m%dT%H%M%SZ}.yml"
    snap = {"policy": cfg.get("policy", {}), "meta": cfg.get("meta", {})}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)
    return str(path)


def append_changelog(changelog_csv: str, action: str, actor: str, note: str, cfg: dict) -> str:
    return _append_row(changelog_csv, {
        "timestamp_utc": f"{_utc_now():%Y-%m-%dT%H:%M:%SZ}",
        "action": action,
        "actor": actor or "system",
        "note": note or "",
        "policy_version": (cfg.get("policy") or {}).get("version", ""),
        "profile": (cfg.get("active_profile") or ""),
    })


def snapshot_effectiveness(df: pd.DataFrame, out_dir: str, meta: dict | None = None) -> str:
    """효과성 결과는 표 형태라 CSV로, 판정 기준(meta)은 같은 이름의 YAML로 저장한다."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = f"{_utc_now():%Y%m%dT%H%M%SZ}"
    csv_path = Path(out_dir) / f"ifrs9_effectiveness_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if meta:
        with open(Path(out_dir) / f"ifrs9_effectiveness_{ts}.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    return str(csv_path)


def append_effectiveness_changelog(csv_path: str, action: str, actor: str, note: str, params: dict) -> str:
    return _append_row(csv_path, {
        "timestamp_utc": f"{_utc_now():%Y-%m-%dT%H:%M:%SZ}",
        "action": action,
        "actor": actor or "system",
        "note": note or "",
        "params": str(params or {}),
    })
