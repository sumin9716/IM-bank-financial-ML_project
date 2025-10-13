import json, yaml, pandas as pd
from pathlib import Path
from datetime import datetime

def snapshot_policy(cfg: dict, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) / f"policy_snapshot_{ts}.yml"
    snap = {"policy": cfg.get("policy", {}), "meta": cfg.get("meta", {})}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)
    return str(path)

def append_changelog(changelog_csv: str, action: str, actor: str, note: str, cfg: dict):
    Path(changelog_csv).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "action": action,  # e.g., apply_profile, tune_commit, manual_change
        "actor": actor or "system",
        "note": note or "",
        "policy_version": (cfg.get("policy") or {}).get("version", ""),
        "profile": (cfg.get("active_profile") or ""),
    }
    df = pd.DataFrame([row])
    if Path(changelog_csv).exists():
        df.to_csv(changelog_csv, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(changelog_csv, index=False, encoding="utf-8-sig")
    return str(changelog_csv)
