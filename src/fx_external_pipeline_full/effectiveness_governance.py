import yaml, pandas as pd
from pathlib import Path
from datetime import datetime

def snapshot_effectiveness(df: pd.DataFrame, out_dir: str, meta: dict|None=None) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # save CSV (tables are friendlier than YAML for data frames)
    csv_path = Path(out_dir) / f"ifrs9_effectiveness_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # meta yaml (parameters used)
    if meta:
        yml_path = Path(out_dir) / f"ifrs9_effectiveness_{ts}.yml"
        with open(yml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    return str(csv_path)

def append_effectiveness_changelog(csv_path: str, action: str, actor: str, note: str, params: dict):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "action": action,  # e.g., assess_quarterly, threshold_change
        "actor": actor or "system",
        "note": note or "",
        "params": str(params or {}),
    }
    df = pd.DataFrame([row])
    if Path(csv_path).exists():
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(csv_path)
