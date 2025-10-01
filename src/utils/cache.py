
from __future__ import annotations
import os, time
from typing import Optional
import pandas as pd

DEF_CACHE_DIR = os.getenv("DATA_LOADER_CACHE_DIR", ".cache")

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def cache_path(namespace: str, filename: str) -> str:
    path = os.path.join(DEF_CACHE_DIR, namespace, filename)
    _ensure_dir(path)
    return path

def write_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=False)

def read_csv_if_fresh(path: str, max_age_hours: int) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    if age_hours > max_age_hours:
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
