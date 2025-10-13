import logging, logging.config, yaml, numpy as np, pandas as pd
from pathlib import Path

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.config.dictConfig(cfg)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("M").to_timestamp("M")

def to_month_end(s: pd.Series) -> pd.Series:
    s = s.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s.index = month_end_index(s.index)
    return s

def encode_categorical(series: pd.Series) -> pd.Series:
    cats = {v: i for i, v in enumerate(sorted(series.dropna().astype(str).unique()))}
    return series.astype(str).map(cats).fillna(-1).astype(int)
