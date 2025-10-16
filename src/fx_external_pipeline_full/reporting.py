import pandas as pd
from pathlib import Path

def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV without BOM for better compatibility with Excel/BI tools"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
