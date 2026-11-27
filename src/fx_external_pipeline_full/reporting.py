import pandas as pd
from pathlib import Path

def save_csv(df: pd.DataFrame, path: str):
    """BOM 없는 UTF-8 CSV로 저장한다(BI 도구가 첫 열 이름에 BOM을 붙여 읽지 않도록)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
