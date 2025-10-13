import pandas as pd
from .utils import to_month_end

def read_eom_series_csv(path: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    date_col = "month" if "month" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise KeyError(f"No month/date column in {path}.")
    if value_col not in df.columns:
        value_col = "spot" if "spot" in df.columns else ("rate" if "rate" in df.columns else value_col)
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(df[date_col])[value_col].sort_index()
    return to_month_end(s).groupby(level=0).last()
