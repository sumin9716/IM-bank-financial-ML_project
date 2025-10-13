import pandas as pd

def compute_monthly_exposure(
    panel: pd.DataFrame,
    export_col: str = "export_amt",
    import_col: str = "import_amt",
    date_col: str = "month",
    company_col: str = "company_id",
) -> pd.DataFrame:
    df = panel.copy()
    for c in (export_col, import_col):
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[date_col] = pd.to_datetime(df[date_col])
    out = (df.groupby([company_col, date_col], as_index=False)
             .agg(export_amt=(export_col, "sum"),
                  import_amt=(import_col, "sum")))
    out["net_exposure"] = out["export_amt"] - out["import_amt"]
    return out
