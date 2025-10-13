#!/usr/bin/env python
import pandas as pd, numpy as np
from pathlib import Path
panel_path="data/panel_base.csv"; out=Path("data/external"); out.mkdir(parents=True, exist_ok=True)
try:
    panel=pd.read_csv(panel_path, parse_dates=["month"])
    m=panel["month"].dropna()
    if len(m)>0:
        m_eom=(m.dt.to_period("M").dt.to_timestamp("M"))
        months=pd.Series(m_eom).drop_duplicates().sort_values().tolist()
    else:
        months=pd.period_range("2024-01","2024-12",freq="M").to_timestamp("M").tolist()
except Exception:
    months=pd.period_range("2024-01","2024-12",freq="M").to_timestamp("M").tolist()
rng=np.random.default_rng(42); n=len(months)
spot=1100 + rng.normal(0,30,size=n).cumsum()/5
kr=0.035 + rng.normal(0,0.001,size=n).cumsum()/10
us=0.045 + rng.normal(0,0.001,size=n).cumsum()/10
pd.DataFrame({"month":months,"spot":spot}).to_csv(out/"spot_usdkrw_eom.csv", index=False)
pd.DataFrame({"month":months,"rate":kr}).to_csv(out/"kr_rates_month.csv", index=False)
pd.DataFrame({"month":months,"rate":us}).to_csv(out/"us_rates_month.csv", index=False)
print("Templates written to", str(out))
