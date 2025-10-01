
"""
Run demo:
    export FRED_API_KEY=YOUR_FRED_KEY
    python examples/cip_demo.py
"""
import os
import pandas as pd
from src.data_loaders import load_usdkrw_spot, load_us_rates
from src.pricing.forwards import price_forward_cip, CIPConfig

def main():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY to run this demo.")
        return

    spot = load_usdkrw_spot(source="FRED", start="2024-01-01")  # DEXKOUS
    # For demo, use KRW ~ CD91 via ECOS would be better, but we keep US side only.
    # We'll approximate KRW rate as DGS1 too (for PoC). In practice, swap to ECOS CALL/CD.
    us = load_us_rates(["DGS1"], start="2024-01-01")
    kr = load_us_rates(["DGS1"], start="2024-01-01")  # placeholder for KR side

    # pick latest available daily points
    cfg = CIPConfig(method="cont")  # continuous-compounding

    out_1M = price_forward_cip(spot, kr, us, tenor="1M", cfg=cfg, kr_rate_series="DGS1", us_rate_series="DGS1")
    out_3M = price_forward_cip(spot, kr, us, tenor="3M", cfg=cfg, kr_rate_series="DGS1", us_rate_series="DGS1")

    print(out_1M.tail(3))
    print(out_3M.tail(3))

if __name__ == "__main__":
    main()
