
"""
Run demo:
    export FRED_API_KEY=YOUR_FRED_KEY
    python examples/ndf_demo.py
"""
import os
import pandas as pd
from src.data_loaders import load_usdkrw_spot, load_us_rates
from src.pricing.forwards import price_forward_cip, CIPConfig
from src.ndf.simulator import make_trades_from_fair_values, simulate_ndf_pnl

def main():
    if not os.getenv("FRED_API_KEY"):
        print("Set FRED_API_KEY to run this demo.")
        return

    # 1) Market data
    spot = load_usdkrw_spot(source="FRED", start="2024-01-01")
    us   = load_us_rates(["DGS1"], start="2024-01-01")
    kr   = load_us_rates(["DGS1"], start="2024-01-01")  # placeholder (swap to ECOS later)

    # 2) CIP fair forwards (3M)
    cfg = CIPConfig(method="cont")
    fair = price_forward_cip(spot, kr, us, tenor="3M", cfg=cfg, kr_rate_series="DGS1", us_rate_series="DGS1")

    # 3) Create trades at each available trade date with notional USD 1M
    trades = make_trades_from_fair_values(fair, tenor="3M", notional_usd=1_000_000.0, direction="long_usd")

    # 4) Simulate NDF PnL with fixing picked by 'preceding'
    pnl = simulate_ndf_pnl(trades, spot, fixing_pick="preceding")
    print(pnl.tail(5))

    # 5) Summary
    print("Total PnL (KRW):", pnl["pnl_krw"].sum())
    print("Mean / Std (KRW):", pnl["pnl_krw"].mean(), pnl["pnl_krw"].std())

if __name__ == "__main__":
    main()
