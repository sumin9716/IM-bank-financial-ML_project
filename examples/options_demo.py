
"""
Run demo:
    export FRED_API_KEY=YOUR_FRED_KEY
    python examples/options_demo.py
"""
import os, math, pandas as pd
from src.data_loaders import load_usdkrw_spot, load_us_rates
from src.pricing.forwards import price_forward_cip, CIPConfig
from src.pricing.options import GKInputs, garman_kohlhagen, realized_vol_from_spot

def main():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY to run this demo.")
        return

    # 1) Load market data
    spot = load_usdkrw_spot(source="FRED", start="2024-01-01")      # DEXKOUS
    us   = load_us_rates(["DGS1"], start="2024-01-01")              # placeholder for USD rate
    kr   = load_us_rates(["DGS1"], start="2024-01-01")              # placeholder for KRW rate (swap to ECOS later)

    # 2) Compute a 3M tenor forward fair value (to define ATM strike)
    cfg = CIPConfig(method="cont")
    fwd = price_forward_cip(spot, kr, us, tenor="3M", cfg=cfg, kr_rate_series="DGS1", us_rate_series="DGS1")
    latest = fwd.dropna().iloc[-1]
    S = float(latest["spot"])
    K_atm = float(latest["fair_fwd"])  # ATM-forward strike

    # 3) Estimate realized vol (63d window, annualized)
    rv = realized_vol_from_spot(spot, window=63)
    sigma = float(rv.dropna().iloc[-1]["rv_63d"])

    # 4) Price ATM call/put (3M)
    T = float(latest["T_dom"])  # same year fraction
    r_dom = float(latest["r_dom"])/100.0
    r_for = float(latest["r_for"])/100.0

    call_in = GKInputs(S=S, K=K_atm, T=T, sigma=sigma, r_dom=r_dom, r_for=r_for, opt_type="call")
    put_in  = GKInputs(S=S, K=K_atm, T=T, sigma=sigma, r_dom=r_dom, r_for=r_for, opt_type="put")

    call = garman_kohlhagen(call_in)
    put  = garman_kohlhagen(put_in)

    print("ATM-forward Call:", call)
    print("ATM-forward Put :", put)
    print("Parity check (C - P ≈ df*(F-K)):", call.price - put.price)

if __name__ == "__main__":
    main()
