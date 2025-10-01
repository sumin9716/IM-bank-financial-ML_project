
"""
Run demo:
    export FRED_API_KEY=YOUR_FRED_KEY
    python examples/var_demo.py
"""
import os
from src.data_loaders import load_usdkrw_spot
from src.risk.metrics import hist_var_es, worst_events, rolling_var_series

def main():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY to run this demo.")
        return

    spot = load_usdkrw_spot(source="FRED", start="2015-01-01")
    res99 = hist_var_es(spot, horizon_days=1, alpha=0.99, notional_usd=1.0)
    res95 = hist_var_es(spot, horizon_days=5, alpha=0.95, notional_usd=1.0)

    print("1-day 99% VaR/ES (KRW) for 1 USD:", res99["VaR"], res99["ES"], "N=",res99["n_obs"])
    print("5-day 95% VaR/ES (KRW) for 1 USD:", res95["VaR"], res95["ES"], "N=",res95["n_obs"])

    worst = worst_events(res99["loss_series"], top_n=10)
    print("Worst 10 daily losses:\n", worst)

    roll = rolling_var_series(spot, window=252, horizon_days=1, alpha=0.99)
    print("Rolling VaR series sample:\n", roll.tail())

if __name__ == "__main__":
    main()
