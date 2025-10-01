
"""
Run demo:
    export FRED_API_KEY=YOUR_FRED_KEY
    python examples/hedge_demo.py
"""
import os
from src.data_loaders import load_usdkrw_spot, load_us_rates
from src.hedge.rules import compute_market_features, recommend_hedge_ratio

def main():
    if not os.getenv("FRED_API_KEY"):
        print("Set FRED_API_KEY to run this demo.")
        return

    spot = load_usdkrw_spot(source="FRED", start="2024-01-01")
    us   = load_us_rates(["DGS1"], start="2024-01-01")
    kr   = load_us_rates(["DGS1"], start="2024-01-01")  # placeholder for KR side (swap to ECOS CALL/CD later)

    feats = compute_market_features(spot, kr_rate_df=kr, us_rate_df=us, kr_series="DGS1", us_series="DGS1")
    reco_std = recommend_hedge_ratio(feats, risk="standard")
    reco_con = recommend_hedge_ratio(feats, risk="conservative")
    reco_agg = recommend_hedge_ratio(feats, risk="aggressive")

    print("Standard policy tail:")
    print(reco_std.tail(5))
    print("Counts:", reco_std["hedge_ratio"].value_counts())

    print("\nConservative policy tail:")
    print(reco_con.tail(5))
    print("Counts:", reco_con["hedge_ratio"].value_counts())

    print("\nAggressive policy tail:")
    print(reco_agg.tail(5))
    print("Counts:", reco_agg["hedge_ratio"].value_counts())

if __name__ == "__main__":
    main()
