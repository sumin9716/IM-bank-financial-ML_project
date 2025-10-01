from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from src.data_loaders import load_usdkrw_spot, load_us_rates, load_holidays
from src.pricing.forwards import price_forward_cip, CIPConfig
from src.pricing.options import GKInputs, garman_kohlhagen, realized_vol_from_spot
from src.risk.metrics import hist_var_es, worst_events, rolling_var_series
from src.ndf.simulator import make_trades_from_fair_values, simulate_ndf_pnl
from src.hedge.rules import compute_market_features, recommend_hedge_ratio

DEFAULT_START = date(2023, 1, 1)


@st.cache_data(show_spinner=False)
def cache_spot_fred(api_key: str, start: str, end: Optional[str]) -> pd.DataFrame:
    return load_usdkrw_spot(source="FRED", start=start, end=end, fred_api_key=api_key)


@st.cache_data(show_spinner=False)
def cache_rates_fred(api_key: str, tenor: str, start: str, end: Optional[str]) -> pd.DataFrame:
    return load_us_rates(tenors=[tenor], start=start, end=end, fred_api_key=api_key)


@st.cache_data(show_spinner=False)
def cache_holidays(country: str, years: Tuple[int, ...]) -> pd.DataFrame:
    return load_holidays(country=country, years=list(years))


def parse_uploaded_csv(file, required: Tuple[str, ...] = ("date", "value")) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(missing)}")
        df = df.dropna(subset=list(required))
    return df


def sort_by_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        return df.sort_values(col).reset_index(drop=True)
    return df


def load_spot_data(source: str, fred_key: str, start: date, end: Optional[date], spot_file) -> pd.DataFrame:
    if source == "FRED (DEXKOUS)":
        if not fred_key:
            raise ValueError("FRED API key is required for FRED spot.")
        return sort_by_date(cache_spot_fred(fred_key, start.isoformat(), end.isoformat() if end else None))
    if spot_file is None:
        raise ValueError("Upload a spot CSV first.")
    return sort_by_date(parse_uploaded_csv(spot_file, ("date", "value")))


def load_rate_data(tenor_key: str, fred_key: str, start: date, end: Optional[date], upload_file, label: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if upload_file is not None:
        df = sort_by_date(parse_uploaded_csv(upload_file, ("date", "value")))
        series_name: Optional[str] = None
        if "series" in df.columns and df["series"].notna().any():
            unique_vals = df["series"].dropna().unique()
            series_name = tenor_key if len(unique_vals) != 1 else str(unique_vals[0])
        return df, series_name
    if not tenor_key:
        raise ValueError(f"Provide a tenor key for {label}.")
    if not fred_key:
        raise ValueError(f"FRED API key is required for {label}.")
    df = sort_by_date(cache_rates_fred(fred_key, tenor_key, start.isoformat(), end.isoformat() if end else None))
    return df, tenor_key


def gather_market_data(
    spot_source: str,
    spot_file,
    usd_tenor: str,
    usd_file,
    use_proxy: bool,
    kr_tenor: str,
    kr_file,
    fred_key: str,
    start: date,
    end: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str], pd.DataFrame, Optional[str]]:
    spot_df = load_spot_data(spot_source, fred_key, start, end, spot_file)
    us_df, us_series = load_rate_data(usd_tenor.strip(), fred_key, start, end, usd_file, "USD rate")
    if use_proxy:
        return spot_df, us_df.copy(), us_series, us_df, us_series
    kr_df, kr_series = load_rate_data(kr_tenor.strip(), fred_key, start, end, kr_file, "KR rate")
    return spot_df, kr_df, kr_series, us_df, us_series


def render_overview():
    st.title("FX Analytics Workbench")
    st.write("This Streamlit app wraps the repository's demos into a single UI.")
    st.markdown(
        "- Fill the FRED API key in the sidebar to fetch live data.\n"
        "- Alternatively upload CSV files with columns `date` and `value`.\n"
        "- Each module mirrors the scripts under `examples/`."
    )
    st.markdown("**Modules Covered**")
    st.markdown(
        "- CIP forward fair values\n"
        "- FX options (Garman?Kohlhagen)\n"
        "- Historical VaR / Expected Shortfall\n"
        "- NDF PnL simulation\n"
        "- Rule-based hedge ratio\n"
        "- Raw data loader checks"
    )
    st.info("Use the sidebar to switch modules and set the default start date.")


def render_data_loaders(fred_key: str, default_start: date):
    st.header("Data Loaders")
    st.caption("Pull or inspect raw inputs before running pricing modules.")
    tab_spot, tab_rates, tab_holidays = st.tabs(["USD/KRW Spot", "US Rates", "Holidays"])

    with tab_spot:
        start = st.date_input("Start date", value=default_start, key="dl_spot_start")
        use_end = st.checkbox("Use end date", value=False, key="dl_spot_use_end")
        end_val = st.date_input("End date", value=date.today(), key="dl_spot_end")
        end = end_val if use_end else None
        if st.button("Fetch spot", key="dl_fetch_spot"):
            if not fred_key:
                st.error("Enter a FRED API key in the sidebar first.")
            else:
                try:
                    df = cache_spot_fred(fred_key, start.isoformat(), end.isoformat() if end else None)
                    st.success(f"Loaded {len(df)} rows.")
                    st.dataframe(df.tail(10))
                except Exception as exc:
                    st.error(f"Spot fetch failed: {exc}")

    with tab_rates:
        start = st.date_input("Start date", value=default_start, key="dl_rates_start")
        use_end = st.checkbox("Use end date", value=False, key="dl_rates_use_end")
        end_val = st.date_input("End date", value=date.today(), key="dl_rates_end")
        tenor = st.text_input("Tenor key (fred_series.yml)", value="SOFR", key="dl_rates_tenor")
        if st.button("Fetch rates", key="dl_fetch_rates"):
            if not fred_key:
                st.error("Enter a FRED API key in the sidebar first.")
            elif not tenor.strip():
                st.error("Enter a tenor key (e.g. SOFR, DGS1).")
            else:
                try:
                    df = cache_rates_fred(fred_key, tenor.strip(), start.isoformat(), end_val.isoformat() if use_end else None)
                    st.success(f"Loaded {len(df)} rows.")
                    st.dataframe(df.tail(10))
                except Exception as exc:
                    st.error(f"Rate fetch failed: {exc}")

    with tab_holidays:
        years = st.multiselect(
            "Years",
            [date.today().year - 1, date.today().year, date.today().year + 1],
            default=[date.today().year],
            key="dl_holiday_years",
        )
        if st.button("Fetch holidays", key="dl_fetch_holidays"):
            if not years:
                st.error("Pick at least one year.")
            else:
                try:
                    df = cache_holidays("KR", tuple(int(y) for y in years))
                    st.success(f"Loaded {len(df)} rows.")
                    st.dataframe(df)
                except Exception as exc:
                    st.error(f"Holiday fetch failed: {exc}")


def render_cip(fred_key: str, default_start: date):
    st.header("CIP Forward Fair Value")
    st.caption("Reproduces `examples/cip_demo.py`." )
    spot_source = st.radio("Spot data source", ["FRED (DEXKOUS)", "Upload CSV"], horizontal=True, key="cip_spot_source")
    spot_file = None if spot_source == "FRED (DEXKOUS)" else st.file_uploader("Spot CSV (columns: date,value)", type="csv", key="cip_spot_file")
    start = st.date_input("Spot/rate start date", value=default_start, key="cip_start")
    tenor = st.selectbox("Forward tenor", ["1M", "3M", "6M", "1Y"], index=1, key="cip_tenor")
    method = st.selectbox("Pricing method", ["cont", "simple"], index=0, key="cip_method")
    biz_conv = st.selectbox("Business day convention", ["following", "modified_following", "preceding"], key="cip_biz")
    dcc_dom = st.selectbox("Domestic day count", ["act/365", "act/360"], index=0, key="cip_dcc_dom")
    dcc_for = st.selectbox("Foreign day count", ["act/360", "act/365"], index=0, key="cip_dcc_for")
    usd_tenor = st.text_input("USD rate tenor key (fred_series.yml)", value="DGS1", key="cip_us_tenor")
    usd_file = st.file_uploader("Optional USD rate CSV override", type="csv", key="cip_us_file")
    use_proxy = st.checkbox("Use USD rate as KR proxy", value=True, key="cip_proxy")
    kr_tenor = ""
    kr_file = None
    if not use_proxy:
        kr_tenor = st.text_input("KR rate tenor key (leave blank if CSV supplies series)", value="CD91", key="cip_kr_tenor")
        kr_file = st.file_uploader("Optional KR rate CSV", type="csv", key="cip_kr_file")
    use_holidays = st.checkbox("Adjust for KR public holidays", value=False, key="cip_holidays_toggle")
    holiday_years: Tuple[int, ...] = ()
    if use_holidays:
        current_year = date.today().year
        selected_years = st.multiselect(
            "Holiday years",
            [current_year - 1, current_year, current_year + 1],
            default=[current_year, current_year + 1],
            key="cip_holiday_years",
        )
        holiday_years = tuple(int(y) for y in selected_years)
    run_btn = st.button("Compute CIP forwards", key="cip_run")
    if not run_btn:
        return
    try:
        spot_df, kr_df, kr_series, us_df, us_series = gather_market_data(
            spot_source, spot_file, usd_tenor, usd_file, use_proxy, kr_tenor, kr_file, fred_key, start
        )
        holidays_df = cache_holidays("KR", holiday_years) if use_holidays and holiday_years else None
        cfg = CIPConfig(method=method, dcc_dom=dcc_dom, dcc_for=dcc_for, biz_conv=biz_conv, holidays=holidays_df)
        fair_df = price_forward_cip(
            spot_df, kr_df, us_df, tenor=tenor, cfg=cfg, kr_rate_series=kr_series, us_rate_series=us_series
        )
        if fair_df.empty:
            st.warning("No forward rows produced. Check that your datasets overlap.")
            return
        st.success(f"Computed {len(fair_df)} rows for tenor {tenor}.")
        st.dataframe(fair_df.tail(10))
        chart_df = fair_df[["date", "fair_fwd"]].dropna().set_index("date")
        if not chart_df.empty:
            st.line_chart(chart_df, height=320)
        st.session_state["cip_latest"] = {
            "fair_df": fair_df,
            "spot_df": spot_df,
            "kr_df": kr_df,
            "kr_series": kr_series,
            "us_df": us_df,
            "us_series": us_series,
            "cfg": cfg,
            "tenor": tenor,
        }
    except Exception as exc:
        st.error(f"CIP calculation failed: {exc}")


def render_options(fred_key: str, default_start: date):
    st.header("FX Options (Garman?Kohlhagen)")
    st.caption("Reproduces `examples/options_demo.py`.")
    spot_source = st.radio("Spot data source", ["FRED (DEXKOUS)", "Upload CSV"], horizontal=True, key="opt_spot_source")
    spot_file = None if spot_source == "FRED (DEXKOUS)" else st.file_uploader("Spot CSV (columns: date,value)", type="csv", key="opt_spot_file")
    start = st.date_input("Data start date", value=default_start, key="opt_start")
    tenor = st.selectbox("Forward tenor", ["1M", "3M", "6M", "1Y"], index=1, key="opt_tenor")
    usd_tenor = st.text_input("USD rate tenor key", value="DGS1", key="opt_us_tenor")
    usd_file = st.file_uploader("Optional USD rate CSV override", type="csv", key="opt_us_file")
    use_proxy = st.checkbox("Use USD rate as KR proxy", value=True, key="opt_proxy")
    kr_tenor = ""
    kr_file = None
    if not use_proxy:
        kr_tenor = st.text_input("KR rate tenor key", value="CD91", key="opt_kr_tenor")
        kr_file = st.file_uploader("Optional KR rate CSV", type="csv", key="opt_kr_file")
    vol_window = st.number_input("Realized vol window (trading days)", min_value=20, max_value=252, value=63, step=1, key="opt_vol_window")
    strike_mode = st.selectbox("Strike selection", ["ATM-forward", "Custom"], key="opt_strike_mode")
    custom_strike = st.number_input("Custom strike (KRW per USD)", min_value=0.0, value=1300.0, key="opt_custom_strike") if strike_mode == "Custom" else None
    run_btn = st.button("Price option", key="opt_run")
    if not run_btn:
        return
    try:
        spot_df, kr_df, kr_series, us_df, us_series = gather_market_data(
            spot_source, spot_file, usd_tenor, usd_file, use_proxy, kr_tenor, kr_file, fred_key, start
        )
        cfg = CIPConfig(method="cont")
        fair_df = price_forward_cip(
            spot_df, kr_df, us_df, tenor=tenor, cfg=cfg, kr_rate_series=kr_series, us_rate_series=us_series
        )
        fair_df = fair_df.dropna(subset=["fair_fwd", "spot", "r_dom", "r_for", "T_dom"])
        if fair_df.empty:
            st.warning("Forward curve is empty. Ensure spot and rates overlap.")
            return
        latest = fair_df.iloc[-1]
        S = float(latest["spot"])
        K = float(latest["fair_fwd"]) if strike_mode == "ATM-forward" else float(custom_strike)
        T = float(latest["T_dom"])
        r_dom = float(latest["r_dom"]) / 100.0
        r_for = float(latest["r_for"]) / 100.0
        rv_df = realized_vol_from_spot(spot_df, window=int(vol_window))
        rv_df = rv_df.dropna()
        if rv_df.empty:
            st.warning("Realized volatility series is empty. Try a smaller window or more history.")
            return
        sigma = float(rv_df.iloc[-1][f"rv_{int(vol_window)}d"])
        call_input = GKInputs(S=S, K=K, T=T, sigma=sigma, r_dom=r_dom, r_for=r_for, opt_type="call")
        put_input = GKInputs(S=S, K=K, T=T, sigma=sigma, r_dom=r_dom, r_for=r_for, opt_type="put")
        call = garman_kohlhagen(call_input)
        put = garman_kohlhagen(put_input)
        res_df = pd.DataFrame(
            {
                "price": [call.price, put.price],
                "d1": [call.d1, put.d1],
                "d2": [call.d2, put.d2],
                "delta": [call.delta, put.delta],
                "gamma": [call.gamma, put.gamma],
                "vega": [call.vega, put.vega],
                "theta": [call.theta, put.theta],
                "rho_dom": [call.rho_dom, put.rho_dom],
                "rho_for": [call.rho_for, put.rho_for],
            },
            index=["Call", "Put"],
        )
        st.success(f"Priced option with sigma={sigma:.2%} and tenor {tenor}.")
        st.dataframe(res_df)
        st.metric("Call - Put parity", value=f"{call.price - put.price:.4f}")
    except Exception as exc:
        st.error(f"Option pricing failed: {exc}")


def render_var(fred_key: str, default_start: date):
    st.header("Historical VaR / ES")
    st.caption("Reproduces `examples/var_demo.py`.")
    spot_source = st.radio("Spot data source", ["FRED (DEXKOUS)", "Upload CSV"], horizontal=True, key="var_spot_source")
    spot_file = None if spot_source == "FRED (DEXKOUS)" else st.file_uploader("Spot CSV (columns: date,value)", type="csv", key="var_spot_file")
    start = st.date_input("Start date", value=default_start, key="var_start")
    horizon = st.number_input("Holding period (days)", min_value=1, max_value=60, value=1, step=1, key="var_horizon")
    alpha = st.slider("Confidence level", min_value=0.90, max_value=0.995, value=0.99, step=0.005, key="var_alpha")
    notional = st.number_input("USD notional", min_value=1_000.0, value=1_000_000.0, step=1000.0, key="var_notional")
    ret_method = st.selectbox("Return type", ["log", "simple"], index=0, key="var_ret")
    run_btn = st.button("Compute VaR / ES", key="var_run")
    if not run_btn:
        return
    try:
        spot_df = load_spot_data(spot_source, fred_key, start, None, spot_file)
        res = hist_var_es(
            spot_df,
            horizon_days=int(horizon),
            alpha=float(alpha),
            ret=ret_method,
            notional_usd=float(notional),
        )
        st.success(f"Computed VaR/ES over {res['n_obs']} observations.")
        st.metric("VaR (KRW loss)", f"{res['VaR']:,.0f}")
        st.metric("ES (KRW loss)", f"{res['ES']:,.0f}")
        st.json({k: float(res[k]) for k in ["ret_mean", "ret_std", "ret_skew", "ret_kurt"]})
        worst = worst_events(res["loss_series"])
        st.subheader("Worst loss events")
        st.dataframe(worst)
        roll = rolling_var_series(
            spot_df,
            window=252,
            horizon_days=int(horizon),
            alpha=float(alpha),
            ret=ret_method,
            notional_usd=float(notional),
        )
        if not roll.empty:
            st.subheader("Rolling VaR")
            st.line_chart(roll.set_index("date"), height=320)
    except Exception as exc:
        st.error(f"VaR calculation failed: {exc}")


def render_ndf(fred_key: str, default_start: date):
    st.header("NDF PnL Simulator")
    st.caption("Reproduces `examples/ndf_demo.py`.")
    spot_source = st.radio("Spot data source", ["FRED (DEXKOUS)", "Upload CSV"], horizontal=True, key="ndf_spot_source")
    spot_file = None if spot_source == "FRED (DEXKOUS)" else st.file_uploader("Spot CSV (columns: date,value)", type="csv", key="ndf_spot_file")
    start = st.date_input("Start date", value=default_start, key="ndf_start")
    tenor = st.selectbox("Tenor to trade", ["1M", "3M", "6M"], index=1, key="ndf_tenor")
    usd_tenor = st.text_input("USD rate tenor key", value="DGS1", key="ndf_us_tenor")
    usd_file = st.file_uploader("Optional USD rate CSV override", type="csv", key="ndf_us_file")
    use_proxy = st.checkbox("Use USD rate as KR proxy", value=True, key="ndf_proxy")
    kr_tenor = ""
    kr_file = None
    if not use_proxy:
        kr_tenor = st.text_input("KR rate tenor key", value="CD91", key="ndf_kr_tenor")
        kr_file = st.file_uploader("Optional KR rate CSV", type="csv", key="ndf_kr_file")
    notional = st.number_input("Trade notional (USD)", min_value=10_000.0, value=1_000_000.0, step=10_000.0, key="ndf_notional")
    direction = st.selectbox("Direction", ["long_usd", "short_usd"], key="ndf_direction")
    trade_depth = st.slider("Number of trades to simulate", min_value=5, max_value=120, value=24, step=1, key="ndf_depth")
    fixing_pick = st.selectbox("Fixing rule", ["preceding", "following"], key="ndf_fixing")
    run_btn = st.button("Simulate NDF PnL", key="ndf_run")
    if not run_btn:
        return
    try:
        spot_df, kr_df, kr_series, us_df, us_series = gather_market_data(
            spot_source, spot_file, usd_tenor, usd_file, use_proxy, kr_tenor, kr_file, fred_key, start
        )
        cfg = CIPConfig(method="cont")
        fair_df = price_forward_cip(
            spot_df, kr_df, us_df, tenor=tenor, cfg=cfg, kr_rate_series=kr_series, us_rate_series=us_series
        )
        fair_df = fair_df.dropna(subset=["fair_fwd", "maturity"])
        if fair_df.empty:
            st.warning("Forward data is empty. Cannot build trades.")
            return
        trades = make_trades_from_fair_values(fair_df, tenor=tenor, notional_usd=float(notional), direction=direction)
        if trades.empty:
            st.warning("No trades generated from forward curve.")
            return
        trades = trades.tail(int(trade_depth)).reset_index(drop=True)
        pnl_df = simulate_ndf_pnl(trades, spot_df, fixing_pick=fixing_pick)
        st.success(f"Simulated {len(pnl_df)} trades.")
        st.dataframe(pnl_df)
        if "pnl_krw" in pnl_df.columns:
            chart = pnl_df[["maturity", "pnl_krw"]].set_index("maturity")
            st.line_chart(chart, height=320)
    except Exception as exc:
        st.error(f"NDF simulation failed: {exc}")


def render_hedge(fred_key: str, default_start: date):
    st.header("Rule-based Hedge Ratio")
    st.caption("Reproduces `examples/hedge_demo.py`.")
    spot_source = st.radio("Spot data source", ["FRED (DEXKOUS)", "Upload CSV"], horizontal=True, key="hedge_spot_source")
    spot_file = None if spot_source == "FRED (DEXKOUS)" else st.file_uploader("Spot CSV (columns: date,value)", type="csv", key="hedge_spot_file")
    start = st.date_input("Start date", value=default_start, key="hedge_start")
    usd_tenor = st.text_input("USD rate tenor key", value="DGS1", key="hedge_us_tenor")
    usd_file = st.file_uploader("Optional USD rate CSV override", type="csv", key="hedge_us_file")
    use_proxy = st.checkbox("Use USD rate as KR proxy", value=True, key="hedge_proxy")
    kr_tenor = ""
    kr_file = None
    if not use_proxy:
        kr_tenor = st.text_input("KR rate tenor key", value="CD91", key="hedge_kr_tenor")
        kr_file = st.file_uploader("Optional KR rate CSV", type="csv", key="hedge_kr_file")
    risk = st.selectbox("Risk appetite", ["conservative", "standard", "aggressive"], index=1, key="hedge_risk")
    tail_rows = st.slider("Rows to display", min_value=5, max_value=120, value=30, step=5, key="hedge_rows")
    run_btn = st.button("Compute hedge ratio recommendations", key="hedge_run")
    if not run_btn:
        return
    try:
        spot_df, kr_df, kr_series, us_df, us_series = gather_market_data(
            spot_source, spot_file, usd_tenor, usd_file, use_proxy, kr_tenor, kr_file, fred_key, start
        )
        features = compute_market_features(spot_df, kr_rate_df=kr_df, us_rate_df=us_df, kr_series=kr_series, us_series=us_series)
        features = features.dropna(subset=["rv_20d", "rv_60d", "carry"], how="all")
        if features.empty:
            st.warning("Feature set is empty. Provide more data.")
            return
        reco = recommend_hedge_ratio(features, risk=risk)
        st.success("Computed hedge ratio recommendations.")
        st.dataframe(reco.tail(int(tail_rows)))
        trend_chart = features[["date", "rv_20d", "rv_60d"]].dropna().set_index("date")
        if not trend_chart.empty:
            st.subheader("Realized volatility (annualized)")
            st.line_chart(trend_chart, height=320)
    except Exception as exc:
        st.error(f"Hedge recommendation failed: {exc}")


def main():
    st.set_page_config(page_title="FX Analytics Workbench", layout="wide")
    st.sidebar.title("Settings")
    fred_key = st.sidebar.text_input("FRED API Key", type="password")
    default_start = st.sidebar.date_input("Default start date", value=DEFAULT_START)
    page = st.sidebar.selectbox(
        "Module",
        [
            "Overview",
            "Data Loaders",
            "CIP Forwards",
            "FX Options",
            "VaR / ES",
            "NDF Simulator",
            "Hedge Rules",
        ],
    )

    if page == "Overview":
        render_overview()
    elif page == "Data Loaders":
        render_data_loaders(fred_key, default_start)
    elif page == "CIP Forwards":
        render_cip(fred_key, default_start)
    elif page == "FX Options":
        render_options(fred_key, default_start)
    elif page == "VaR / ES":
        render_var(fred_key, default_start)
    elif page == "NDF Simulator":
        render_ndf(fred_key, default_start)
    elif page == "Hedge Rules":
        render_hedge(fred_key, default_start)


if __name__ == "__main__":
    main()
