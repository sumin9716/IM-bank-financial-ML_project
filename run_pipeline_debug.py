#!/usr/bin/env python
import argparse, sys, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'src'))

from fx_external_pipeline_full.utils import load_yaml, setup_logging, ensure_dir
from fx_external_pipeline_full.external_loader import read_eom_series_csv
from fx_external_pipeline_full.exposure import compute_monthly_exposure
from fx_external_pipeline_full.policy import apply_policy
from fx_external_pipeline_full.pricing import cip_forward_theoretical, apply_spread
from fx_external_pipeline_full.backtest import monthly_forward_strategy
from fx_external_pipeline_full.reporting import save_csv

def parse_args():
    ap = argparse.ArgumentParser(description='ML_project pipeline runner - Debug Version')
    ap.add_argument('--config', default=str(BASE/'config'/'config.yml'))
    ap.add_argument('--panel', default=str(BASE/'data'/'panel_base.csv'))
    ap.add_argument('--spot', default=str(BASE/'data'/'external'/'spot_usdkrw_eom.csv'))
    ap.add_argument('--kr',   default=str(BASE/'data'/'external'/'kr_rates_month.csv'))
    ap.add_argument('--us',   default=str(BASE/'data'/'external'/'us_rates_month.csv'))
    ap.add_argument('--reports_dir', default=str(BASE/'reports'))
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--spread_bps', type=float, default=10.0)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    
    try:
        setup_logging(str(BASE/'config'/'logging.yml'))
    except:
        print("[WARN] Logging setup failed, continuing...")

    reports_dir = args.reports_dir
    ensure_dir(reports_dir)

    print("🚀 Starting ML_project pipeline...")
    
    # Panel & exposure
    print("📊 Loading panel data...")
    panel = pd.read_csv(args.panel, parse_dates=['month'])
    print(f"   Panel shape: {panel.shape}")
    print(f"   Panel columns: {list(panel.columns)}")
    
    print("💰 Computing exposure...")
    exposure_df = compute_monthly_exposure(panel)
    print(f"   Exposure shape: {exposure_df.shape}")
    print(f"   Exposure columns: {list(exposure_df.columns)}")
    
    print("📋 Applying policy...")
    exposure_df = apply_policy(exposure_df, cfg=cfg)
    print(f"   After policy shape: {exposure_df.shape}")
    save_csv(exposure_df, str(Path(reports_dir)/'exposure.csv'))
    print("✅ Exposure saved")

    # Market data
    print("📈 Loading market data...")
    spot_eom = read_eom_series_csv(args.spot, 'spot')
    kr_eom = read_eom_series_csv(args.kr, 'rate') 
    us_eom = read_eom_series_csv(args.us, 'rate')
    print(f"   Spot USD/KRW: {len(spot_eom)} points")
    print(f"   KR rates: {len(kr_eom)} points") 
    print(f"   US rates: {len(us_eom)} points")

    # Forward & backtest
    print("💱 Computing forward prices...")
    fwd_theo = cip_forward_theoretical(spot_eom, us_eom, kr_eom, days=args.days)
    fwd_adj  = apply_spread(fwd_theo, spread_bps=args.spread_bps, side='sell_usd')
    print(f"   Forward prices: {len(fwd_adj)} points")

    print("🔄 Running backtest...")
    pnl_df = monthly_forward_strategy(
        exposure_df, spot_eom, fwd_adj,
        countries=tuple((cfg.get('holidays') or {}).get('countries', ['KR'])),
        holiday_cache_dir=(cfg.get('holidays') or {}).get('cache_dir','data/reference/holidays_cache')
    )
    print(f"   PnL DataFrame shape: {pnl_df.shape}")
    print(f"   PnL DataFrame columns: {list(pnl_df.columns)}")
    print("   Sample PnL data:")
    print(pnl_df.head())
    
    save_csv(pnl_df, str(Path(reports_dir)/'pnl_by_trade.csv'))
    print("✅ PnL saved")

    # Company summary (only if company_id exists)
    if 'company_id' in pnl_df.columns:
        print("📊 Computing company summary...")
        comp = (pnl_df.groupby('company_id').agg(
            trades=('pnl_krw','count'),
            pnl_sum=('pnl_krw','sum'),
            pnl_mean=('pnl_krw','mean'),
            pnl_std=('pnl_krw','std')).reset_index())
        save_csv(comp, str(Path(reports_dir)/'summary_company.csv'))
        print("✅ Company summary saved")
    else:
        print("⚠️ No company_id column found in PnL data, skipping company summary")

    print("✅ 🎉 Pipeline completed successfully!")
    print(f"📁 Check results in: {reports_dir}")

if __name__ == '__main__':
    main()