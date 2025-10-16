#!/usr/bin/env python
import argparse, sys, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'src'))

from fx_external_pipeline_full.utils import load_yaml, setup_logging, ensure_dir, encode_categorical
from fx_external_pipeline_full.external_loader import read_eom_series_csv
from fx_external_pipeline_full.exposure import compute_monthly_exposure
from fx_external_pipeline_full.policy import apply_policy
from fx_external_pipeline_full.pricing import cip_forward_theoretical, apply_spread
from fx_external_pipeline_full.backtest import monthly_forward_strategy
from fx_external_pipeline_full.risk import historical_var, expected_shortfall
from fx_external_pipeline_full.pricing_option import garman_kohlhagen_call_put
from fx_external_pipeline_full.hedge_effectiveness import dollar_offset_ratio, regression_r2, judge_effective, regression_stats, sample_series
from fx_external_pipeline_full.effectiveness_governance import snapshot_effectiveness, append_effectiveness_changelog
from fx_external_pipeline_full.clustering import run_clustering
from fx_external_pipeline_full.reporting import save_csv
from fx_external_pipeline_full.features import compute_features
from fx_external_pipeline_full.policy_tuner import grid_tune_policy
from fx_external_pipeline_full.governance import snapshot_policy, append_changelog

def parse_args():
    ap = argparse.ArgumentParser(description='ML_project pipeline runner')
    ap.add_argument('--config', default=str(BASE/'config'/'config.yml'))
    ap.add_argument('--panel', default=str(BASE/'data'/'panel_base.csv'))
    ap.add_argument('--spot', default=str(BASE/'data'/'external'/'spot_usdkrw_eom.csv'))
    ap.add_argument('--kr',   default=str(BASE/'data'/'external'/'kr_rates_month.csv'))
    ap.add_argument('--us',   default=str(BASE/'data'/'external'/'us_rates_month.csv'))
    ap.add_argument('--reports_dir', default=str(BASE/'reports'))
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--spread_bps', type=float, default=10.0)
    ap.add_argument('--clusters', type=int, default=4)
    ap.add_argument('--profile', choices=['baseline','conservative','aggressive'], help='override parameters via profile')
    ap.add_argument('--tune_policy', action='store_true', help='run grid search to tune policy thresholds')
    ap.add_argument('--freeze_policy', action='store_true', help='snapshot current policy to artifacts and log change')
    ap.add_argument('--changelog_note', default='', help='note to append to policy changelog')
    ap.add_argument('--actor', default=None, help='actor name for governance changelog')
    ap.add_argument('--freeze_ifrs9', action='store_true', help='snapshot IFRS9 effectiveness results and log change')
    ap.add_argument('--ifrs9_note', default='', help='note for IFRS9 effectiveness changelog')
    return ap.parse_args()


def _resample_ifrs9_series(spot_eom: pd.Series,
                           fwd_eom: pd.Series,
                           spot_daily: pd.Series|None,
                           holidays_cfg: dict,
                           sampling_cfg: dict):
    """Return dict {'M': (target, hedge), 'Q': (target, hedge)} according to sampling rules.
    target=spot, hedge=fwd (levels aligned).
    """
    from fx_external_pipeline_full.holiday_calendar import previous_business_day, is_business_day
    if not isinstance(spot_eom.index, pd.DatetimeIndex): spot_eom.index = pd.to_datetime(spot_eom.index)
    if not isinstance(fwd_eom.index, pd.DatetimeIndex):  fwd_eom.index  = pd.to_datetime(fwd_eom.index)
    # monthly is trivial
    out = {'M': (spot_eom.copy(), fwd_eom.copy())}

    # quarterly
    rule = (sampling_cfg or {}).get('rule','month_end')
    # quarter end dates derived from spot_eom
    q_ends = spot_eom.index.to_period('Q').to_timestamp('Q')
    q_ends = pd.Index(sorted(q_ends.unique()))
    if rule == 'prev_business_day' and spot_daily is not None and len(spot_daily)>0:
        if not isinstance(spot_daily.index, pd.DatetimeIndex): spot_daily.index = pd.to_datetime(spot_daily.index)
        countries = (holidays_cfg or {}).get('countries', ['KR'])
        cache_dir = (holidays_cfg or {}).get('cache_dir', 'data/reference/holidays_cache')
        tgt_vals = []
        for q in q_ends:
            d = previous_business_day(q, countries=countries, cache_dir=cache_dir)
            s = spot_daily[spot_daily.index <= d]
            tgt_vals.append(float(s.iloc[-1]) if len(s)>0 else (float(spot_eom.loc[q]) if q in spot_eom.index else float('nan')))
        target_q = pd.Series(tgt_vals, index=q_ends)
    elif rule == 'avg':
        target_q = spot_eom.groupby(spot_eom.index.to_period('Q')).mean()
        target_q.index = target_q.index.to_timestamp('Q')
    else:  # month_end (use quarter-end EOM)
        # ensure we have a value at quarter end (last EOM inside the quarter)
        # map each quarter to the last available EOM <= quarter end
        target_q = spot_eom.groupby(spot_eom.index.to_period('Q')).last()
        target_q.index = target_q.index.to_timestamp('Q')

    # Hedge (forward) at quarter end: last EOM of the quarter, or average if rule=avg
    if rule == 'avg':
        hedge_q = fwd_eom.groupby(fwd_eom.index.to_period('Q')).mean()
        hedge_q.index = hedge_q.index.to_timestamp('Q')
    else:
        hedge_q = fwd_eom.groupby(fwd_eom.index.to_period('Q')).last()
        hedge_q.index = hedge_q.index.to_timestamp('Q')

    out['Q'] = (target_q, hedge_q)
    return out


def _load_spot_rates_from_config(cfg, args):
    from fx_external_pipeline_full.external_loader import read_eom_series_csv
    
    ds = cfg.get('data_source', {}) or {}

    # Spot
    if ds.get('spot','csv') == 'fred':
        try:
            from fx_external_pipeline_full.fred_loader import load_usdkrw_spot_eom_from_fred
            fr = cfg.get('fred',{}); rng = (fr.get('date_range') or {})
            spot_eom = load_usdkrw_spot_eom_from_fred(rng.get('start') or None, rng.get('end') or None)
        except ImportError:
            print("[WARN] FRED loader not available. Using CSV.")
            spot_eom = read_eom_series_csv(args.spot, 'spot')
    else:
        spot_eom = read_eom_series_csv(args.spot, 'spot')

    # US rates
    choice_us = ds.get('rates_us','csv')
    if choice_us == 'fred':
        try:
            from fx_external_pipeline_full.fred_loader import load_us_1y_yield_eom_from_fred
            fr = cfg.get('fred',{}); rng = (fr.get('date_range') or {})
            us_eom = load_us_1y_yield_eom_from_fred(rng.get('start') or None, rng.get('end') or None)
        except ImportError:
            print("[WARN] FRED loader not available. Using CSV.")
            us_eom = read_eom_series_csv(args.us, 'rate')
    elif choice_us == 'ecos':
        try:
            from fx_external_pipeline_full.ecos_loader import load_ecos_series
            ec = cfg.get('ecos',{}); dr = ec.get('date_range') or {}
            it = (ec.get('items') or {}).get('us_rate') or {"stat_code":"722Y001","cycle":"M","item_code":""}
            us_eom = load_ecos_series(it['stat_code'], it['cycle'], dr.get('start') or "2018-01", dr.get('end') or "", it.get('item_code',""))/100.0
        except ImportError:
            print("[WARN] ECOS loader not available. Using CSV.")
            us_eom = read_eom_series_csv(args.us, 'rate')
    else:
        us_eom = read_eom_series_csv(args.us, 'rate')

    # KR rates
    choice_kr = ds.get('rates_kr','csv')
    if choice_kr == 'ecos':
        try:
            from fx_external_pipeline_full.ecos_loader import load_ecos_series
            ec = cfg.get('ecos',{}); dr = ec.get('date_range') or {}
            it = (ec.get('items') or {}).get('kr_policy_rate') or {"stat_code":"722Y001","cycle":"M","item_code":""}
            kr_eom = load_ecos_series(it['stat_code'], it['cycle'], dr.get('start') or "2018-01", dr.get('end') or "", it.get('item_code',""))/100.0
        except ImportError:
            print("[WARN] ECOS loader not available. Using CSV.")
            kr_eom = read_eom_series_csv(args.kr, 'rate')
    elif choice_kr == 'fred':
        try:
            from fx_external_pipeline_full.fred_loader import load_sofr_eom_from_fred
            fr = cfg.get('fred',{}); rng = (fr.get('date_range') or {})
            kr_eom = load_sofr_eom_from_fred(rng.get('start') or None, rng.get('end') or None)  # proxy
        except ImportError:
            print("[WARN] FRED loader not available. Using CSV.")
            kr_eom = read_eom_series_csv(args.kr, 'rate')
    else:
        kr_eom = read_eom_series_csv(args.kr, 'rate')

    return spot_eom, kr_eom, us_eom


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    setup_logging(str(BASE/'config'/'logging.yml'))

    # Apply profile overrides (deep merge)
    def deep_update(d, u):
        for k, v in (u or {}).items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    if args.profile:
        prof = (cfg.get('profiles') or {}).get(args.profile)
        if prof:
            cfg = deep_update(cfg, prof)
            cfg['active_profile'] = args.profile
            print(f"[INFO] Profile applied: {args.profile}")
        else:
            print(f"[WARN] Profile '{args.profile}' not found. Using base config.")

    reports_dir = args.reports_dir
    ensure_dir(reports_dir)

    # Panel & exposure
    panel = pd.read_csv(args.panel, parse_dates=['month'])
    exposure_df = compute_monthly_exposure(panel)

    # Market curves via config-driven loader
    spot_eom, kr_eom, us_eom = _load_spot_rates_from_config(cfg, args)
    
    # Try to load FRED daily spot for precise business-day fixing
    spot_daily = None
    try:
        ds = cfg.get('data_source', {}) or {}
        if ds.get('spot','csv') == 'fred':
            fr = cfg.get('fred',{}); rng = (fr.get('date_range') or {})
            from fx_external_pipeline_full.fred_loader import load_usdkrw_spot_daily_from_fred
            spot_daily = load_usdkrw_spot_daily_from_fred(rng.get('start') or None, rng.get('end') or None)
    except Exception as e:
        print(f"[WARN] Could not load FRED daily spot. Falling back to EOM. Reason: {e}")

    # Market data preparation for ML models
    market_df = pd.DataFrame({
        'spot': spot_eom,
        'us_rate': us_eom,
        'kr_rate': kr_eom
    })
    
    # Compute features for policy application
    print("[INFO] Computing features for policy application...")
    features_df = None
    try:
        features_df = compute_features(spot_eom, us_eom, kr_eom, cfg)
        print(f"[INFO] Features computed for {len(features_df)} time periods")
    except Exception as e:
        print(f"[WARN] Feature computation failed: {e}. Policy will use v0 fallback.")
    
    # ML Models Training and Application
    print("[INFO] Training ML models...")
    ml_models = {}
    try:
        from fx_external_pipeline_full.ml_models import train_ml_models
        ml_models = train_ml_models(exposure_df, market_df, cfg)
        
        if ml_models:
            print(f"[INFO] Successfully trained {len(ml_models)} ML models")
            
            # Log model performance
            for model_name, model_info in ml_models.items():
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    if 'test_r2' in metrics:
                        print(f"  {model_name}: R² = {metrics['test_r2']:.3f}")
                    
                    # Feature importance for hedge ratio predictor
                    if model_name == 'hedge_ratio_predictor' and 'feature_importance' in model_info:
                        print(f"  Top features: {list(model_info['feature_importance'].keys())[:3]}")
        else:
            print("[INFO] No ML models were trained")
            
    except Exception as e:
        print(f"[WARN] ML model training failed: {e}")
        ml_models = {}
    
    # Apply enhanced policy (with ML if available, features if computed)
    exposure_df = apply_policy(exposure_df, features_df=features_df, cfg=cfg, 
                              ml_models=ml_models, market_df=market_df)
    save_csv(exposure_df, str(Path(reports_dir)/'exposure.csv'))
    
    # Future exposure forecasting (if ML model available)
    if 'exposure_forecaster' in ml_models:
        try:
            forecaster = ml_models['exposure_forecaster']['model']
            future_exposure = forecaster.predict_future_exposure(exposure_df)
            save_csv(future_exposure, str(Path(reports_dir)/'future_exposure_forecast.csv'))
            print(f"[INFO] Future exposure forecast saved: {len(future_exposure)} companies")
        except Exception as e:
            print(f"[WARN] Future exposure forecasting failed: {e}")

    # Forward & backtest
    fwd_theo = cip_forward_theoretical(spot_eom, us_eom, kr_eom, days=args.days)
    fwd_adj  = apply_spread(fwd_theo, spread_bps=args.spread_bps, side='sell_usd')

    # Optional policy tuning
    if args.tune_policy:
        res = grid_tune_policy(panel, exposure_df, spot_daily, spot_eom, us_eom, kr_eom, fwd_theo, cfg)
        save_csv(res, str(Path(reports_dir)/'policy_tuning_results.csv'))
        print('[INFO] Policy tuning done. Top row is the best candidate.')

    pnl_df = monthly_forward_strategy(
        exposure_df, spot_eom, fwd_adj,
        countries=tuple((cfg.get('holidays') or {}).get('countries', ['KR'])),
        holiday_cache_dir=(cfg.get('holidays') or {}).get('cache_dir','data/reference/holidays_cache')
    )
    save_csv(pnl_df, str(Path(reports_dir)/'pnl_by_trade.csv'))

    # Company summary (only if PnL data exists and has company_id)
    if not pnl_df.empty and 'company_id' in pnl_df.columns:
        comp = (pnl_df.groupby('company_id').agg(
            trades=('pnl_krw','count'),
            pnl_sum=('pnl_krw','sum'),
            pnl_mean=('pnl_krw','mean'),
            pnl_std=('pnl_krw','std')).reset_index())
        save_csv(comp, str(Path(reports_dir)/'summary_company.csv'))
        print(f"[INFO] Company summary saved: {len(comp)} companies")
    else:
        print("[INFO] No PnL trades generated, skipping company summary")

    # Industry summary (if available)
    if 'industry_large' in panel.columns:
        first_ind = (panel.sort_values(['company_id','month'])
                          .groupby('company_id')['industry_large'].first().reset_index())
        comp_ind = comp.merge(first_ind, on='company_id', how='left')
        ind_sum = (comp_ind.groupby('industry_large')
                   .agg(companies=('company_id','nunique'),
                        trades=('trades','sum'),
                        pnl_sum=('pnl_sum','sum'),
                        pnl_mean=('pnl_mean','mean'),
                        pnl_std=('pnl_std','mean')).reset_index())
        save_csv(ind_sum, str(Path(reports_dir)/'summary_industry_large.csv'))

    # Risk metrics (default + per jurisdiction if provided)
    risk_cfg = cfg.get('risk', {}) or {}
    alpha_default = float(risk_cfg.get('alpha', 0.99))
    window_default = int(risk_cfg.get('window_days', 252))
    
    if not pnl_df.empty and 'pnl_krw' in pnl_df.columns:
        pnl_series = pnl_df.set_index('fix_month')['pnl_krw'].sort_index()
        var_a = historical_var(pnl_series, alpha=alpha_default, window=min(window_default, len(pnl_series)))
        es_a  = expected_shortfall(pnl_series, alpha=alpha_default, window=min(window_default, len(pnl_series)))
        save_csv(pd.DataFrame([{'alpha':alpha_default,'window':window_default,'VaR':var_a,'ES':es_a}]), 
                 str(Path(reports_dir)/'var_es_summary.csv'))
        print(f"[INFO] Risk metrics calculated: VaR={var_a:.2f}, ES={es_a:.2f}")

        jurs = risk_cfg.get('jurisdictions', {}) or {}
        for jur, rc in jurs.items():
            a = float(rc.get('alpha', alpha_default)); w = int(rc.get('window_days', window_default))
            v = historical_var(pnl_series, alpha=a, window=min(w, len(pnl_series)))
            e = expected_shortfall(pnl_series, alpha=a, window=min(w, len(pnl_series)))
            save_csv(pd.DataFrame([{'alpha':a,'window':w,'VaR':v,'ES':e}]), 
                     str(Path(reports_dir)/f'var_es_summary_{jur}.csv'))
    else:
        print("[INFO] No PnL data available, skipping risk metrics")

    # Options PoC
    if len(spot_eom)>0:
        S0 = float(spot_eom.iloc[-1])
        opt = cfg.get('options_poC',{})
        vol=float(opt.get('vol_assumption',0.12)); rd=float(opt.get('risk_free_kr',0.035)); rf=float(opt.get('risk_free_us',0.045))
        expiries=opt.get('expiries_days',[30,90,180]); strikes_pct=opt.get('strikes_pct_of_spot',[0.95,1.0,1.05])
        rows=[]
        for d in expiries:
            T=d/365.0
            for p in strikes_pct:
                K=S0*p; call,put=garman_kohlhagen_call_put(S0,K,T,rd,rf,vol)
                rows.append({'spot_S0':S0,'K':K,'days':d,'vol':vol,'rd':rd,'rf':rf,'call':call,'put':put})
        save_csv(pd.DataFrame(rows), str(Path(reports_dir)/'options_summary.csv'))

    # IFRS9: monthly & quarterly effectiveness with sampling rules
    ifrs9_sampling = cfg.get('ifrs9', {}).get('sampling', {}) or {}
    period = (ifrs9_sampling.get('period') or 'M').upper()
    rule = (ifrs9_sampling.get('rule') or 'month_end')
    bounds = tuple(cfg.get('ifrs9',{}).get('dollar_offset_bounds',[0.8,1.25]))
    r2_th = float(cfg.get('ifrs9',{}).get('regression_r2_threshold',0.8))

    # Base monthly series
    y_m = spot_eom.copy()
    x_m = fwd_adj.copy()

    # Quarterly sampling if requested
    if period == 'Q':
        y_q = sample_series(spot_daily, spot_eom, period='Q', rule=rule,
                            countries=tuple((cfg.get('holidays') or {}).get('countries', ['KR'])),
                            cache_dir=(cfg.get('holidays') or {}).get('cache_dir','data/reference/holidays_cache'))
        # For forward, we don't have daily; use EOM fwd and sample by quarter end (equivalent to EOM of quarter)
        x_q = sample_series(None, fwd_adj, period='Q', rule='month_end',
                            countries=tuple((cfg.get('holidays') or {}).get('countries', ['KR'])),
                            cache_dir=(cfg.get('holidays') or {}).get('cache_dir','data/reference/holidays_cache'))
    else:
        y_q = None; x_q = None

    # Monthly detail
    do_m = dollar_offset_ratio(y_m, x_m); r2_m = regression_r2(y_m, x_m)
    stats_m = regression_stats(y_m, x_m)
    judge_m = judge_effective(do_m, r2_m, bounds=bounds, r2_th=r2_th)
    save_csv(pd.DataFrame([{**{'period':'M','DoR':do_m,'R2':r2_m,'judgement':judge_m}, **stats_m}]), 
             str(Path(reports_dir)/'ifrs9_effectiveness_M.csv'))

    # Quarterly detail (if applicable)
    if y_q is not None and x_q is not None and len(y_q)>2:
        do_q = dollar_offset_ratio(y_q, x_q); r2_q = regression_r2(y_q, x_q)
        stats_q = regression_stats(y_q, x_q)
        judge_q = judge_effective(do_q, r2_q, bounds=bounds, r2_th=r2_th)
        save_csv(pd.DataFrame([{**{'period':'Q','DoR':do_q,'R2':r2_q,'judgement':judge_q}, **stats_q}]), 
                 str(Path(reports_dir)/'ifrs9_effectiveness_Q.csv'))

    # IFRS9 effectiveness (Monthly & Quarterly)
    ifrs9_cfg = cfg.get('ifrs9',{}) or {}
    sampling_cfg = ifrs9_cfg.get('sampling', {'period':'M','rule':'month_end'})
    pairs = _resample_ifrs9_series(spot_eom, fwd_adj, spot_daily, cfg.get('holidays',{}), sampling_cfg)
    bounds = tuple(ifrs9_cfg.get('dollar_offset_bounds',[0.8,1.25]))
    r2_th = float(ifrs9_cfg.get('regression_r2_threshold',0.8))

    rows = []
    for tag, (tgt, hgd) in pairs.items():
        do = dollar_offset_ratio(tgt, hgd)
        stats = regression_stats(tgt, hgd)
        judge = judge_effective(do, stats['r2'], bounds=bounds, r2_th=r2_th)
        rows.append({'period':tag,'dollar_offset':do,'beta':stats['beta'],'alpha':stats['alpha'],
                     'r2':stats['r2'],'stderr_beta':stats['stderr_beta'],'t_beta':stats['t_beta'],
                     'p_beta':stats['p_beta'],'bounds_low':bounds[0],'bounds_high':bounds[1],
                     'r2_th':r2_th,'judgement':judge})
    eff_df = pd.DataFrame(rows)
    save_csv(eff_df, str(Path(reports_dir)/'ifrs9_effectiveness_summary.csv'))

    # Detailed regression tables per period
    for tag, (tgt, hgd) in pairs.items():
        stats = regression_stats(tgt, hgd)
        det = pd.DataFrame([stats])
        save_csv(det, str(Path(reports_dir)/f'ifrs9_regression_detail_{tag}.csv'))

    # Clustering
    cl = cfg.get('clustering', {}); feats = cl.get('features', ['export_amt','import_amt','net_exposure'])
    latest = (exposure_df.sort_values(['company_id','month']).groupby('company_id').tail(1))
    # Join with panel to get categorical columns if needed for clustering
    if 'region_sido_code' in feats or 'corp_grade_code' in feats:
        panel_cols = panel[['company_id', 'month', 'region_sido', 'industry', 'corp_grade']].drop_duplicates()
        latest = latest.merge(panel_cols, on=['company_id', 'month'], how='left')
        if 'region_sido_code' in feats and 'region_sido' in latest.columns: 
            latest['region_sido_code']=encode_categorical(latest['region_sido'])
        if 'corp_grade_code' in feats and 'corp_grade' in latest.columns: 
            latest['corp_grade_code']=encode_categorical(latest['corp_grade'])
    
    clustered, km_sum, gmm_sum = run_clustering(latest, feats, k_kmeans=int(cl.get('kmeans_k',4)), k_gmm=int(cl.get('gmm_k',4)))
    save_csv(km_sum, str(Path(reports_dir)/'cluster_summary_kmeans.csv'))
    save_csv(gmm_sum, str(Path(reports_dir)/'cluster_summary_gmm.csv'))
    clustered[['company_id','cluster_kmeans','cluster_gmm']].to_csv(str(Path(reports_dir)/'cluster_labels.csv'), 
                                                                    index=False, encoding='utf-8-sig')

    # Governance: snapshot + changelog
    if args.freeze_policy and (cfg.get('governance') or {}).get('enabled', True):
        gcfg = cfg.get('governance') or {}
        snap = snapshot_policy(cfg, str(Path(reports_dir)/Path(gcfg.get('snapshot_dir','reports/_artifacts/policy_snapshots')).relative_to('reports')) if str(gcfg.get('snapshot_dir','')).startswith('reports/') else gcfg.get('snapshot_dir','reports/_artifacts/policy_snapshots'))
        actor = args.actor or (gcfg.get('default_actor') or 'system')
        append_changelog(str(Path(reports_dir)/Path(gcfg.get('changelog_csv') or 'reports/policy_changelog.csv').name), 
                        action='pipeline_run', actor=actor, note=args.changelog_note, cfg=cfg)
        print(f"[INFO] Policy snapshot saved: {snap}")
    
    # IFRS9 governance snapshot
    if args.freeze_ifrs9 and (cfg.get('ifrs9') or {}).get('governance', {}).get('enabled', True):
        eg = cfg.get('ifrs9').get('governance') or {}
        eff_files = []
        for fn in ['ifrs9_effectiveness_M.csv', 'ifrs9_effectiveness_Q.csv']:
            p = Path(reports_dir)/fn
            if p.exists():
                eff_files.append(p)
        if eff_files:
            dd = pd.concat([pd.read_csv(p) for p in eff_files], ignore_index=True)
            snap = snapshot_effectiveness(dd, str(Path(reports_dir)/Path(eg.get('snapshot_dir','reports/_artifacts/ifrs9_snapshots')).relative_to('reports')) if str(eg.get('snapshot_dir','')).startswith('reports/') else eg.get('snapshot_dir','reports/_artifacts/ifrs9_snapshots'),
                                          meta={'bounds': bounds, 'r2_th': r2_th, 'sampling': {'period': period, 'rule': rule}})
            append_effectiveness_changelog(str(Path(reports_dir)/Path((eg.get('changelog_csv') or 'reports/ifrs9_changelog.csv')).name),
                                           action='assess', actor=(args.actor or 'system'), note=args.changelog_note,
                                           params={'bounds': bounds, 'r2_th': r2_th, 'sampling': {'period': period, 'rule': rule}})
            print(f"[INFO] IFRS9 snapshot saved: {snap}")
    
    # ML Performance Monitoring and Reporting
    if 'ml_models' in locals() and ml_models:
        try:
            from fx_external_pipeline_full.ml_monitoring import generate_comprehensive_ml_report
            
            # 기존 규칙 기반 exposure 데이터 생성 (비교용)
            original_exposure_df = compute_monthly_exposure(panel)
            original_exposure_df = apply_policy(original_exposure_df, cfg=cfg)  # 규칙 기반만
            
            # PnL 데이터 로드 (이미 생성된 백테스트 결과 사용)
            pnl_file = Path(reports_dir) / "pnl_by_trade.csv" 
            pnl_results = pd.read_csv(pnl_file) if pnl_file.exists() else pd.DataFrame()
            
            # ML 성과 리포트 생성
            generate_comprehensive_ml_report(
                ml_models=ml_models,
                exposure_df_original=original_exposure_df,
                exposure_df_enhanced=exposure_df,
                pnl_results=pnl_results,
                reports_dir=reports_dir
            )
            print("[INFO] ML performance report generated")
            
        except Exception as e:
            print(f"[WARN] ML performance reporting failed: {e}")
    
    print('Pipeline completed. Reports saved to:', reports_dir)


if __name__=='__main__':
    main()