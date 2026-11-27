#!/usr/bin/env python
"""월별 수출입 패널로 환노출을 계산하고 헤지 정책·백테스트·리스크·IFRS9 결과를 reports/에 저장한다."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from fx_external_pipeline_full.backtest import monthly_forward_strategy  # noqa: E402
from fx_external_pipeline_full.clustering import run_clustering  # noqa: E402
from fx_external_pipeline_full.company_risk_analytics import create_company_risk_analytics  # noqa: E402
from fx_external_pipeline_full.data_quality import (  # noqa: E402
    fill_monthly_gaps,
    generate_data_quality_report,
    recommend_data_fixes,
    validate_monthly_continuity,
)
from fx_external_pipeline_full.exposure import compute_monthly_exposure  # noqa: E402
from fx_external_pipeline_full.external_loader import read_eom_series_csv  # noqa: E402
from fx_external_pipeline_full.features import compute_features  # noqa: E402
from fx_external_pipeline_full.governance import (  # noqa: E402
    append_changelog,
    append_effectiveness_changelog,
    snapshot_effectiveness,
    snapshot_policy,
)
from fx_external_pipeline_full.hedge_effectiveness import (  # noqa: E402
    dollar_offset_ratio,
    judge_effective,
    regression_stats,
    sample_series,
)
from fx_external_pipeline_full.holiday_calendar import DEFAULT_CACHE_DIR  # noqa: E402
from fx_external_pipeline_full.ml_models import train_ml_models  # noqa: E402
from fx_external_pipeline_full.ml_monitoring import generate_comprehensive_ml_report  # noqa: E402
from fx_external_pipeline_full.policy import apply_policy  # noqa: E402
from fx_external_pipeline_full.policy_tuner import grid_tune_policy  # noqa: E402
from fx_external_pipeline_full.pricing import apply_spread, cip_forward_theoretical  # noqa: E402
from fx_external_pipeline_full.pricing_option import garman_kohlhagen_call_put  # noqa: E402
from fx_external_pipeline_full.reporting import save_csv  # noqa: E402
from fx_external_pipeline_full.risk import expected_shortfall, historical_var  # noqa: E402
from fx_external_pipeline_full.utils import encode_categorical, ensure_dir, load_yaml, setup_logging  # noqa: E402

# ECOS 항목을 설정하지 않았을 때 쓰는 기본값(한국은행 기준금리 통계표)
DEFAULT_ECOS_ITEM = {"stat_code": "722Y001", "cycle": "M", "item_code": ""}


def parse_args():
    ap = argparse.ArgumentParser(description='FX hedging pipeline runner')
    ap.add_argument('--config', default=str(PROJECT_ROOT/'config'/'config.yml'))
    ap.add_argument('--panel', default=str(PROJECT_ROOT/'data'/'panel_base.csv'))
    ap.add_argument('--spot', default=str(PROJECT_ROOT/'data'/'external'/'spot_usdkrw_eom.csv'))
    ap.add_argument('--kr',   default=str(PROJECT_ROOT/'data'/'external'/'kr_rates_month.csv'))
    ap.add_argument('--us',   default=str(PROJECT_ROOT/'data'/'external'/'us_rates_month.csv'))
    ap.add_argument('--reports_dir', default=str(PROJECT_ROOT/'reports'))
    ap.add_argument('--days', type=int, default=30, help='forward tenor in days')
    ap.add_argument('--spread_bps', type=float, default=10.0)
    ap.add_argument('--clusters', type=int, default=None,
                    help='number of K-Means/GMM clusters (overrides clustering.kmeans_k/gmm_k, default 4)')
    ap.add_argument('--profile', choices=['baseline','conservative','aggressive'],
                    help='apply overrides from the config "profiles" section')
    ap.add_argument('--tune_policy', action='store_true', help='run grid search to tune policy thresholds')
    ap.add_argument('--freeze_policy', action='store_true', help='snapshot current policy to artifacts and log change')
    ap.add_argument('--changelog_note', default='', help='note to append to policy changelog')
    ap.add_argument('--actor', default=None, help='actor name for governance changelog')
    ap.add_argument('--freeze_ifrs9', action='store_true', help='snapshot IFRS9 effectiveness results and log change')
    ap.add_argument('--ifrs9_note', default='', help='note for IFRS9 effectiveness changelog')
    return ap.parse_args()


def deep_update(d: dict, u: dict | None) -> dict:
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def apply_profile(cfg: dict, profile: str | None) -> dict:
    if not profile:
        return cfg
    prof = (cfg.get('profiles') or {}).get(profile)
    if not prof:
        print(f"[WARN] Profile '{profile}' not found. Using base config.")
        return cfg
    cfg = deep_update(cfg, prof)
    cfg['active_profile'] = profile
    print(f"[INFO] Profile applied: {profile}")
    return cfg


def resolve_path(path) -> str:
    """설정 파일의 상대 경로를 작업 디렉터리가 아니라 프로젝트 루트 기준으로 해석한다."""
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def holiday_settings(cfg: dict) -> tuple[tuple, str]:
    hcfg = cfg.get('holidays') or {}
    return tuple(hcfg.get('countries', ['KR'])), resolve_path(hcfg.get('cache_dir', DEFAULT_CACHE_DIR))


def under_reports_dir(path_cfg: str, reports_dir: str) -> str:
    """설정의 'reports/...' 경로를 --reports_dir 아래로 옮긴다."""
    if str(path_cfg).startswith('reports/'):
        return str(Path(reports_dir) / Path(path_cfg).relative_to('reports'))
    return path_cfg


def _fred_range(cfg: dict):
    rng = cfg.get('fred', {}).get('date_range') or {}
    return rng.get('start') or None, rng.get('end') or None


def _fred_series(cfg: dict, loader_name: str) -> pd.Series:
    # requests가 없을 때 ImportError로 CSV 대체가 되도록 loader는 호출 시점에 import한다.
    from fx_external_pipeline_full import fred_loader
    return getattr(fred_loader, loader_name)(*_fred_range(cfg))


def _ecos_rate(cfg: dict, item_key: str) -> pd.Series:
    from fx_external_pipeline_full.ecos_loader import load_ecos_series
    ec = cfg.get('ecos', {})
    dr = ec.get('date_range') or {}
    it = (ec.get('items') or {}).get(item_key) or DEFAULT_ECOS_ITEM
    pct = load_ecos_series(it['stat_code'], it['cycle'], dr.get('start') or "2018-01", dr.get('end') or "",
                           it.get('item_code', ""))
    return pct / 100.0


def _load_series(source: str, remote_loaders: dict, csv_path: str, csv_value_col: str) -> pd.Series:
    loader = remote_loaders.get(source)
    if loader is not None:
        try:
            return loader()
        except ImportError:
            print(f"[WARN] {source.upper()} loader not available. Using CSV.")
    return read_eom_series_csv(csv_path, csv_value_col)


def load_market_data(cfg: dict, args) -> tuple[pd.Series, pd.Series, pd.Series]:
    """data_source 설정에 따라 월말 USD/KRW, 한국 금리, 미국 금리를 CSV 또는 API에서 읽는다."""
    ds = cfg.get('data_source', {}) or {}
    spot_eom = _load_series(ds.get('spot', 'csv'),
                            {'fred': lambda: _fred_series(cfg, 'load_usdkrw_spot_eom_from_fred')},
                            args.spot, 'spot')
    us_eom = _load_series(ds.get('rates_us', 'csv'),
                          {'fred': lambda: _fred_series(cfg, 'load_us_1y_yield_eom_from_fred'),
                           'ecos': lambda: _ecos_rate(cfg, 'us_rate')},
                          args.us, 'rate')
    # FRED에는 한국 금리가 없어 SOFR를 대용으로 쓴다.
    kr_eom = _load_series(ds.get('rates_kr', 'csv'),
                          {'ecos': lambda: _ecos_rate(cfg, 'kr_policy_rate'),
                           'fred': lambda: _fred_series(cfg, 'load_sofr_eom_from_fred')},
                          args.kr, 'rate')
    return spot_eom, kr_eom, us_eom


def load_daily_spot(cfg: dict) -> pd.Series | None:
    """FRED에서 환율을 받는 설정일 때만 영업일 기준 IFRS9 표본용 환율을 읽는다."""
    if (cfg.get('data_source', {}) or {}).get('spot', 'csv') != 'fred':
        return None
    try:
        return _fred_series(cfg, 'load_usdkrw_spot_daily_from_fred')
    except Exception as e:  # 네트워크·인증 오류여도 월말 값으로 계속 진행한다
        print(f"[WARN] Could not load FRED daily spot. Falling back to EOM. Reason: {e}")
        return None


def prepare_exposure(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """패널을 기업·월별 순노출로 집계하고, 설정에 따라 월 공백을 채운다."""
    exposure_df = compute_monthly_exposure(panel)

    print("[INFO] Performing data quality validation...")
    continuity_check = validate_monthly_continuity(exposure_df)
    if not continuity_check['has_gaps']:
        print("[INFO] Data quality check passed: No monthly gaps detected")
        return exposure_df

    summary = continuity_check['summary']
    print(f"[WARN] Data quality issue: {summary['companies_with_gaps']}/{summary['total_companies']} "
          f"companies have monthly gaps ({summary['gap_percentage']:.1f}%)")

    dq_cfg = cfg.get('data_quality', {})
    fill_method = dq_cfg.get('gap_fill_method', 'zero')
    if not dq_cfg.get('auto_fill_gaps', True):
        print("[WARN] Gap filling disabled. Proceeding with original data (may cause maturity inconsistencies)")
        return exposure_df

    print(f"[INFO] Auto-filling gaps using method: {fill_method}")
    exposure_df = fill_monthly_gaps(exposure_df, fill_method=fill_method)
    print(f"[INFO] Gap filling completed. Dataset now has {len(exposure_df)} records")
    return exposure_df


def compute_policy_features(spot_daily, spot_eom, us_eom, kr_eom, cfg) -> pd.DataFrame | None:
    print("[INFO] Computing features for policy application...")
    try:
        features_df = compute_features(spot_daily, spot_eom, us_eom, kr_eom, cfg)
    except Exception as e:  # 특성이 없으면 apply_policy가 v0 규칙으로 내려간다
        print(f"[WARN] Feature computation failed: {e}. Policy will use v0 fallback.")
        return None
    print(f"[INFO] Features computed for {len(features_df)} time periods")
    return features_df


def train_models(exposure_df: pd.DataFrame, market_df: pd.DataFrame, cfg: dict) -> dict:
    print("[INFO] Training ML models...")
    try:
        ml_models = train_ml_models(exposure_df, market_df, cfg)
    except Exception as e:  # 모델이 없으면 규칙 기반 정책만 쓴다
        print(f"[WARN] ML model training failed: {e}")
        return {}

    if not ml_models:
        print("[INFO] No ML models were trained")
        return ml_models

    print(f"[INFO] Successfully trained {len(ml_models)} ML models")
    for model_name, model_info in ml_models.items():
        metrics = model_info.get('metrics', {})
        if 'test_r2' in metrics:
            print(f"  {model_name}: R² = {metrics['test_r2']:.3f}")
        if model_name == 'hedge_ratio_predictor' and 'feature_importance' in model_info:
            print(f"  Top features: {list(model_info['feature_importance'].keys())[:3]}")
    return ml_models


def write_exposure_forecast(ml_models: dict, exposure_df: pd.DataFrame, reports_dir: str) -> None:
    if 'exposure_forecaster' not in ml_models:
        return
    try:
        forecaster = ml_models['exposure_forecaster']['model']
        future_exposure = forecaster.predict_future_exposure(exposure_df)
    except Exception as e:
        print(f"[WARN] Future exposure forecasting failed: {e}")
        return
    save_csv(future_exposure, str(Path(reports_dir)/'future_exposure_forecast.csv'))
    print(f"[INFO] Future exposure forecast saved: {len(future_exposure)} companies")


def write_data_quality(exposure_df: pd.DataFrame, pnl_df: pd.DataFrame, reports_dir: str) -> None:
    print("[INFO] Generating comprehensive data quality report...")
    quality_report = generate_data_quality_report(
        exposure_df=exposure_df,
        pnl_df=pnl_df,
        output_path=str(Path(reports_dir)/'data_quality_report.json')
    )
    print("[INFO] Data Quality Recommendations:")
    for i, rec in enumerate(recommend_data_fixes(quality_report), 1):
        print(f"  {i}. {rec}")

    maturity = quality_report.get('maturity_validation', {})
    if maturity.get('has_inconsistencies'):
        stats = maturity['maturity_stats']
        print(f"[WARN] Forward maturity analysis: Mean={stats['mean_days']:.1f} days, "
              f"Range=[{stats['min_days']}-{stats['max_days']}] days")


def split_closed_trades(pnl_df: pd.DataFrame, reports_dir: str) -> pd.DataFrame:
    """미체결 포지션은 open_positions.csv로 따로 저장하고, 요약·리스크 계산에는 체결 거래만 돌려준다."""
    if pnl_df.empty or 'status' not in pnl_df.columns:
        return pnl_df
    open_positions = pnl_df[pnl_df['status'] == 'open']
    if not open_positions.empty:
        save_csv(open_positions, str(Path(reports_dir)/'open_positions.csv'))
        print(f"[INFO] Open positions saved: {len(open_positions)} positions")
    return pnl_df[pnl_df['status'] == 'closed']


def write_company_summaries(closed: pd.DataFrame, panel: pd.DataFrame, reports_dir: str) -> None:
    if closed.empty or 'company_id' not in closed.columns:
        print("[INFO] No PnL trades generated, skipping company summary")
        return

    comp = (closed.groupby('company_id').agg(
        trades=('pnl_krw','count'),
        pnl_sum=('pnl_krw','sum'),
        pnl_mean=('pnl_krw','mean'),
        pnl_std=('pnl_krw','std')).reset_index())
    save_csv(comp, str(Path(reports_dir)/'summary_company.csv'))
    print(f"[INFO] Company summary saved: {len(comp)} companies")

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


def write_var_es(closed: pd.DataFrame, cfg: dict, reports_dir: str) -> None:
    """체결 거래 손익으로 기본 및 관할(jurisdiction)별 VaR/ES를 저장한다."""
    if closed.empty or 'pnl_krw' not in closed.columns:
        print("[INFO] No PnL data available, skipping risk metrics")
        return

    # 노출이 0이라 손익이 0인 거래는 분포를 0 쪽으로 끌어당기므로 뺀다.
    non_zero_pnl = closed[closed['pnl_krw'].abs() > 1e-6]
    if len(non_zero_pnl) == 0:
        print("[INFO] No non-zero PnL trades available for risk calculation")
        return

    risk_cfg = cfg.get('risk', {}) or {}
    alpha_default = float(risk_cfg.get('alpha', 0.99))
    window_default = int(risk_cfg.get('window_days', 252))
    pnl_series = non_zero_pnl.set_index('fix_month')['pnl_krw'].sort_index()

    def save_var_es(alpha: float, window: int, filename: str):
        n = min(window, len(pnl_series))
        var = historical_var(pnl_series, alpha=alpha, window=n)
        es = expected_shortfall(pnl_series, alpha=alpha, window=n)
        save_csv(pd.DataFrame([{'alpha': alpha, 'window': window, 'VaR': var, 'ES': es}]),
                 str(Path(reports_dir)/filename))
        return var, es

    var_a, es_a = save_var_es(alpha_default, window_default, 'var_es_summary.csv')
    print(f"[INFO] Risk metrics calculated: VaR={var_a:.2f}, ES={es_a:.2f} (from {len(pnl_series)} trades)")

    for jur, rc in (risk_cfg.get('jurisdictions', {}) or {}).items():
        save_var_es(float(rc.get('alpha', alpha_default)), int(rc.get('window_days', window_default)),
                    f'var_es_summary_{jur}.csv')


def write_company_effectiveness(closed: pd.DataFrame, reports_dir: str) -> None:
    """기업별 간이 효과성 판정: 손익 표준편차가 평균 절댓값의 2배 이하이면 효과적으로 본다.

    IFRS9의 Dollar-offset·회귀 검정이 아니라 기업별 손익 안정성을 보는 보조 지표다.
    """
    print("[INFO] Performing IFRS9 hedge effectiveness tests...")
    if closed.empty or len(closed) < 8:
        print("[INFO] Insufficient data for IFRS9 effectiveness testing")
        return

    results = []
    for company_id, company_pnl in closed.groupby('company_id'):
        if len(company_pnl) < 4:
            continue
        pnl_values = company_pnl['pnl_krw'].values
        pnl_std = np.std(pnl_values)
        pnl_mean = np.abs(np.mean(pnl_values))
        results.append({
            'company_id': company_id,
            'test_type': 'dollar_offset_simplified',
            'is_effective': pnl_std <= (pnl_mean * 2) if pnl_mean > 0 else True,
            'pnl_mean': pnl_mean,
            'pnl_std': pnl_std,
            'sample_size': len(company_pnl),
            'test_date': company_pnl['fix_month'].max()
        })

    if not results:
        print("[INFO] No companies met minimum requirements for effectiveness testing")
        return
    effectiveness_df = pd.DataFrame(results)
    save_csv(effectiveness_df, str(Path(reports_dir)/'ifrs9_company_effectiveness.csv'))
    effective_rate = effectiveness_df['is_effective'].mean() * 100
    print(f"[INFO] IFRS9 effectiveness tests completed: {effective_rate:.1f}% effective "
          f"({len(effectiveness_df)} companies)")


def write_option_quotes(spot_eom: pd.Series, cfg: dict, reports_dir: str) -> None:
    """마지막 월말 환율 기준 Garman-Kohlhagen 옵션 가격표(PoC)."""
    if len(spot_eom) == 0:
        return
    S0 = float(spot_eom.iloc[-1])
    opt = cfg.get('options_poC', {})
    vol = float(opt.get('vol_assumption', 0.12))
    rd = float(opt.get('risk_free_kr', 0.035))
    rf = float(opt.get('risk_free_us', 0.045))
    rows = []
    for d in opt.get('expiries_days', [30, 90, 180]):
        T = d / 365.0
        for p in opt.get('strikes_pct_of_spot', [0.95, 1.0, 1.05]):
            K = S0 * p
            call, put = garman_kohlhagen_call_put(S0, K, T, rd, rf, vol)
            rows.append({'spot_S0':S0,'K':K,'days':d,'vol':vol,'rd':rd,'rf':rf,'call':call,'put':put})
    save_csv(pd.DataFrame(rows), str(Path(reports_dir)/'options_summary.csv'))


def ifrs9_samples(spot_eom, fwd_eom, spot_daily, rule, countries, cache_dir) -> dict:
    """{'M': (피헤지 대상, 헤지 수단), 'Q': (...)} — 대상은 환율, 수단은 선도환율."""
    # 선도환율은 일별 자료가 없어 'avg'가 아니면 분기 마지막 월말 값을 쓴다.
    hedge_rule = 'avg' if rule == 'avg' else 'month_end'
    target_q = sample_series(spot_daily, spot_eom, period='Q', rule=rule,
                             countries=countries, cache_dir=cache_dir)
    hedge_q = sample_series(None, fwd_eom, period='Q', rule=hedge_rule,
                            countries=countries, cache_dir=cache_dir)
    return {'M': (spot_eom, fwd_eom), 'Q': (target_q, hedge_q)}


def write_ifrs9_effectiveness(spot_eom, fwd_adj, spot_daily, cfg, reports_dir) -> dict:
    """기간별 Dollar-offset·회귀 결과를 저장하고, 거버넌스 기록에 남길 판정 기준을 돌려준다."""
    ifrs9_cfg = cfg.get('ifrs9', {}) or {}
    sampling = ifrs9_cfg.get('sampling', {}) or {}
    period = (sampling.get('period') or 'M').upper()
    rule = sampling.get('rule') or 'month_end'
    bounds = tuple(ifrs9_cfg.get('dollar_offset_bounds', [0.8, 1.25]))
    r2_th = float(ifrs9_cfg.get('regression_r2_threshold', 0.8))
    countries, cache_dir = holiday_settings(cfg)

    rows = []
    for tag, (target, hedge) in ifrs9_samples(spot_eom, fwd_adj, spot_daily, rule, countries, cache_dir).items():
        dor = dollar_offset_ratio(target, hedge)
        stats = regression_stats(target, hedge)
        judgement = judge_effective(dor, stats['r2'], bounds=bounds, r2_th=r2_th)

        # ifrs9_effectiveness_M/Q.csv는 --freeze_ifrs9 스냅샷의 입력이다. Q는 sampling.period가 Q일 때만 만든다.
        if tag == 'M' or (period == 'Q' and len(target) > 2):
            save_csv(pd.DataFrame([{'period': tag, 'DoR': dor, 'R2': stats['r2'], 'judgement': judgement, **stats}]),
                     str(Path(reports_dir)/f'ifrs9_effectiveness_{tag}.csv'))
        save_csv(pd.DataFrame([stats]), str(Path(reports_dir)/f'ifrs9_regression_detail_{tag}.csv'))
        rows.append({'period': tag, 'dollar_offset': dor, 'beta': stats['beta'], 'alpha': stats['alpha'],
                     'r2': stats['r2'], 'stderr_beta': stats['stderr_beta'], 't_beta': stats['t_beta'],
                     'p_beta': stats['p_beta'], 'bounds_low': bounds[0], 'bounds_high': bounds[1],
                     'r2_th': r2_th, 'judgement': judgement})
    save_csv(pd.DataFrame(rows), str(Path(reports_dir)/'ifrs9_effectiveness_summary.csv'))
    return {'bounds': bounds, 'r2_th': r2_th, 'sampling': {'period': period, 'rule': rule}}


def write_clusters(exposure_df: pd.DataFrame, panel: pd.DataFrame, cfg: dict, n_clusters: int | None,
                   reports_dir: str) -> None:
    """기업별 마지막 달 노출로 K-Means·GMM 군집을 만든다."""
    cl = cfg.get('clustering', {})
    feats = cl.get('features', ['export_amt','import_amt','net_exposure'])
    latest = exposure_df.sort_values(['company_id','month']).groupby('company_id').tail(1)
    if 'region_sido_code' in feats or 'corp_grade_code' in feats:
        panel_cols = panel[['company_id', 'month', 'region_sido', 'industry', 'corp_grade']].drop_duplicates()
        latest = latest.merge(panel_cols, on=['company_id', 'month'], how='left')
        if 'region_sido_code' in feats and 'region_sido' in latest.columns:
            latest['region_sido_code'] = encode_categorical(latest['region_sido'])
        if 'corp_grade_code' in feats and 'corp_grade' in latest.columns:
            latest['corp_grade_code'] = encode_categorical(latest['corp_grade'])

    k_kmeans = n_clusters or int(cl.get('kmeans_k', 4))
    k_gmm = n_clusters or int(cl.get('gmm_k', 4))
    clustered, km_sum, gmm_sum = run_clustering(latest, feats, k_kmeans=k_kmeans, k_gmm=k_gmm)
    save_csv(km_sum, str(Path(reports_dir)/'cluster_summary_kmeans.csv'))
    save_csv(gmm_sum, str(Path(reports_dir)/'cluster_summary_gmm.csv'))
    clustered[['company_id','cluster_kmeans','cluster_gmm']].to_csv(
        str(Path(reports_dir)/'cluster_labels.csv'), index=False, encoding='utf-8-sig')


def write_governance(args, cfg: dict, reports_dir: str, ifrs9_meta: dict) -> None:
    gcfg = cfg.get('governance') or {}
    if args.freeze_policy and gcfg.get('enabled', True):
        snap_dir = under_reports_dir(gcfg.get('snapshot_dir', 'reports/_artifacts/policy_snapshots'), reports_dir)
        snap = snapshot_policy(cfg, snap_dir)
        changelog = Path(reports_dir) / Path(gcfg.get('changelog_csv') or 'reports/policy_changelog.csv').name
        append_changelog(str(changelog), action='pipeline_run',
                         actor=args.actor or gcfg.get('default_actor') or 'system',
                         note=args.changelog_note, cfg=cfg)
        print(f"[INFO] Policy snapshot saved: {snap}")

    eg = (cfg.get('ifrs9') or {}).get('governance') or {}
    if not (args.freeze_ifrs9 and eg.get('enabled', True)):
        return
    eff_files = [p for p in (Path(reports_dir)/'ifrs9_effectiveness_M.csv', Path(reports_dir)/'ifrs9_effectiveness_Q.csv')
                 if p.exists()]
    if not eff_files:
        return
    results = pd.concat([pd.read_csv(p) for p in eff_files], ignore_index=True)
    snap_dir = under_reports_dir(eg.get('snapshot_dir', 'reports/_artifacts/ifrs9_snapshots'), reports_dir)
    snap = snapshot_effectiveness(results, snap_dir, meta=ifrs9_meta)
    changelog = Path(reports_dir) / Path(eg.get('changelog_csv') or 'reports/ifrs9_changelog.csv').name
    append_effectiveness_changelog(str(changelog), action='assess', actor=args.actor or 'system',
                                   note=args.ifrs9_note or args.changelog_note, params=ifrs9_meta)
    print(f"[INFO] IFRS9 snapshot saved: {snap}")


def write_ml_report(ml_models: dict, panel: pd.DataFrame, cfg: dict, exposure_df: pd.DataFrame,
                    pnl_df: pd.DataFrame, reports_dir: str) -> None:
    """ML 헤지 비율과 규칙 기반(v0) 헤지 비율을 비교한 리포트와 그림을 만든다."""
    if not ml_models:
        return
    try:
        rule_based = apply_policy(compute_monthly_exposure(panel), cfg=cfg)
        generate_comprehensive_ml_report(
            ml_models=ml_models,
            exposure_df_original=rule_based,
            exposure_df_enhanced=exposure_df,
            pnl_results=pnl_df,
            reports_dir=reports_dir
        )
    except Exception as e:  # 부가 리포트가 실패해도 핵심 산출물은 이미 저장되어 있다
        print(f"[WARN] ML performance reporting failed: {e}")
        return
    print("[INFO] ML performance report generated")


def write_company_risk(args, reports_dir: str) -> None:
    print("[INFO] Generating comprehensive company risk analytics...")
    try:
        _, risk_summary = create_company_risk_analytics(
            str(Path(reports_dir)/'exposure.csv'),
            str(Path(reports_dir)/'company_risk_analytics.csv'),
            spot_file=args.spot,
        )
    except Exception as e:  # 부가 리포트가 실패해도 핵심 산출물은 이미 저장되어 있다
        print(f"[WARN] Company risk analytics generation failed: {e}")
        return
    print(f"[INFO] Company risk analytics completed: {risk_summary['total_records']} records "
          f"for {risk_summary['unique_companies']} companies")


def main():
    args = parse_args()
    cfg = apply_profile(load_yaml(args.config), args.profile)
    setup_logging(str(PROJECT_ROOT/'config'/'logging.yml'))

    reports_dir = args.reports_dir
    ensure_dir(reports_dir)
    countries, holiday_cache_dir = holiday_settings(cfg)

    panel = pd.read_csv(args.panel, parse_dates=['month'])
    exposure_df = prepare_exposure(panel, cfg)

    spot_eom, kr_eom, us_eom = load_market_data(cfg, args)
    spot_daily = load_daily_spot(cfg)
    market_df = pd.DataFrame({'spot': spot_eom, 'us_rate': us_eom, 'kr_rate': kr_eom})

    features_df = compute_policy_features(spot_daily, spot_eom, us_eom, kr_eom, cfg)
    ml_models = train_models(exposure_df, market_df, cfg)
    exposure_df = apply_policy(exposure_df, features_df=features_df, cfg=cfg,
                               ml_models=ml_models, market_df=market_df)
    save_csv(exposure_df, str(Path(reports_dir)/'exposure.csv'))
    write_exposure_forecast(ml_models, exposure_df, reports_dir)

    fwd_theo = cip_forward_theoretical(spot_eom, us_eom, kr_eom, days=args.days)
    fwd_adj = apply_spread(fwd_theo, spread_bps=args.spread_bps, side='sell_usd')

    if args.tune_policy:
        res = grid_tune_policy(panel, exposure_df, spot_daily, spot_eom, us_eom, kr_eom, fwd_theo, cfg)
        save_csv(res, str(Path(reports_dir)/'policy_tuning_results.csv'))
        print('[INFO] Policy tuning done. Top row is the best candidate.')

    pnl_df = monthly_forward_strategy(exposure_df, spot_eom, fwd_adj,
                                      countries=countries, holiday_cache_dir=holiday_cache_dir)
    save_csv(pnl_df, str(Path(reports_dir)/'pnl_by_trade.csv'))
    write_data_quality(exposure_df, pnl_df, reports_dir)

    closed = split_closed_trades(pnl_df, reports_dir)
    write_company_summaries(closed, panel, reports_dir)
    write_var_es(closed, cfg, reports_dir)
    write_company_effectiveness(closed, reports_dir)
    write_option_quotes(spot_eom, cfg, reports_dir)
    ifrs9_meta = write_ifrs9_effectiveness(spot_eom, fwd_adj, spot_daily, cfg, reports_dir)
    write_clusters(exposure_df, panel, cfg, args.clusters, reports_dir)
    write_governance(args, cfg, reports_dir, ifrs9_meta)
    write_ml_report(ml_models, panel, cfg, exposure_df, pnl_df, reports_dir)
    write_company_risk(args, reports_dir)

    print('Pipeline completed. Reports saved to:', reports_dir)


if __name__=='__main__':
    main()
