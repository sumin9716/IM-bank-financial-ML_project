import itertools

import numpy as np
import pandas as pd

from .features import compute_features
from .policy import apply_policy
from .backtest import monthly_forward_strategy
from .risk import expected_shortfall

def _objective(pnl: pd.Series, w_sigma=1.0, w_es=1.0, w_mean=0.5):
    """작을수록 좋은 위험 비용: 손익 표준편차와 ES99 손실 크기를 더하고 양(+)의 평균 손익을 뺀다."""
    if pnl is None or len(pnl)==0:
        return np.inf
    sigma = float(pnl.std())
    es99 = float(expected_shortfall(pnl, alpha=0.99, window=min(252, len(pnl))))
    es_loss = -min(0.0, es99)
    mean = float(pnl.mean())
    return w_sigma*sigma + w_es*es_loss - w_mean*max(0.0, mean)

def grid_tune_policy(panel: pd.DataFrame,
                     exposure_df: pd.DataFrame,
                     spot_daily: pd.Series|None,
                     spot_eom: pd.Series, r_us_eom: pd.Series, r_kr_eom: pd.Series,
                     forward_eom: pd.Series,
                     cfg: dict) -> pd.DataFrame:
    """v1 정책의 rv20/carry 임계값 격자를 백테스트해 목적함수 순으로 정렬한 표를 돌려준다.

    평균 손익이 v0 정책의 min_mean_ratio배 이상인 후보(mean_ok)를 먼저 놓는다.
    """
    # 신호 값은 임계값과 무관하므로 한 번만 계산한다
    base_feats = compute_features(spot_daily, spot_eom, r_us_eom, r_kr_eom, cfg)

    pol = cfg.get('policy', {})
    fcfg = pol.get('features', {})
    grid = (cfg.get('tuning') or {}).get('grid', {})
    rv_grid = grid.get('rv20_thresh', [0.01, 0.015, 0.02])
    carry_grid = grid.get('carry_thresh', [-0.002, 0.0, 0.002])
    weights = (cfg.get('tuning') or {}).get('objective_weights', {'sigma':1.0,'es':1.0,'mean':0.5})
    min_mean_ratio = float((cfg.get('tuning') or {}).get('min_mean_ratio', 0.9))

    results = []
    cfg_v0 = dict(cfg)
    cfg_v0['policy'] = dict(pol)
    cfg_v0['policy']['version'] = 'v0'
    expo_v0 = apply_policy(exposure_df, None, cfg_v0)
    pnl_v0 = monthly_forward_strategy(expo_v0, spot_eom, forward_eom)['pnl_krw']
    baseline_mean = float(pnl_v0.mean()) if len(pnl_v0)>0 else 0.0

    for rv_th, cy_th in itertools.product(rv_grid, carry_grid):
        test_cfg = dict(cfg)
        test_cfg['policy'] = dict(pol)
        test_cfg['policy']['version'] = 'v1'
        test_cfg['policy']['features'] = dict(fcfg)
        test_cfg['policy']['features']['rv20_thresh'] = float(rv_th)
        test_cfg['policy']['features']['carry_thresh'] = float(cy_th)

        expo = apply_policy(exposure_df, base_feats, test_cfg)
        pnl_df = monthly_forward_strategy(expo, spot_eom, forward_eom)
        pnl = pnl_df['pnl_krw'] if 'pnl_krw' in pnl_df.columns else pd.Series(dtype=float)

        mean_ok = True
        if baseline_mean != 0.0:
            mean_ok = (float(pnl.mean()) >= baseline_mean * min_mean_ratio)

        obj = _objective(pnl, w_sigma=weights.get('sigma',1.0),
                              w_es=weights.get('es',1.0),
                              w_mean=weights.get('mean',0.5))
        results.append({
            'rv20_thresh': rv_th,
            'carry_thresh': cy_th,
            'mean': float(pnl.mean()) if len(pnl)>0 else float('nan'),
            'std': float(pnl.std()) if len(pnl)>0 else float('nan'),
            'es99': float(expected_shortfall(pnl, alpha=0.99, window=min(252, len(pnl)))) if len(pnl)>0 else float('nan'),
            'objective': obj,
            'mean_ok': mean_ok
        })
    res = pd.DataFrame(results).sort_values(['mean_ok','objective','std'], ascending=[False, True, True]).reset_index(drop=True)
    return res
