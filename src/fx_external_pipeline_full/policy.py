import pandas as pd

def recommend_hedge_ratio(row, size_thresholds=(1e6, 5e6), ratios=(0.0, 0.5, 0.8)):
    ne = abs(float(row.get("net_exposure", 0.0)))
    t1, t2 = size_thresholds
    r0, r1, r2 = ratios
    if ne < t1: return r0
    if ne < t2: return r1
    return r2

def _apply_policy_v0(exposure_df: pd.DataFrame, cfg: dict|None=None) -> pd.DataFrame:
    df = exposure_df.copy()
    pol = (cfg or {}).get('policy', {}) if cfg else {}
    size_thresholds = tuple(pol.get('size_thresholds', [1e6, 5e6]))
    ratios = tuple(pol.get('ratios', [0.0, 0.5, 0.8]))
    df["hedge_ratio"] = df.apply(recommend_hedge_ratio, axis=1,
                                 args=(size_thresholds, ratios))
    df["policy_version"] = "v0"
    return df

def _apply_policy_v1(exposure_df: pd.DataFrame, features_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # base v0
    df = _apply_policy_v0(exposure_df, cfg)
    pol = (cfg.get('policy') or {})
    fcfg = (pol.get('features') or {})
    rv20_th = float(fcfg.get('rv20_thresh', 0.015))
    carry_th = float(fcfg.get('carry_thresh', 0.0))
    bump_to = float((pol.get('bump_to') or 0.8))
    reduce_to = float((pol.get('reduce_to') or 0.5))

    # merge monthly features (on 'month')
    feats = features_df.copy()
    if 'month' not in feats.columns:
        raise KeyError("features_df must include 'month' column")
    dd = df.merge(feats, on='month', how='left')

    # signals
    dd["sig_trend_up"] = (dd["ma20"] > dd["ma60"]).astype(int)
    dd["sig_high_vol"] = (dd["rv20"] > rv20_th).astype(int)
    dd["sig_carry_neg"] = (dd["carry"] < carry_th).astype(int)

    # decision logic:
    # if high vol OR negative carry OR (trend up AND net_exposure>0) => bump_to
    # elif trend down AND net_exposure<0 => reduce_to (hedge less if favorable trend & importer bias)
    cond_bump = (dd["sig_high_vol"]==1) | (dd["sig_carry_neg"]==1) | ((dd["sig_trend_up"]==1) & (dd["net_exposure"]>0))
    cond_reduce = ((dd["sig_trend_up"]==0) & (dd["net_exposure"]<0))

    dd.loc[cond_bump, "hedge_ratio"] = dd.loc[cond_bump, "hedge_ratio"].clip(lower=bump_to)
    dd.loc[cond_reduce, "hedge_ratio"] = dd.loc[cond_reduce, "hedge_ratio"].clip(upper=reduce_to)
    dd["policy_version"] = "v1"
    return dd

def apply_policy(exposure_df: pd.DataFrame, features_df: pd.DataFrame|None=None, cfg: dict|None=None, 
                 ml_models: dict|None=None, market_df: pd.DataFrame|None=None) -> pd.DataFrame:
    """
    Enhanced policy application with ML-based hedge ratio prediction
    """
    import logging
    logger = logging.getLogger(__name__)
    
    pol = (cfg or {}).get('policy', {}) if cfg else {}
    ml_cfg = (cfg or {}).get('ml_models', {}) if cfg else {}
    ver = pol.get('version', 'v0')
    
    # Check if ML hedge ratio prediction is available and enabled
    use_ml_hedge = (ml_models is not None and 
                    'hedge_ratio_predictor' in ml_models and
                    ml_cfg.get('hedge_ratio_prediction', {}).get('enabled', False))
    
    if use_ml_hedge and market_df is not None:
        try:
            logger.info("Applying ML-based hedge ratio prediction")
            hedge_predictor = ml_models['hedge_ratio_predictor']['model']
            predicted_ratios = hedge_predictor.predict_hedge_ratios(exposure_df, market_df)
            
            df = exposure_df.copy()
            df['hedge_ratio'] = predicted_ratios
            df['policy_version'] = 'ML_enhanced'
            
            # Still apply weights and bounds
            df = _apply_weights_and_bounds(df, cfg or {})
            
            logger.info(f"Applied ML hedge ratios. Mean: {predicted_ratios.mean():.3f}, "
                       f"Min: {predicted_ratios.min():.3f}, Max: {predicted_ratios.max():.3f}")
            
            return df
            
        except Exception as e:
            logger.warning(f"ML hedge ratio prediction failed: {e}. Falling back to rule-based.")
            # Set fallback flag to ensure proper fallback handling
            ver = pol.get('v1', {}).get('fallback_to_v0', True) and 'v0' or ver
    
    # Determine appropriate policy version based on available inputs
    if ver == 'v1':
        if features_df is not None and cfg is not None:
            try:
                logger.info("Applying rule-based v1 policy with features")
                return _apply_weights_and_bounds(_apply_policy_v1(exposure_df, features_df, cfg), cfg)
            except Exception as e:
                logger.warning(f"Policy v1 failed: {e}. Falling back to v0")
                # Fallback to v0 if v1 configuration allows it
                if pol.get('v1', {}).get('fallback_to_v0', True):
                    ver = 'v0'
                else:
                    raise e
        else:
            # Missing features_df for v1, check fallback option
            logger.warning(f"Policy v1 requires features_df but not provided. "
                          f"features_df is None: {features_df is None}")
            if pol.get('v1', {}).get('fallback_to_v0', True):
                logger.info("Falling back to v0 policy due to missing features")
                ver = 'v0'
            else:
                raise ValueError("Policy v1 requires features_df, but fallback to v0 is disabled")
    
    # Apply v0 policy (default or fallback)
    logger.info("Applying rule-based v0 policy")
    return _apply_weights_and_bounds(_apply_policy_v0(exposure_df, cfg or {}), cfg or {})

def _normalize_series(x):
    import numpy as np, pandas as pd
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().all(): 
        return s.fillna(0.0)
    m = float(s.mean()); sd = float(s.std(ddof=0)) or 1.0
    z = (s - m) / sd
    # squash to [-1,1] via tanh
    return z.apply(lambda v: float(np.tanh(v)))

def _apply_weights_and_bounds(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pol = (cfg.get('policy') or {})
    weights = (pol.get('weights') or {})
    w_credit = float(weights.get('credit', 0.0))
    w_size = float(weights.get('size', 0.0))
    w_rel = float(weights.get('relationship', 0.0))

    # Build normalized factors if columns exist
    credit = None
    for c in ['credit_score','corp_grade_code','corp_grade']:
        if c in df.columns:
            credit = df[c]
            break
    size = df.get('net_exposure', 0.0)
    rel = df.get('relationship_score', 0.0)

    import pandas as pd
    cred_n = _normalize_series(credit) if credit is not None else pd.Series(0.0, index=df.index)
    size_n = _normalize_series(size) if 'net_exposure' in df.columns else pd.Series(0.0, index=df.index)
    rel_n  = _normalize_series(rel) if 'relationship_score' in df.columns else pd.Series(0.0, index=df.index)

    adj_factor = 1.0 + w_credit*cred_n + w_size*size_n + w_rel*rel_n
    df['hedge_ratio'] = (df['hedge_ratio'] * adj_factor).astype(float)

    # floor/cap
    bounds = pol.get('bounds', {})
    min_r = float(bounds.get('min_ratio', 0.0))
    max_r = float(bounds.get('max_ratio', 1.0))
    df['hedge_ratio'] = df['hedge_ratio'].clip(lower=min_r, upper=max_r)
    
    # 순노출이 0인 경우 헤지비율을 0으로 강제 설정 (무의미한 트레이드 방지)
    if 'net_exposure' in df.columns:
        zero_exposure_mask = (df['net_exposure'].abs() < 1e-6)
        if zero_exposure_mask.any():
            df.loc[zero_exposure_mask, 'hedge_ratio'] = 0.0
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Set hedge ratio to 0 for {zero_exposure_mask.sum()} companies with zero net exposure")
    
    # 헤지비율 반올림 (보고서 품질 개선)
    df['hedge_ratio'] = df['hedge_ratio'].round(3)

    return df
