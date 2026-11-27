"""기업·월별 미헤지 노출의 VaR/ES와 극단 시나리오 손익을 계산한다.

수익률 분포는 월말 USD/KRW 환율 전체 기간에서 한 번 추정하고, 모든 기업·월에
같은 분포를 적용한다.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPOT_FILE = PROJECT_ROOT / "data" / "external" / "spot_usdkrw_eom.csv"

# 미헤지 노출이 이 금액보다 작으면 분석에서 뺀다
MIN_UNHEDGED_EXPOSURE = 1000

# (하한, 등급) — 위에서부터 처음 만족하는 구간을 쓴다
RISK_RATING_BANDS = [
    (1_000_000, 'Very High'),
    (500_000, 'High'),
    (100_000, 'Medium'),
    (50_000, 'Low'),
]
EXPOSURE_SIZE_BANDS = [
    (10_000_000, 'XL'),
    (5_000_000, 'L'),
    (1_000_000, 'M'),
    (500_000, 'S'),
]


def classify_risk_rating(var_99_amount: float) -> str:
    for lower, label in RISK_RATING_BANDS:
        if var_99_amount >= lower:
            return label
    return 'Very Low'


def classify_exposure_size(abs_exposure: float) -> str:
    for lower, label in EXPOSURE_SIZE_BANDS:
        if abs_exposure >= lower:
            return label
    return 'XS'


def create_company_risk_analytics(exposure_file, output_file, spot_file=DEFAULT_SPOT_FILE):
    """exposure.csv를 읽어 기업·월별 리스크 표를 output_file에 저장하고 (표, 요약)을 돌려준다."""
    exposure_df = pd.read_csv(exposure_file)
    spot_df = pd.read_csv(spot_file)
    logger.info(f"Loaded {len(exposure_df)} exposure records")

    spot_df['returns'] = spot_df['spot'].pct_change()
    # 월간 수익률인데 일별 기준인 sqrt(252)를 곱한다(월간 기준 sqrt(12)의 약 4.6배). 커밋된 결과와 맞추려고 그대로 둔다.
    historical_vol = spot_df['returns'].std() * np.sqrt(252) * 100

    returns_clean = spot_df['returns'].dropna()
    scenarios = {
        'crash_1pct': np.percentile(returns_clean, 1),
        'crash_5pct': np.percentile(returns_clean, 5),
        'surge_95pct': np.percentile(returns_clean, 95),
        'surge_99pct': np.percentile(returns_clean, 99)
    }
    var_95_pct = np.percentile(returns_clean, 5)
    var_99_pct = np.percentile(returns_clean, 1)
    es_95_pct = returns_clean[returns_clean <= var_95_pct].mean()
    es_99_pct = returns_clean[returns_clean <= var_99_pct].mean()

    risk_analytics = []
    for _, row in exposure_df.iterrows():
        net_exposure = row['net_exposure']
        hedge_ratio = row['hedge_ratio']
        unhedged_exposure = net_exposure * (1 - hedge_ratio)

        if abs(unhedged_exposure) < MIN_UNHEDGED_EXPOSURE:
            continue

        var_95_amount = abs(unhedged_exposure * var_95_pct)
        var_99_amount = abs(unhedged_exposure * var_99_pct)
        es_95_amount = abs(unhedged_exposure * es_95_pct)
        es_99_amount = abs(unhedged_exposure * es_99_pct)

        stress_pnls = {f'pnl_{name}': unhedged_exposure * fx_change
                       for name, fx_change in scenarios.items()}
        worst_case_pnl = min(stress_pnls.values())
        best_case_pnl = max(stress_pnls.values())

        risk_analytics.append({
            'company_id': row['company_id'],
            'month': row['month'],
            'net_exposure': round(net_exposure, 2),
            'hedge_ratio': round(hedge_ratio, 3),
            'unhedged_exposure': round(unhedged_exposure, 2),
            'var_95_percent': round(var_95_pct * 100, 3),
            'var_99_percent': round(var_99_pct * 100, 3),
            'es_95_percent': round(es_95_pct * 100, 3),
            'es_99_percent': round(es_99_pct * 100, 3),
            'var_95_amount': round(var_95_amount, 2),
            'var_99_amount': round(var_99_amount, 2),
            'es_95_amount': round(es_95_amount, 2),
            'es_99_amount': round(es_99_amount, 2),
            'historical_volatility_pct': round(historical_vol, 2),
            'worst_case_pnl': round(worst_case_pnl, 2),
            'best_case_pnl': round(best_case_pnl, 2),
            'pnl_range': round(best_case_pnl - worst_case_pnl, 2),
            'risk_rating': classify_risk_rating(var_99_amount),
            'exposure_size_category': classify_exposure_size(abs(net_exposure)),
            **{k: round(v, 2) for k, v in stress_pnls.items()}
        })

    risk_df = pd.DataFrame(risk_analytics).sort_values(['company_id', 'month'])
    risk_df.to_csv(output_file, index=False, encoding='utf-8')

    summary_stats = {
        'total_records': len(risk_df),
        'unique_companies': risk_df['company_id'].nunique(),
        'avg_var_99_amount': risk_df['var_99_amount'].mean(),
        'max_var_99_amount': risk_df['var_99_amount'].max(),
        'high_risk_records': len(risk_df[risk_df['risk_rating'].isin(['High', 'Very High'])]),
        'total_unhedged_exposure': risk_df['unhedged_exposure'].abs().sum()
    }
    logger.info(f"Company risk analytics saved to: {output_file}")
    logger.info(f"Risk analytics summary: {summary_stats}")
    return risk_df, summary_stats
