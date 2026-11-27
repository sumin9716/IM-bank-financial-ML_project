"""
간단한 기업별 리스크 분석 생성
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_simple_risk_analytics(exposure_file='reports/exposure.csv', 
                                output_file='reports/company_risk_analytics.csv'):
    """기업별, 월별 상세 리스크 분석 생성"""
    
    try:
        # 데이터 로드
        exposure_df = pd.read_csv(exposure_file)
        spot_df = pd.read_csv('data/external/spot_usdkrw_eom.csv')
        
        print(f"Loaded {len(exposure_df)} exposure records")
        
        # 환율 변화율 계산
        spot_df['returns'] = spot_df['spot'].pct_change()
        
        # 역사적 변동성 계산 (연환산)
        historical_vol = spot_df['returns'].std() * np.sqrt(252) * 100
        
        # 극단 시나리오 (역사적 1%, 5%, 95%, 99% 분위수)
        returns_clean = spot_df['returns'].dropna()
        scenarios = {
            'crash_1pct': np.percentile(returns_clean, 1),
            'crash_5pct': np.percentile(returns_clean, 5),
            'surge_95pct': np.percentile(returns_clean, 95),
            'surge_99pct': np.percentile(returns_clean, 99)
        }
        
        # 기업별, 월별 리스크 분석
        risk_analytics = []
        
        for _, row in exposure_df.iterrows():
            company_id = row['company_id']
            month = row['month']
            net_exposure = row['net_exposure']
            hedge_ratio = row['hedge_ratio']
            
            # 미헤지 노출 금액
            unhedged_exposure = net_exposure * (1 - hedge_ratio)
            
            if abs(unhedged_exposure) < 1000:  # 소액 제외
                continue
            
            # VaR 계산 (95%, 99% 신뢰구간)
            var_95_pct = np.percentile(returns_clean, 5)  # 95% VaR
            var_99_pct = np.percentile(returns_clean, 1)  # 99% VaR
            
            # Expected Shortfall
            es_95_pct = returns_clean[returns_clean <= var_95_pct].mean()
            es_99_pct = returns_clean[returns_clean <= var_99_pct].mean()
            
            # 금액 기준 VaR/ES
            var_95_amount = abs(unhedged_exposure * var_95_pct)
            var_99_amount = abs(unhedged_exposure * var_99_pct)
            es_95_amount = abs(unhedged_exposure * es_95_pct)
            es_99_amount = abs(unhedged_exposure * es_99_pct)
            
            # 극단 시나리오 PnL
            stress_pnls = {}
            for scenario_name, fx_change in scenarios.items():
                pnl = unhedged_exposure * fx_change
                stress_pnls[f'pnl_{scenario_name}'] = pnl
            
            # 최악/최선 시나리오
            all_pnls = list(stress_pnls.values())
            worst_case_pnl = min(all_pnls)
            best_case_pnl = max(all_pnls)
            
            # 리스크 등급
            if var_99_amount >= 1000000:
                risk_rating = 'Very High'
            elif var_99_amount >= 500000:
                risk_rating = 'High'
            elif var_99_amount >= 100000:
                risk_rating = 'Medium'
            elif var_99_amount >= 50000:
                risk_rating = 'Low'
            else:
                risk_rating = 'Very Low'
            
            # 노출 규모 카테고리
            abs_exposure = abs(net_exposure)
            if abs_exposure >= 10000000:
                exposure_category = 'XL'
            elif abs_exposure >= 5000000:
                exposure_category = 'L'
            elif abs_exposure >= 1000000:
                exposure_category = 'M'
            elif abs_exposure >= 500000:
                exposure_category = 'S'
            else:
                exposure_category = 'XS'
            
            # 결과 저장
            risk_record = {
                'company_id': company_id,
                'month': month,
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
                'risk_rating': risk_rating,
                'exposure_size_category': exposure_category,
                **{k: round(v, 2) for k, v in stress_pnls.items()}
            }
            
            risk_analytics.append(risk_record)
        
        # DataFrame 생성 및 저장
        risk_df = pd.DataFrame(risk_analytics)
        risk_df = risk_df.sort_values(['company_id', 'month'])
        
        # CSV 저장 (BOM 없이)
        risk_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 요약 통계
        summary_stats = {
            'total_records': len(risk_df),
            'unique_companies': risk_df['company_id'].nunique(),
            'avg_var_99_amount': risk_df['var_99_amount'].mean(),
            'max_var_99_amount': risk_df['var_99_amount'].max(),
            'high_risk_records': len(risk_df[risk_df['risk_rating'].isin(['High', 'Very High'])]),
            'total_unhedged_exposure': risk_df['unhedged_exposure'].abs().sum()
        }
        
        print(f"Company risk analytics saved to: {output_file}")
        print(f"Generated {len(risk_df)} records for {risk_df['company_id'].nunique()} companies")
        print("Summary statistics:")
        for key, value in summary_stats.items():
            print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value:,}")
        
        return risk_df, summary_stats
        
    except Exception as e:
        print(f"Error generating company risk analytics: {e}")
        raise e

if __name__ == "__main__":
    create_simple_risk_analytics()