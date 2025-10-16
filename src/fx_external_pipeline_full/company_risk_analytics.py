"""
기업별, 월별 상세 리스크 분석 모듈
- VaR/ES 계산 (기업별, 월별)
- 급등기 PnL 시뮬레이션
- 환율 노출 규모 분석
- 권고 헤지비율 종합
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CompanyRiskAnalytics:
    def __init__(self, confidence_level=0.05, window_days=252):
        self.confidence_level = confidence_level
        self.window_days = window_days
        
    def calculate_company_monthly_var_es(self, 
                                        panel_data: pd.DataFrame,
                                        spot_rates: pd.DataFrame,
                                        kr_rates: pd.DataFrame,
                                        us_rates: pd.DataFrame) -> pd.DataFrame:
        """기업별, 월별 VaR/ES 계산"""
        
        logger.info("Calculating company-level monthly VaR/ES...")
        
        # 환율 변화율 계산
        spot_rates = spot_rates.copy()
        spot_rates['returns'] = spot_rates['spot_rate'].pct_change()
        spot_rates = spot_rates.dropna()
        
        results = []
        
        # 컬럼명 확인 및 조정
        if 'month' not in panel_data.columns:
            logger.warning("'month' column not found, trying alternative column names")
            return pd.DataFrame()
        
        # 기업별, 월별 분석
        for (company_id, month), group in panel_data.groupby(['company_id', 'month']):
            if len(group) == 0:
                continue
                
            # 해당 월의 노출 금액
            net_exposure = group['net_exposure'].iloc[0]
            hedge_ratio = group['hedge_ratio'].iloc[0]
            unhedged_exposure = net_exposure * (1 - hedge_ratio)
            
            if abs(unhedged_exposure) < 1000:  # 소액 제외
                continue
            
            # 해당 월 이전 252일간의 환율 데이터 사용
            month_date = pd.to_datetime(month)
            start_date = month_date - pd.Timedelta(days=self.window_days)
            
            historical_rates = spot_rates[
                (spot_rates['date'] >= start_date) & 
                (spot_rates['date'] <= month_date)
            ]['returns'].dropna()
            
            if len(historical_rates) < 30:  # 최소 30일 데이터 필요
                continue
            
            # VaR/ES 계산 (역사적 시뮬레이션)
            var_95 = np.percentile(historical_rates, self.confidence_level * 100)
            var_99 = np.percentile(historical_rates, 1)
            
            # Expected Shortfall (CVaR)
            es_95 = historical_rates[historical_rates <= var_95].mean()
            es_99 = historical_rates[historical_rates <= var_99].mean()
            
            # 금액 기준 VaR/ES
            var_95_amount = abs(unhedged_exposure * var_95)
            var_99_amount = abs(unhedged_exposure * var_99)
            es_95_amount = abs(unhedged_exposure * es_95)
            es_99_amount = abs(unhedged_exposure * es_99)
            
            results.append({
                'company_id': company_id,
                'month': month,
                'net_exposure': net_exposure,
                'hedge_ratio': hedge_ratio,
                'unhedged_exposure': unhedged_exposure,
                'var_95_pct': var_95 * 100,
                'var_99_pct': var_99 * 100,
                'es_95_pct': es_95 * 100,
                'es_99_pct': es_99 * 100,
                'var_95_amount': var_95_amount,
                'var_99_amount': var_99_amount,
                'es_95_amount': es_95_amount,
                'es_99_amount': es_99_amount,
                'historical_volatility': historical_rates.std() * np.sqrt(252) * 100
            })
        
        df_results = pd.DataFrame(results)
        logger.info(f"Calculated VaR/ES for {len(df_results)} company-month combinations")
        
        return df_results
    
    def simulate_stress_pnl(self, 
                           panel_data: pd.DataFrame,
                           spot_rates: pd.DataFrame) -> pd.DataFrame:
        """급등기 PnL 시뮬레이션"""
        
        logger.info("Running stress PnL simulation for extreme FX movements...")
        
        # 환율 급등/급락 시나리오 정의 (역사적 극단값 기준)
        spot_rates = spot_rates.copy()
        spot_rates['returns'] = spot_rates['spot_rate'].pct_change()
        
        # 역사적 극단값 (상위/하위 1%, 5%)
        extreme_scenarios = {
            'crash_1pct': np.percentile(spot_rates['returns'].dropna(), 1),
            'crash_5pct': np.percentile(spot_rates['returns'].dropna(), 5),
            'surge_95pct': np.percentile(spot_rates['returns'].dropna(), 95),
            'surge_99pct': np.percentile(spot_rates['returns'].dropna(), 99),
            'black_swan_down': -0.15,  # 15% 급락
            'black_swan_up': 0.15      # 15% 급등
        }
        
        results = []
        
        for (company_id, month), group in panel_data.groupby(['company_id', 'month']):
            if len(group) == 0:
                continue
                
            net_exposure = group['net_exposure'].iloc[0]
            hedge_ratio = group['hedge_ratio'].iloc[0]
            unhedged_exposure = net_exposure * (1 - hedge_ratio)
            
            if abs(unhedged_exposure) < 1000:
                continue
            
            # 각 시나리오별 PnL 계산
            scenario_pnls = {}
            for scenario_name, fx_change in extreme_scenarios.items():
                # 미헤지 포지션의 PnL (환율 상승시 수출기업은 이익)
                pnl = unhedged_exposure * fx_change
                scenario_pnls[f'pnl_{scenario_name}'] = pnl
            
            result = {
                'company_id': company_id,
                'month': month,
                'net_exposure': net_exposure,
                'hedge_ratio': hedge_ratio,
                'unhedged_exposure': unhedged_exposure,
                **scenario_pnls
            }
            
            # 최악의 손실 시나리오
            all_pnls = list(scenario_pnls.values())
            result['worst_case_pnl'] = min(all_pnls)
            result['best_case_pnl'] = max(all_pnls)
            result['pnl_range'] = max(all_pnls) - min(all_pnls)
            
            results.append(result)
        
        df_results = pd.DataFrame(results)
        logger.info(f"Completed stress testing for {len(df_results)} company-month combinations")
        
        return df_results
    
    def create_comprehensive_risk_report(self,
                                       panel_data: pd.DataFrame,
                                       spot_rates: pd.DataFrame,
                                       kr_rates: pd.DataFrame,
                                       us_rates: pd.DataFrame) -> pd.DataFrame:
        """종합 리스크 보고서 생성"""
        
        logger.info("Creating comprehensive company risk analytics report...")
        
        # VaR/ES 계산
        var_es_data = self.calculate_company_monthly_var_es(
            panel_data, spot_rates, kr_rates, us_rates
        )
        
        # 급등기 PnL 시뮬레이션
        stress_pnl_data = self.simulate_stress_pnl(panel_data, spot_rates)
        
        # 데이터 결합
        comprehensive_report = pd.merge(
            var_es_data,
            stress_pnl_data[['company_id', 'month', 'worst_case_pnl', 'best_case_pnl', 'pnl_range',
                           'pnl_crash_1pct', 'pnl_surge_99pct', 'pnl_black_swan_down', 'pnl_black_swan_up']],
            on=['company_id', 'month'],
            how='inner'
        )
        
        # 추가 분석 지표
        comprehensive_report['risk_rating'] = self._calculate_risk_rating(comprehensive_report)
        comprehensive_report['hedge_effectiveness'] = comprehensive_report['hedge_ratio'] * 100
        comprehensive_report['exposure_size_category'] = self._categorize_exposure_size(comprehensive_report['net_exposure'])
        
        # 정렬 및 포매팅
        comprehensive_report = comprehensive_report.sort_values(['company_id', 'month'])
        
        # 금액 컬럼 반올림
        amount_columns = [col for col in comprehensive_report.columns if 'amount' in col or 'pnl_' in col or 'exposure' in col]
        for col in amount_columns:
            if col in comprehensive_report.columns:
                comprehensive_report[col] = comprehensive_report[col].round(2)
        
        # 퍼센트 컬럼 반올림
        pct_columns = [col for col in comprehensive_report.columns if '_pct' in col or 'volatility' in col or 'ratio' in col]
        for col in pct_columns:
            if col in comprehensive_report.columns:
                comprehensive_report[col] = comprehensive_report[col].round(3)
        
        logger.info(f"Comprehensive risk report generated for {len(comprehensive_report)} records")
        
        return comprehensive_report
    
    def _calculate_risk_rating(self, data: pd.DataFrame) -> pd.Series:
        """리스크 등급 계산"""
        conditions = [
            (data['var_99_amount'] >= 1000000),  # 100만 이상
            (data['var_99_amount'] >= 500000),   # 50만 이상
            (data['var_99_amount'] >= 100000),   # 10만 이상
            (data['var_99_amount'] >= 50000),    # 5만 이상
            (data['var_99_amount'] < 50000)      # 5만 미만
        ]
        
        choices = ['Very High', 'High', 'Medium', 'Low', 'Very Low']
        
        return np.select(conditions, choices, default='Low')
    
    def _categorize_exposure_size(self, exposure: pd.Series) -> pd.Series:
        """노출 규모 카테고리 분류"""
        abs_exposure = abs(exposure)
        
        conditions = [
            (abs_exposure >= 10000000),   # 1천만 이상
            (abs_exposure >= 5000000),    # 500만 이상
            (abs_exposure >= 1000000),    # 100만 이상
            (abs_exposure >= 500000),     # 50만 이상
            (abs_exposure < 500000)       # 50만 미만
        ]
        
        choices = ['XL', 'L', 'M', 'S', 'XS']
        
        return np.select(conditions, choices, default='S')

def generate_company_risk_analytics(panel_data: pd.DataFrame,
                                  spot_rates: pd.DataFrame,
                                  kr_rates: pd.DataFrame,
                                  us_rates: pd.DataFrame,
                                  output_path: str = "reports/company_risk_analytics.csv"):
    """기업별 리스크 분석 실행 및 저장"""
    
    try:
        # 분석기 초기화
        analyzer = CompanyRiskAnalytics()
        
        # 종합 리스크 보고서 생성
        risk_report = analyzer.create_comprehensive_risk_report(
            panel_data, spot_rates, kr_rates, us_rates
        )
        
        # CSV 저장 (BOM 없이)
        risk_report.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Company risk analytics saved to: {output_path}")
        logger.info(f"Report contains {len(risk_report)} company-month risk assessments")
        
        # 요약 통계
        summary_stats = {
            'total_records': len(risk_report),
            'unique_companies': risk_report['company_id'].nunique(),
            'avg_var_99_amount': risk_report['var_99_amount'].mean(),
            'max_var_99_amount': risk_report['var_99_amount'].max(),
            'high_risk_companies': len(risk_report[risk_report['risk_rating'].isin(['High', 'Very High'])]),
            'total_unhedged_exposure': risk_report['unhedged_exposure'].abs().sum()
        }
        
        logger.info(f"Risk analytics summary: {summary_stats}")
        
        return risk_report, summary_stats
        
    except Exception as e:
        logger.error(f"Error in company risk analytics: {str(e)}")
        raise e