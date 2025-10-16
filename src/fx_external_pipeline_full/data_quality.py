"""
Data Quality and Continuity Validation Module
데이터 품질 및 연속성 검증 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_monthly_continuity(exposure_df: pd.DataFrame, 
                              company_col: str = 'company_id',
                              date_col: str = 'month') -> Dict[str, any]:
    """
    회사별 월별 데이터 연속성 검증
    """
    validation_results = {
        'has_gaps': False,
        'companies_with_gaps': [],
        'gap_details': [],
        'summary': {}
    }
    
    gap_companies = []
    gap_details = []
    
    for company_id, group in exposure_df.groupby(company_col):
        group = group.sort_values(date_col)
        dates = pd.to_datetime(group[date_col])
        
        if len(dates) < 2:
            continue
            
        # 월별 연속성 체크
        expected_months = pd.date_range(
            start=dates.min(), 
            end=dates.max(), 
            freq='ME'
        )
        
        actual_months = set(dates.dt.to_period('M'))
        expected_months_set = set(expected_months.to_period('M'))
        
        missing_months = expected_months_set - actual_months
        
        if missing_months:
            gap_companies.append(company_id)
            for missing_month in sorted(missing_months):
                gap_details.append({
                    'company_id': company_id,
                    'missing_month': str(missing_month),
                    'gap_type': 'missing_month'
                })
                
            logger.warning(f"Company {company_id}: {len(missing_months)} missing months - "
                         f"{sorted([str(m) for m in missing_months])}")
    
    validation_results['has_gaps'] = len(gap_companies) > 0
    validation_results['companies_with_gaps'] = gap_companies
    validation_results['gap_details'] = gap_details
    validation_results['summary'] = {
        'total_companies': exposure_df[company_col].nunique(),
        'companies_with_gaps': len(gap_companies),
        'gap_percentage': len(gap_companies) / exposure_df[company_col].nunique() * 100,
        'total_missing_months': len(gap_details)
    }
    
    return validation_results


def fill_monthly_gaps(exposure_df: pd.DataFrame, 
                     company_col: str = 'company_id',
                     date_col: str = 'month',
                     fill_method: str = 'zero') -> pd.DataFrame:
    """
    월별 데이터 공백을 채우는 함수
    
    Args:
        exposure_df: 원본 노출 데이터
        fill_method: 'zero' (0으로 채움), 'forward_fill' (앞 값으로 채움), 'interpolate' (보간)
    """
    logger.info(f"Filling monthly gaps using method: {fill_method}")
    
    result_dfs = []
    fill_stats = {'companies_filled': 0, 'months_added': 0}
    
    for company_id, group in exposure_df.groupby(company_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        
        if len(group) < 2:
            result_dfs.append(group)
            continue
            
        # 전체 기간의 월말 날짜 생성
        start_date = pd.to_datetime(group[date_col].min())
        end_date = pd.to_datetime(group[date_col].max())
        
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # 기존 데이터를 날짜로 인덱스
        group[date_col] = pd.to_datetime(group[date_col])
        group_indexed = group.set_index(date_col)
        
        # 전체 날짜 범위로 리인덱싱
        full_indexed = group_indexed.reindex(full_date_range)
        
        # company_id는 항상 채움
        full_indexed[company_col] = company_id
        
        missing_count = full_indexed.isnull().any(axis=1).sum()
        
        if missing_count > 0:
            fill_stats['companies_filled'] += 1
            fill_stats['months_added'] += missing_count
            
            logger.info(f"Company {company_id}: Filling {missing_count} missing months")
            
            if fill_method == 'zero':
                # 노출량 관련 컬럼들을 0으로 채움
                exposure_cols = [col for col in full_indexed.columns 
                               if 'exposure' in col.lower() or 'hedge' in col.lower()]
                full_indexed[exposure_cols] = full_indexed[exposure_cols].fillna(0.0)
                
                # 기타 수치형 컬럼도 0으로 채움
                numeric_cols = full_indexed.select_dtypes(include=[np.number]).columns
                full_indexed[numeric_cols] = full_indexed[numeric_cols].fillna(0.0)
                
            elif fill_method == 'forward_fill':
                full_indexed = full_indexed.fillna(method='ffill')
                
            elif fill_method == 'interpolate':
                numeric_cols = full_indexed.select_dtypes(include=[np.number]).columns
                full_indexed[numeric_cols] = full_indexed[numeric_cols].interpolate()
                
            # 문자열 컬럼 처리
            string_cols = full_indexed.select_dtypes(include=['object']).columns
            for col in string_cols:
                if col != company_col:
                    full_indexed[col] = full_indexed[col].fillna('filled')
        
        # 인덱스를 컬럼으로 다시 변환
        full_indexed = full_indexed.reset_index()
        full_indexed = full_indexed.rename(columns={'index': date_col})
        
        result_dfs.append(full_indexed)
    
    result_df = pd.concat(result_dfs, ignore_index=True)
    result_df = result_df.sort_values([company_col, date_col]).reset_index(drop=True)
    
    logger.info(f"Gap filling complete: {fill_stats['companies_filled']} companies, "
               f"{fill_stats['months_added']} months added")
    
    return result_df


def validate_forward_maturity_consistency(pnl_df: pd.DataFrame,
                                        expected_maturity_days: int = 30) -> Dict[str, any]:
    """
    선도 만기 일수 일관성 검증
    """
    if pnl_df.empty or 'trade_month' not in pnl_df.columns or 'fix_month' not in pnl_df.columns:
        return {'has_inconsistencies': False, 'details': []}
    
    pnl_df = pnl_df.copy()
    pnl_df['trade_month'] = pd.to_datetime(pnl_df['trade_month'])
    pnl_df['fix_month'] = pd.to_datetime(pnl_df['fix_month'])
    
    # 실제 만기 일수 계산
    pnl_df['actual_maturity_days'] = (pnl_df['fix_month'] - pnl_df['trade_month']).dt.days
    
    # 예상과 다른 만기 찾기
    tolerance = 10  # ±10일 허용
    inconsistent_mask = abs(pnl_df['actual_maturity_days'] - expected_maturity_days) > tolerance
    
    inconsistencies = pnl_df[inconsistent_mask].copy()
    
    validation_results = {
        'has_inconsistencies': len(inconsistencies) > 0,
        'total_trades': len(pnl_df),
        'inconsistent_trades': len(inconsistencies),
        'inconsistency_percentage': len(inconsistencies) / len(pnl_df) * 100 if len(pnl_df) > 0 else 0,
        'maturity_stats': {
            'mean_days': pnl_df['actual_maturity_days'].mean(),
            'median_days': pnl_df['actual_maturity_days'].median(),
            'min_days': pnl_df['actual_maturity_days'].min(),
            'max_days': pnl_df['actual_maturity_days'].max(),
            'std_days': pnl_df['actual_maturity_days'].std()
        },
        'details': inconsistencies[['company_id', 'trade_month', 'fix_month', 'actual_maturity_days']].to_dict('records') if len(inconsistencies) > 0 else []
    }
    
    if validation_results['has_inconsistencies']:
        logger.warning(f"Found {len(inconsistencies)} trades with inconsistent maturity "
                      f"(expected ~{expected_maturity_days} days, tolerance ±{tolerance} days)")
        
        # 심각한 이상치 별도 로깅
        extreme_mask = pnl_df['actual_maturity_days'] > 60  # 2개월 초과
        if extreme_mask.any():
            extreme_count = extreme_mask.sum()
            max_days = pnl_df.loc[extreme_mask, 'actual_maturity_days'].max()
            logger.error(f"Found {extreme_count} trades with extremely long maturity (max: {max_days} days)")
    
    return validation_results


def generate_data_quality_report(exposure_df: pd.DataFrame, 
                               pnl_df: pd.DataFrame = None,
                               output_path: str = None) -> Dict[str, any]:
    """
    종합적인 데이터 품질 보고서 생성
    """
    logger.info("Generating comprehensive data quality report")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'exposure_validation': validate_monthly_continuity(exposure_df),
        'data_overview': {
            'total_companies': exposure_df['company_id'].nunique(),
            'date_range': {
                'start': exposure_df['month'].min(),
                'end': exposure_df['month'].max()
            },
            'total_records': len(exposure_df)
        }
    }
    
    if pnl_df is not None and not pnl_df.empty:
        report['maturity_validation'] = validate_forward_maturity_consistency(pnl_df)
        report['pnl_overview'] = {
            'total_trades': len(pnl_df),
            'companies_with_trades': pnl_df['company_id'].nunique() if 'company_id' in pnl_df.columns else 0
        }
    
    # 보고서 저장
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            # JSON 직렬화를 위한 처리
            report_json = report.copy()
            if 'data_overview' in report_json:
                report_json['data_overview']['date_range']['start'] = str(report_json['data_overview']['date_range']['start'])
                report_json['data_overview']['date_range']['end'] = str(report_json['data_overview']['date_range']['end'])
            
            json.dump(report_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Data quality report saved to: {output_path}")
    
    return report


def recommend_data_fixes(validation_report: Dict[str, any]) -> List[str]:
    """
    데이터 품질 문제에 대한 수정 권고사항 생성
    """
    recommendations = []
    
    # 노출 데이터 연속성 문제
    exposure_val = validation_report.get('exposure_validation', {})
    if exposure_val.get('has_gaps', False):
        gap_pct = exposure_val.get('summary', {}).get('gap_percentage', 0)
        if gap_pct > 50:
            recommendations.append("CRITICAL: Over 50% of companies have monthly data gaps. Consider data collection review.")
        elif gap_pct > 20:
            recommendations.append("WARNING: Significant number of companies have monthly gaps. Consider gap filling strategy.")
        else:
            recommendations.append("INFO: Minor monthly gaps detected. Apply zero-fill or forward-fill as appropriate.")
    
    # 만기 일관성 문제
    maturity_val = validation_report.get('maturity_validation', {})
    if maturity_val.get('has_inconsistencies', False):
        inconsistency_pct = maturity_val.get('inconsistency_percentage', 0)
        max_days = maturity_val.get('maturity_stats', {}).get('max_days', 0)
        
        if inconsistency_pct > 30:
            recommendations.append("CRITICAL: High percentage of trades with inconsistent maturity. Review forward pricing model.")
        elif max_days > 90:
            recommendations.append("WARNING: Some trades have maturity >90 days. Consider term-structure pricing.")
        else:
            recommendations.append("INFO: Minor maturity inconsistencies. Monitor forward pricing accuracy.")
    
    # 일반적인 권고사항
    if not recommendations:
        recommendations.append("GOOD: No significant data quality issues detected.")
    
    return recommendations