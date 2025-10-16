"""
ML Model Performance Monitoring and Reporting
머신러닝 모델 성과 모니터링 및 리포팅
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def evaluate_hedge_performance(original_exposure_df: pd.DataFrame, 
                              ml_enhanced_exposure_df: pd.DataFrame,
                              pnl_results: pd.DataFrame) -> Dict[str, Any]:
    """
    ML 기반 헤징 전략과 기존 규칙 기반 전략 성과 비교
    """
    results = {}
    
    try:
        # 헤지 비율 분포 비교
        original_ratios = original_exposure_df['hedge_ratio'].describe()
        ml_ratios = ml_enhanced_exposure_df['hedge_ratio'].describe()
        
        results['hedge_ratio_comparison'] = {
            'original_mean': float(original_ratios['mean']),
            'ml_enhanced_mean': float(ml_ratios['mean']),
            'original_std': float(original_ratios['std']),
            'ml_enhanced_std': float(ml_ratios['std']),
            'improvement_mean': float(ml_ratios['mean'] - original_ratios['mean']),
        }
        
        # PnL 분석 (백테스트 결과 있을 경우)
        if not pnl_results.empty:
            total_pnl = pnl_results['pnl_krw'].sum()
            avg_pnl = pnl_results['pnl_krw'].mean()
            pnl_volatility = pnl_results['pnl_krw'].std()
            
            results['pnl_analysis'] = {
                'total_pnl_krw': float(total_pnl),
                'average_pnl_krw': float(avg_pnl),
                'pnl_volatility': float(pnl_volatility),
                'sharpe_ratio': float(avg_pnl / pnl_volatility) if pnl_volatility > 0 else 0,
                'positive_pnl_ratio': float((pnl_results['pnl_krw'] > 0).mean()),
            }
        
        # 헤지 효율성 지표
        hedge_utilization = (ml_enhanced_exposure_df['hedge_ratio'] > 0).mean()
        avg_hedge_size = ml_enhanced_exposure_df[ml_enhanced_exposure_df['hedge_ratio'] > 0]['hedge_ratio'].mean()
        
        results['hedge_efficiency'] = {
            'hedge_utilization_rate': float(hedge_utilization),
            'average_hedge_ratio_when_hedged': float(avg_hedge_size) if not np.isnan(avg_hedge_size) else 0,
            'companies_with_hedge': int((ml_enhanced_exposure_df['hedge_ratio'] > 0).sum()),
            'total_companies': len(ml_enhanced_exposure_df)
        }
        
    except Exception as e:
        logger.error(f"Error in hedge performance evaluation: {e}")
        results['error'] = str(e)
    
    return results


def generate_ml_feature_importance_report(ml_models: Dict[str, Any]) -> pd.DataFrame:
    """
    ML 모델들의 특성 중요도 리포트 생성
    """
    importance_data = []
    
    for model_name, model_info in ml_models.items():
        if 'feature_importance' in model_info:
            importance = model_info['feature_importance']
            
            for feature, score in importance.items():
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance_score': score,
                    'rank': list(importance.keys()).index(feature) + 1
                })
    
    if importance_data:
        df = pd.DataFrame(importance_data)
        return df.sort_values(['model', 'importance_score'], ascending=[True, False])
    else:
        return pd.DataFrame()


def create_ml_performance_dashboard(ml_models: Dict[str, Any], 
                                  results: Dict[str, Any],
                                  save_dir: str) -> None:
    """
    ML 모델 성과 대시보드 생성 (시각화)
    """
    try:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 1. 특성 중요도 차트
        importance_df = generate_ml_feature_importance_report(ml_models)
        if not importance_df.empty:
            plt.figure(figsize=(12, 8))
            
            # 헤지 비율 예측 모델의 특성 중요도
            hedge_importance = importance_df[importance_df['model'] == 'hedge_ratio_predictor']
            if not hedge_importance.empty:
                plt.subplot(2, 1, 1)
                top_features = hedge_importance.head(10)
                plt.barh(top_features['feature'], top_features['importance_score'])
                plt.title('Hedge Ratio Predictor - Top 10 Feature Importance')
                plt.xlabel('Importance Score')
            
            # 노출량 예측 모델이 있다면
            exposure_importance = importance_df[importance_df['model'] == 'exposure_forecaster']
            if not exposure_importance.empty:
                plt.subplot(2, 1, 2)
                top_features = exposure_importance.head(10)
                plt.barh(top_features['feature'], top_features['importance_score'])
                plt.title('Exposure Forecaster - Top 10 Feature Importance')
                plt.xlabel('Importance Score')
            
            plt.tight_layout()
            plt.savefig(save_path / 'ml_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 헤지 비율 분포 비교
        if 'hedge_ratio_comparison' in results:
            plt.figure(figsize=(10, 6))
            
            comparison = results['hedge_ratio_comparison']
            categories = ['Original', 'ML Enhanced']
            means = [comparison['original_mean'], comparison['ml_enhanced_mean']]
            stds = [comparison['original_std'], comparison['ml_enhanced_std']]
            
            x_pos = np.arange(len(categories))
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Hedge Strategy')
            plt.ylabel('Average Hedge Ratio')
            plt.title('Hedge Ratio Comparison: Rule-based vs ML-enhanced')
            plt.xticks(x_pos, categories)
            
            # 개선 정도 표시
            improvement = comparison['improvement_mean']
            plt.text(0.5, max(means) * 0.9, f'Improvement: {improvement:+.3f}', 
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral'))
            
            plt.savefig(save_path / 'hedge_ratio_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. PnL 분석 (데이터가 있을 경우)
        if 'pnl_analysis' in results:
            pnl_data = results['pnl_analysis']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Total PnL
            ax1.bar(['Total PnL'], [pnl_data['total_pnl_krw']/1e6], color='green' if pnl_data['total_pnl_krw'] > 0 else 'red')
            ax1.set_ylabel('PnL (Million KRW)')
            ax1.set_title('Total PnL')
            
            # Sharpe Ratio
            ax2.bar(['Sharpe Ratio'], [pnl_data['sharpe_ratio']], color='blue')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Risk-Adjusted Return')
            
            # Positive PnL Ratio
            ax3.pie([pnl_data['positive_pnl_ratio'], 1-pnl_data['positive_pnl_ratio']], 
                   labels=['Positive', 'Negative'], autopct='%1.1f%%', colors=['green', 'red'])
            ax3.set_title('PnL Distribution')
            
            # Hedge Efficiency
            efficiency = results.get('hedge_efficiency', {})
            if efficiency:
                hedge_stats = [
                    efficiency.get('hedge_utilization_rate', 0),
                    efficiency.get('average_hedge_ratio_when_hedged', 0)
                ]
                ax4.bar(['Utilization Rate', 'Avg Hedge Ratio'], hedge_stats, color=['orange', 'purple'])
                ax4.set_ylabel('Ratio')
                ax4.set_title('Hedge Efficiency Metrics')
            
            plt.tight_layout()
            plt.savefig(save_path / 'pnl_performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"ML performance dashboard saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to create ML performance dashboard: {e}")


def save_ml_performance_report(ml_models: Dict[str, Any], 
                              evaluation_results: Dict[str, Any],
                              save_path: str) -> None:
    """
    ML 모델 성과 리포트를 CSV와 JSON으로 저장
    """
    try:
        # 1. 모델 메트릭스 저장
        model_metrics = []
        for model_name, model_info in ml_models.items():
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                metrics['model_name'] = model_name
                model_metrics.append(metrics)
        
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics)
            metrics_df.to_csv(f"{save_path}_model_metrics.csv", index=False, encoding="utf-8")
        
        # 2. 특성 중요도 저장
        importance_df = generate_ml_feature_importance_report(ml_models)
        if not importance_df.empty:
            importance_df.to_csv(f"{save_path}_feature_importance.csv", index=False, encoding="utf-8")
        
        # 3. 성과 평가 결과 저장
        import json
        with open(f"{save_path}_evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ML performance reports saved with prefix: {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save ML performance report: {e}")


def monitor_model_drift(current_predictions: np.ndarray, 
                       reference_predictions: np.ndarray,
                       threshold: float = 0.1) -> Dict[str, Any]:
    """
    모델 드리프트 모니터링
    """
    try:
        # 예측값 분포 비교
        current_mean = np.mean(current_predictions)
        reference_mean = np.mean(reference_predictions)
        
        current_std = np.std(current_predictions)
        reference_std = np.std(reference_predictions)
        
        # 드리프트 검출
        mean_drift = abs(current_mean - reference_mean) / reference_mean
        std_drift = abs(current_std - reference_std) / reference_std
        
        drift_detected = mean_drift > threshold or std_drift > threshold
        
        return {
            'drift_detected': drift_detected,
            'mean_drift_ratio': float(mean_drift),
            'std_drift_ratio': float(std_drift),
            'threshold': threshold,
            'current_mean': float(current_mean),
            'reference_mean': float(reference_mean),
            'recommendation': 'Retrain model' if drift_detected else 'Model is stable'
        }
        
    except Exception as e:
        logger.error(f"Model drift monitoring failed: {e}")
        return {'error': str(e)}


# 자동 리포트 생성 함수
def generate_comprehensive_ml_report(ml_models: Dict[str, Any],
                                   exposure_df_original: pd.DataFrame,
                                   exposure_df_enhanced: pd.DataFrame, 
                                   pnl_results: pd.DataFrame,
                                   reports_dir: str) -> None:
    """
    종합적인 ML 성과 리포트 생성
    """
    logger.info("Generating comprehensive ML performance report...")
    
    try:
        # 성과 평가
        evaluation_results = evaluate_hedge_performance(
            exposure_df_original, exposure_df_enhanced, pnl_results
        )
        
        # 리포트 저장
        report_prefix = f"{reports_dir}/ml_performance_report"
        save_ml_performance_report(ml_models, evaluation_results, report_prefix)
        
        # 대시보드 생성
        create_ml_performance_dashboard(ml_models, evaluation_results, f"{reports_dir}/ml_dashboard")
        
        logger.info("ML performance report generation completed")
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive ML report: {e}")