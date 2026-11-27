"""학습한 ML 모델의 지표·특성 중요도와 규칙 기반 대비 헤지 비율 비교를 리포트로 저장한다."""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_hedge_performance(original_exposure_df: pd.DataFrame, 
                              ml_enhanced_exposure_df: pd.DataFrame,
                              pnl_results: pd.DataFrame) -> Dict[str, Any]:
    """규칙 기반과 ML 헤지 비율의 분포, 백테스트 손익, 헤지 사용률을 요약한다."""
    results = {}

    original_ratios = original_exposure_df['hedge_ratio'].describe()
    ml_ratios = ml_enhanced_exposure_df['hedge_ratio'].describe()
    results['hedge_ratio_comparison'] = {
        'original_mean': float(original_ratios['mean']),
        'ml_enhanced_mean': float(ml_ratios['mean']),
        'original_std': float(original_ratios['std']),
        'ml_enhanced_std': float(ml_ratios['std']),
        'improvement_mean': float(ml_ratios['mean'] - original_ratios['mean']),
    }

    if not pnl_results.empty and 'pnl_krw' in pnl_results.columns:
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

    hedged = ml_enhanced_exposure_df['hedge_ratio'] > 0
    avg_hedge_size = ml_enhanced_exposure_df.loc[hedged, 'hedge_ratio'].mean()
    results['hedge_efficiency'] = {
        'hedge_utilization_rate': float(hedged.mean()),
        'average_hedge_ratio_when_hedged': float(avg_hedge_size) if not np.isnan(avg_hedge_size) else 0,
        'companies_with_hedge': int(hedged.sum()),
        'total_companies': len(ml_enhanced_exposure_df)
    }
    return results


def generate_ml_feature_importance_report(ml_models: Dict[str, Any]) -> pd.DataFrame:
    importance_data = []
    for model_name, model_info in ml_models.items():
        importance = model_info.get('feature_importance')
        if importance is None:
            continue
        for rank, (feature, score) in enumerate(importance.items(), start=1):
            importance_data.append({
                'model': model_name,
                'feature': feature,
                'importance_score': score,
                'rank': rank
            })

    if not importance_data:
        return pd.DataFrame()
    return pd.DataFrame(importance_data).sort_values(['model', 'importance_score'], ascending=[True, False])


def create_ml_performance_dashboard(ml_models: Dict[str, Any], 
                                  results: Dict[str, Any],
                                  save_dir: str) -> None:
    """특성 중요도, 헤지 비율 비교, 손익 요약 그림을 save_dir에 PNG로 저장한다."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    importance_df = generate_ml_feature_importance_report(ml_models)
    if not importance_df.empty:
        plt.figure(figsize=(12, 8))

        hedge_importance = importance_df[importance_df['model'] == 'hedge_ratio_predictor']
        if not hedge_importance.empty:
            plt.subplot(2, 1, 1)
            top_features = hedge_importance.head(10)
            plt.barh(top_features['feature'], top_features['importance_score'])
            plt.title('Hedge Ratio Predictor - Top 10 Feature Importance')
            plt.xlabel('Importance Score')

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

        improvement = comparison['improvement_mean']
        plt.text(0.5, max(means) * 0.9, f'Improvement: {improvement:+.3f}', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral'))

        plt.savefig(save_path / 'hedge_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    if 'pnl_analysis' in results:
        pnl_data = results['pnl_analysis']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.bar(['Total PnL'], [pnl_data['total_pnl_krw']/1e6], color='green' if pnl_data['total_pnl_krw'] > 0 else 'red')
        ax1.set_ylabel('PnL (Million KRW)')
        ax1.set_title('Total PnL')

        ax2.bar(['Sharpe Ratio'], [pnl_data['sharpe_ratio']], color='blue')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Return')

        ax3.pie([pnl_data['positive_pnl_ratio'], 1-pnl_data['positive_pnl_ratio']], 
               labels=['Positive', 'Negative'], autopct='%1.1f%%', colors=['green', 'red'])
        ax3.set_title('PnL Distribution')

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


def save_ml_performance_report(ml_models: Dict[str, Any], 
                              evaluation_results: Dict[str, Any],
                              save_path: str) -> None:
    """{save_path}_model_metrics.csv, _feature_importance.csv, _evaluation_results.json을 저장한다."""
    model_metrics = [{**info['metrics'], 'model_name': name}
                     for name, info in ml_models.items() if 'metrics' in info]
    if model_metrics:
        pd.DataFrame(model_metrics).to_csv(f"{save_path}_model_metrics.csv", index=False, encoding="utf-8")

    importance_df = generate_ml_feature_importance_report(ml_models)
    if not importance_df.empty:
        importance_df.to_csv(f"{save_path}_feature_importance.csv", index=False, encoding="utf-8")

    with open(f"{save_path}_evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    logger.info(f"ML performance reports saved with prefix: {save_path}")


def generate_comprehensive_ml_report(ml_models: Dict[str, Any],
                                   exposure_df_original: pd.DataFrame,
                                   exposure_df_enhanced: pd.DataFrame, 
                                   pnl_results: pd.DataFrame,
                                   reports_dir: str) -> None:
    logger.info("Generating comprehensive ML performance report...")
    evaluation_results = evaluate_hedge_performance(
        exposure_df_original, exposure_df_enhanced, pnl_results
    )
    save_ml_performance_report(ml_models, evaluation_results, f"{reports_dir}/ml_performance_report")
    create_ml_performance_dashboard(ml_models, evaluation_results, f"{reports_dir}/ml_dashboard")
    logger.info("ML performance report generation completed")
