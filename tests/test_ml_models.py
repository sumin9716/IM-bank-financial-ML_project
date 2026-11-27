"""ML 모델, ML 정책 연동, 성과 요약 함수 테스트."""

import unittest

import numpy as np
import pandas as pd

from fx_external_pipeline_full.ml_models import (
    HedgeRatioPredictor, ExposureForecaster, train_ml_models
)
from fx_external_pipeline_full.ml_monitoring import evaluate_hedge_performance
from fx_external_pipeline_full.policy import apply_policy


class TestHedgeRatioPredictor(unittest.TestCase):
    def setUp(self):
        self.small_exposure_df = pd.DataFrame({
            'company_id': ['A', 'B'],
            'month': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'net_exposure': [1000000, -500000]
        })

        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-12-01', freq='MS')[:36]
        companies = [f'COMP_{i:03d}' for i in range(20)]

        self.normal_exposure_df = pd.DataFrame([
            {
                'company_id': comp,
                'month': date,
                'net_exposure': np.random.uniform(-2e6, 2e6)
            }
            for comp in companies
            for date in dates[:6]  # 기업 20개 × 6개월 = 120행(학습 최소 표본 50 이상)
        ])

        self.market_df = pd.DataFrame({
            'spot': np.random.uniform(1200, 1400, len(dates)),
            'us_rate': np.random.uniform(0.01, 0.05, len(dates)),
            'kr_rate': np.random.uniform(0.02, 0.04, len(dates))
        }, index=dates)

    def test_small_dataset_handling(self):
        predictor = HedgeRatioPredictor('random_forest')

        # 표본이 10개 미만이면 학습하지 않는다
        with self.assertRaises(ValueError):
            tiny_df = self.small_exposure_df.iloc[:1]
            predictor.train(tiny_df, self.market_df)

    def test_normal_dataset_training(self):
        predictor = HedgeRatioPredictor('random_forest')

        metrics = predictor.train(self.normal_exposure_df, self.market_df)

        self.assertIn('train_r2', metrics)
        self.assertIn('n_samples', metrics)
        self.assertEqual(metrics['n_samples'], len(self.normal_exposure_df))
        self.assertTrue(predictor.is_fitted)

    def test_prediction_after_training(self):
        predictor = HedgeRatioPredictor('random_forest')
        predictor.train(self.normal_exposure_df, self.market_df)

        test_exposure = self.normal_exposure_df.iloc[:5]
        predictions = predictor.predict_hedge_ratios(test_exposure, self.market_df)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))

    def test_feature_importance(self):
        predictor = HedgeRatioPredictor('random_forest')
        predictor.train(self.normal_exposure_df, self.market_df)

        self.assertIsNotNone(predictor.feature_importance_)
        self.assertIsInstance(predictor.feature_importance_, dict)
        self.assertTrue(len(predictor.feature_importance_) > 0)


class TestExposureForecaster(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-12-01', freq='MS')
        companies = [f'COMP_{i:03d}' for i in range(10)]

        self.exposure_df = pd.DataFrame([
            {
                'company_id': comp,
                'month': date,
                'net_exposure': np.random.uniform(-1e6, 1e6)
            }
            for comp in companies
            for date in dates
        ])

    def test_forecasting_training(self):
        forecaster = ExposureForecaster('random_forest')

        metrics = forecaster.train(self.exposure_df)

        self.assertIn('n_samples', metrics)
        self.assertTrue(forecaster.is_fitted)

    def test_future_prediction(self):
        forecaster = ExposureForecaster('random_forest')
        forecaster.train(self.exposure_df)

        future_exposure = forecaster.predict_future_exposure(self.exposure_df)

        self.assertIsInstance(future_exposure, pd.DataFrame)
        self.assertIn('predicted_exposure', future_exposure.columns)
        self.assertTrue(len(future_exposure) > 0)


class TestMLPolicyIntegration(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'policy': {
                'version': 'v1',
                'v1': {
                    'fallback_to_v0': True,
                    'weights': {'credit': 0.1, 'size': 0.2, 'relationship': 0.0},
                    'bounds': {'min_ratio': 0.0, 'max_ratio': 0.9}
                },
                'v0': {
                    'size_thresholds': [100000, 1000000],
                    'ratios': [0.3, 0.5, 0.8]
                }
            },
            'ml_models': {
                'hedge_ratio_prediction': {'enabled': True}
            }
        }

        np.random.seed(42)
        self.exposure_df = pd.DataFrame({
            'company_id': [f'COMP_{i}' for i in range(10)],
            'month': pd.to_datetime('2024-01-01'),
            'net_exposure': np.random.uniform(-1e6, 1e6, 10)
        })

        dates = pd.date_range('2024-01-01', '2024-06-01', freq='MS')
        self.market_df = pd.DataFrame({
            'spot': [1300] * len(dates),
            'us_rate': [0.03] * len(dates),
            'kr_rate': [0.025] * len(dates)
        }, index=dates)

    def test_policy_without_ml(self):
        result = apply_policy(self.exposure_df, cfg=self.cfg)

        # v1은 features_df가 필요하므로 v0 규칙으로 내려간다
        self.assertIn('hedge_ratio', result.columns)
        self.assertTrue(all(result['policy_version'] == 'v0'))

    def test_policy_with_ml_failure_fallback(self):
        # 예측할 수 없는 모델을 넣으면 규칙 기반(v0)으로 내려가야 한다
        ml_models = {'hedge_ratio_predictor': {'model': None}}

        result = apply_policy(self.exposure_df, cfg=self.cfg,
                            ml_models=ml_models, market_df=self.market_df)

        self.assertTrue(all(result['policy_version'] == 'v0'))
        self.assertTrue(all(0 <= r <= 1 for r in result['hedge_ratio']))

    def test_policy_v1_without_features_raises_when_fallback_disabled(self):
        cfg = {**self.cfg, 'policy': {**self.cfg['policy'], 'v1': {'fallback_to_v0': False}}}
        with self.assertRaises(ValueError):
            apply_policy(self.exposure_df, features_df=None, cfg=cfg)


class TestMLMonitoringFunctions(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.ml_exposure_df = pd.DataFrame({
            'company_id': [f'COMP_{i}' for i in range(10)],
            'month': pd.to_datetime('2024-01-01'),
            'net_exposure': np.random.uniform(-1e6, 1e6, 10),
            'hedge_ratio': np.random.uniform(0.1, 0.8, 10),
            'policy_version': ['ML_enhanced'] * 10
        })

        self.rule_exposure_df = self.ml_exposure_df.copy()
        self.rule_exposure_df['hedge_ratio'] = 0.5
        self.rule_exposure_df['policy_version'] = 'v0'

        self.pnl_df = pd.DataFrame({
            'company_id': [f'COMP_{i}' for i in range(10)],
            'month': pd.to_datetime('2024-01-01'),
            'pnl_krw': np.random.uniform(-50000, 50000, 10)
        })

    def test_performance_evaluation_function(self):
        results = evaluate_hedge_performance(
            original_exposure_df=self.rule_exposure_df,
            ml_enhanced_exposure_df=self.ml_exposure_df,
            pnl_results=self.pnl_df
        )
        self.assertEqual(set(results), {'hedge_ratio_comparison', 'pnl_analysis', 'hedge_efficiency'})
        self.assertAlmostEqual(results['hedge_ratio_comparison']['original_mean'], 0.5)
        self.assertAlmostEqual(results['pnl_analysis']['total_pnl_krw'], self.pnl_df['pnl_krw'].sum())
        self.assertEqual(results['hedge_efficiency']['companies_with_hedge'], 10)

    def test_performance_evaluation_without_pnl(self):
        results = evaluate_hedge_performance(self.rule_exposure_df, self.ml_exposure_df, pd.DataFrame())
        self.assertNotIn('pnl_analysis', results)


class TestIntegratedMLPipeline(unittest.TestCase):
    def test_ml_models_training_integration(self):
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-06-01', freq='MS')
        companies = [f'COMP_{i:03d}' for i in range(15)]

        exposure_df = pd.DataFrame([
            {
                'company_id': comp,
                'month': date,
                'net_exposure': np.random.uniform(-2e6, 2e6)
            }
            for comp in companies[:5]
            for date in dates[:12]  # 기업 5개 × 12개월 = 60행
        ])

        market_df = pd.DataFrame({
            'spot': np.random.uniform(1200, 1400, len(dates)),
            'us_rate': np.random.uniform(0.01, 0.05, len(dates)),
            'kr_rate': np.random.uniform(0.02, 0.04, len(dates))
        }, index=dates)

        config = {
            'ml_models': {
                'enabled': True,
                'hedge_ratio_prediction': {
                    'enabled': True,
                    'model_type': 'random_forest'
                },
                'exposure_forecasting': {
                    'enabled': True,
                    'model_type': 'random_forest'
                }
            }
        }

        ml_models = train_ml_models(exposure_df, market_df, config)

        self.assertEqual(set(ml_models), {'hedge_ratio_predictor', 'exposure_forecaster'})
        for model_info in ml_models.values():
            self.assertIn('model', model_info)
            self.assertIn('metrics', model_info)


def test_rule_based_target_follows_exposure_tiers():
    df = pd.DataFrame({'net_exposure': [-2_000_000, 1_000_000, 999_999, 100_000, -50_000, 0]})
    target = HedgeRatioPredictor().create_target_variable(df)
    assert list(target) == [0.8, 0.8, 0.5, 0.5, 0.3, 0.3]
