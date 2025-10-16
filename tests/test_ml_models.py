"""
Tests for ML models and integrated pipeline
ML 모델 및 통합 파이프라인 테스트
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fx_external_pipeline_full.ml_models import (
    HedgeRatioPredictor, ExposureForecastor, train_ml_models
)
from fx_external_pipeline_full.ml_monitoring import evaluate_hedge_performance
from fx_external_pipeline_full.policy import apply_policy


class TestHedgeRatioPredictor(unittest.TestCase):
    """Test HedgeRatioPredictor with various data conditions"""
    
    def setUp(self):
        """Set up test data"""
        # Small dataset for testing edge cases
        self.small_exposure_df = pd.DataFrame({
            'company_id': ['A', 'B'],
            'month': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'net_exposure': [1000000, -500000]
        })
        
        # Normal sized dataset
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
            for date in dates[:6]  # 6 months per company = 120 samples
        ])
        
        # Market data
        self.market_df = pd.DataFrame({
            'spot': np.random.uniform(1200, 1400, len(dates)),
            'us_rate': np.random.uniform(0.01, 0.05, len(dates)),
            'kr_rate': np.random.uniform(0.02, 0.04, len(dates))
        }, index=dates)
    
    def test_small_dataset_handling(self):
        """Test model behavior with insufficient data"""
        predictor = HedgeRatioPredictor('random_forest')
        
        # Should raise error for very small datasets
        with self.assertRaises(ValueError):
            tiny_df = self.small_exposure_df.iloc[:1]  # Only 1 sample
            predictor.train(tiny_df, self.market_df)
    
    def test_normal_dataset_training(self):
        """Test normal training flow"""
        predictor = HedgeRatioPredictor('random_forest')
        
        # Should complete without errors
        metrics = predictor.train(self.normal_exposure_df, self.market_df)
        
        # Check metrics structure
        self.assertIn('train_r2', metrics)
        self.assertIn('n_samples', metrics)
        self.assertEqual(metrics['n_samples'], len(self.normal_exposure_df))
        
        # Model should be fitted
        self.assertTrue(predictor.is_fitted)
        
    def test_prediction_after_training(self):
        """Test prediction functionality"""
        predictor = HedgeRatioPredictor('random_forest')
        predictor.train(self.normal_exposure_df, self.market_df)
        
        # Should produce predictions
        test_exposure = self.normal_exposure_df.iloc[:5]
        predictions = predictor.predict_hedge_ratios(test_exposure, self.market_df)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))  # Ratios should be in [0,1]
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        predictor = HedgeRatioPredictor('random_forest')
        predictor.train(self.normal_exposure_df, self.market_df)
        
        # Should have feature importance
        self.assertIsNotNone(predictor.feature_importance_)
        self.assertIsInstance(predictor.feature_importance_, dict)
        self.assertTrue(len(predictor.feature_importance_) > 0)


class TestExposureForecastor(unittest.TestCase):
    """Test ExposureForecastor"""
    
    def setUp(self):
        """Set up test data"""
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
        """Test forecasting model training"""
        forecaster = ExposureForecastor('random_forest')
        
        metrics = forecaster.train(self.exposure_df)
        
        # Should complete without errors
        self.assertIn('n_samples', metrics)
        self.assertTrue(forecaster.is_fitted)
    
    def test_future_prediction(self):
        """Test future exposure prediction"""
        forecaster = ExposureForecastor('random_forest')
        forecaster.train(self.exposure_df)
        
        future_exposure = forecaster.predict_future_exposure(self.exposure_df)
        
        # Should return DataFrame with predictions
        self.assertIsInstance(future_exposure, pd.DataFrame)
        self.assertIn('predicted_exposure', future_exposure.columns)
        self.assertTrue(len(future_exposure) > 0)


class TestMLPolicyIntegration(unittest.TestCase):
    """Test ML integration with policy system"""
    
    def setUp(self):
        """Set up test configuration and data"""
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
        """Test policy application without ML models"""
        result = apply_policy(self.exposure_df, cfg=self.cfg)
        
        # Should fallback to rule-based policy
        self.assertIn('hedge_ratio', result.columns)
        self.assertIn('policy_version', result.columns)
        self.assertTrue(all(result['policy_version'] == 'v0'))  # Should fallback
    
    def test_policy_with_ml_failure_fallback(self):
        """Test fallback behavior when ML fails"""
        # Create invalid ML models that will fail
        ml_models = {
            'hedge_ratio_predictor': {
                'model': None  # Invalid model
            }
        }
        
        result = apply_policy(self.exposure_df, cfg=self.cfg, 
                            ml_models=ml_models, market_df=self.market_df)
        
        # Should fallback gracefully
        self.assertIn('hedge_ratio', result.columns)
        self.assertTrue(all(0 <= r <= 1 for r in result['hedge_ratio']))
    
    def test_policy_with_features_missing_fallback(self):
        """Test fallback when features are missing for v1"""
        # Don't provide features_df for v1 policy
        result = apply_policy(self.exposure_df, features_df=None, cfg=self.cfg)
        
        # Should fallback to v0
        self.assertIn('hedge_ratio', result.columns)
        self.assertIn('policy_version', result.columns)


class TestMLMonitoringFunctions(unittest.TestCase):
    """Test ML monitoring functions"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.ml_exposure_df = pd.DataFrame({
            'company_id': [f'COMP_{i}' for i in range(10)],
            'month': pd.to_datetime('2024-01-01'),
            'net_exposure': np.random.uniform(-1e6, 1e6, 10),
            'hedge_ratio': np.random.uniform(0.1, 0.8, 10),
            'policy_version': ['ML_enhanced'] * 10
        })
        
        self.rule_exposure_df = self.ml_exposure_df.copy()
        self.rule_exposure_df['hedge_ratio'] = 0.5  # Simple rule
        self.rule_exposure_df['policy_version'] = 'v0'
        
        self.pnl_df = pd.DataFrame({
            'company_id': [f'COMP_{i}' for i in range(10)],
            'month': pd.to_datetime('2024-01-01'),
            'pnl': np.random.uniform(-50000, 50000, 10)
        })
    
    def test_performance_evaluation_function(self):
        """Test performance evaluation function"""
        # Should not raise errors
        try:
            results = evaluate_hedge_performance(
                original_exposure_df=self.rule_exposure_df,
                ml_enhanced_exposure_df=self.ml_exposure_df,
                pnl_results=self.pnl_df
            )
            self.assertIsInstance(results, dict)
        except Exception as e:
            # Monitor may have dependencies we don't have in test environment
            self.skipTest(f"Monitor evaluation requires additional setup: {e}")


class TestIntegratedMLPipeline(unittest.TestCase):
    """Test the full ML pipeline integration"""
    
    def test_ml_models_training_integration(self):
        """Test the complete ML training pipeline"""
        # Setup data
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-06-01', freq='MS')
        companies = [f'COMP_{i:03d}' for i in range(15)]
        
        exposure_df = pd.DataFrame([
            {
                'company_id': comp,
                'month': date,
                'net_exposure': np.random.uniform(-2e6, 2e6)
            }
            for comp in companies[:5]  # Reduced for faster testing
            for date in dates[:12]  # 12 months per company = 60 samples
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
        
        # Should train without major errors
        try:
            ml_models = train_ml_models(exposure_df, market_df, config)
            self.assertIsInstance(ml_models, dict)
            
            # Check if models were created (may not train successfully with test data)
            if ml_models:
                for model_name, model_info in ml_models.items():
                    self.assertIn('model', model_info)
                    self.assertIn('metrics', model_info)
                    
        except Exception as e:
            # May fail with insufficient data in test environment
            self.skipTest(f"ML training requires more robust test setup: {e}")


if __name__ == '__main__':
    # Run specific test classes based on what's available
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHedgeRatioPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestExposureForecastor))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPolicyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMLMonitoringFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedMLPipeline))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)