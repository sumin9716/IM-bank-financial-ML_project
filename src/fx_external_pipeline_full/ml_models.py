"""
Advanced ML Models for FX Hedging Optimization
트리 기반 모델을 사용한 헤징 전략 최적화
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


class HedgeRatioPredictor:
    """
    트리 기반 모델로 최적 헤지 비율 예측
    현재의 정적 규칙 기반 → 동적 ML 기반 헤지 비율 결정
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
        self.is_fitted = False
        
        # 모델 초기화
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Fallback to Random Forest
            logger.warning(f"Model {model_type} not available. Using RandomForest.")
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
    
    def prepare_features(self, exposure_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        헤지 비율 예측을 위한 특성 엔지니어링
        """
        features_list = []
        
        for idx, row in exposure_df.iterrows():
            company_id = row['company_id']
            month = row['month']
            net_exposure = row['net_exposure']
            
            # 기본 노출량 특성
            feature_dict = {
                'net_exposure_abs': abs(net_exposure),
                'net_exposure_log': np.log1p(abs(net_exposure)),
                'exposure_direction': 1 if net_exposure > 0 else -1,
                'exposure_size_tier': self._get_exposure_tier(abs(net_exposure)),
            }
            
            # 시계열 특성
            if month in market_df.index:
                market_row = market_df.loc[month]
                feature_dict.update({
                    'usd_krw_level': market_row.get('spot', 1300),  # 환율 수준
                    'volatility_20d': self._calculate_volatility(market_df, month, 20),
                    'volatility_60d': self._calculate_volatility(market_df, month, 60),
                    'carry_signal': self._calculate_carry_signal(market_df, month),
                    'momentum_signal': self._calculate_momentum(market_df, month),
                })
            else:
                # 기본값으로 채움
                feature_dict.update({
                    'usd_krw_level': 1300,
                    'volatility_20d': 0.015,
                    'volatility_60d': 0.015,
                    'carry_signal': 0.0,
                    'momentum_signal': 0.0,
                })
            
            # 계절성 특성
            feature_dict.update({
                'month_of_year': month.month,
                'quarter': (month.month - 1) // 3 + 1,
                'is_year_end': 1 if month.month == 12 else 0,
            })
            
            # 기업별 특성 (클러스터링 결과 활용)
            if hasattr(exposure_df, 'company_cluster'):
                feature_dict['company_cluster'] = row.get('company_cluster', 0)
            else:
                feature_dict['company_cluster'] = 0
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def _get_exposure_tier(self, abs_exposure: float) -> int:
        """노출량 규모별 티어 분류"""
        if abs_exposure >= 1000000:
            return 3  # Large
        elif abs_exposure >= 100000:
            return 2  # Medium
        else:
            return 1  # Small
    
    def _calculate_volatility(self, market_df: pd.DataFrame, date: pd.Timestamp, window: int) -> float:
        """지정 기간 변동성 계산"""
        try:
            end_idx = market_df.index.get_loc(date)
            start_idx = max(0, end_idx - window + 1)
            prices = market_df.iloc[start_idx:end_idx + 1]['spot']
            
            if len(prices) < 2:
                return 0.015  # 기본값
            
            returns = prices.pct_change().dropna()
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.015
        except:
            return 0.015
    
    def _calculate_carry_signal(self, market_df: pd.DataFrame, date: pd.Timestamp) -> float:
        """캐리 트레이드 시그널 계산"""
        try:
            if 'us_rate' in market_df.columns and 'kr_rate' in market_df.columns:
                us_rate = market_df.loc[date, 'us_rate']
                kr_rate = market_df.loc[date, 'kr_rate']
                return kr_rate - us_rate  # 한국금리 - 미국금리
        except:
            pass
        return 0.0
    
    def _calculate_momentum(self, market_df: pd.DataFrame, date: pd.Timestamp) -> float:
        """환율 모멘텀 시그널"""
        try:
            end_idx = market_df.index.get_loc(date)
            if end_idx >= 20:
                current_price = market_df.iloc[end_idx]['spot']
                past_price = market_df.iloc[end_idx - 20]['spot']
                return (current_price - past_price) / past_price
        except:
            pass
        return 0.0
    
    def create_target_variable(self, exposure_df: pd.DataFrame, 
                             pnl_results: pd.DataFrame = None) -> np.ndarray:
        """
        학습용 타겟 변수 생성 (최적 헤지 비율)
        실제 백테스트 결과를 바탕으로 최적 비율 역산
        """
        if pnl_results is not None:
            # 실제 PnL 데이터가 있다면 최적화된 비율 계산
            return self._optimize_from_pnl(exposure_df, pnl_results)
        else:
            # 규칙 기반 헤지 비율 (기존 로직)
            return self._rule_based_ratios(exposure_df)
    
    def _rule_based_ratios(self, exposure_df: pd.DataFrame) -> np.ndarray:
        """기존 규칙 기반 헤지 비율"""
        ratios = []
        for _, row in exposure_df.iterrows():
            abs_exposure = abs(row['net_exposure'])
            if abs_exposure >= 1000000:
                ratio = 0.8
            elif abs_exposure >= 100000:
                ratio = 0.5
            else:
                ratio = 0.3
            ratios.append(ratio)
        return np.array(ratios)
    
    def _optimize_from_pnl(self, exposure_df: pd.DataFrame, 
                          pnl_results: pd.DataFrame) -> np.ndarray:
        """PnL 결과를 바탕으로 최적 비율 역산 (고급 기능)"""
        # 향후 구현: 실제 손익을 바탕으로 최적 비율 학습
        return self._rule_based_ratios(exposure_df)
    
    def train(self, exposure_df: pd.DataFrame, market_df: pd.DataFrame, 
              pnl_results: pd.DataFrame = None) -> Dict[str, float]:
        """
        모델 학습 - 안전장치 포함
        """
        logger.info(f"Training hedge ratio predictor using {self.model_type}")
        
        # 특성 준비
        X = self.prepare_features(exposure_df, market_df)
        y = self.create_target_variable(exposure_df, pnl_results)
        
        # 최소 표본 수 검사
        min_samples = 50  # 최소 50개 관측치 필요
        min_samples_for_cv = 25  # 교차검증을 위한 최소 표본수
        
        if len(X) < min_samples:
            logger.warning(f"Insufficient data for training: {len(X)} < {min_samples}. "
                         f"Using simplified model or fallback to rule-based approach.")
            # 간단한 모델로 학습 또는 에러 반환
            if len(X) < 10:
                raise ValueError(f"Data too small for any ML training: {len(X)} samples")
            
            # 매우 작은 데이터셋용 간단한 설정
            if hasattr(self.model, 'n_estimators'):
                self.model.set_params(n_estimators=min(10, len(X)//2))
            if hasattr(self.model, 'max_depth'):
                self.model.set_params(max_depth=3)
        
        # 데이터 분할 전략 결정
        if len(X) >= min_samples:
            # 충분한 데이터가 있으면 시계열 순서 고려한 분할
            if 'month' in exposure_df.columns:
                # 시계열 분할: 최근 20%를 테스트로
                exposure_df_sorted = exposure_df.sort_values('month')
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                # 일반적인 random 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            # 데이터가 부족하면 전체를 train으로 사용
            X_train, X_test = X, X.iloc[:0]  # 빈 테스트셋
            y_train, y_test = y, y[:0]
            logger.warning("Using all data for training due to small dataset size")
        
        # 모델 학습
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # 특성 중요도 저장
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        
        # 성능 평가
        y_pred_train = self.model.predict(X_train)
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'n_samples': len(X),
        }
        
        # 테스트 성능 (데이터가 충분한 경우만)
        if len(X_test) > 0:
            y_pred_test = self.model.predict(X_test)
            metrics.update({
                'test_r2': r2_score(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            })
        else:
            metrics.update({'test_r2': None, 'test_rmse': None})
        
        # 교차 검증 (충분한 데이터가 있는 경우만)
        if len(X_train) >= min_samples_for_cv:
            try:
                cv_folds = min(5, len(X_train) // 5)  # 동적 폴드 수 결정
                if cv_folds >= 2:
                    cv_scores = cross_val_score(self.model, X_train, y_train, 
                                              cv=cv_folds, scoring='r2')
                    metrics['cv_r2_mean'] = cv_scores.mean()
                    metrics['cv_r2_std'] = cv_scores.std()
                else:
                    metrics['cv_r2_mean'] = None
                    metrics['cv_r2_std'] = None
                    logger.warning("Skipping cross-validation due to insufficient data")
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                metrics['cv_r2_mean'] = None
                metrics['cv_r2_std'] = None
        else:
            metrics['cv_r2_mean'] = None
            metrics['cv_r2_std'] = None
        
        test_r2 = metrics.get('test_r2', metrics['train_r2'])
        logger.info(f"Model training completed. Test R²: {test_r2}")
        
        return metrics
    
    def predict_hedge_ratios(self, exposure_df: pd.DataFrame, 
                           market_df: pd.DataFrame) -> np.ndarray:
        """
        헤지 비율 예측
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(exposure_df, market_df)
        ratios = self.model.predict(X)
        
        # 0-1 범위로 제한
        ratios = np.clip(ratios, 0.0, 1.0)
        
        return ratios
    
    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 반환"""
        if self.feature_importance_ is None:
            return {}
        
        # 중요도 순으로 정렬
        sorted_importance = dict(
            sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_importance
    
    def save_model(self, path: str):
        """모델 저장"""
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance_,
                'is_fitted': self.is_fitted
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_importance_ = data['feature_importance']
        self.is_fitted = data['is_fitted']
        logger.info(f"Model loaded from {path}")


class ExposureForecastor:
    """
    미래 노출량 예측 모델 (선제적 헤징을 위한)
    """
    
    def __init__(self, model_type: str = 'random_forest', horizon_months: int = 3):
        self.model_type = model_type
        self.horizon_months = horizon_months
        self.model = None
        self.is_fitted = False
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )
        elif model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1, 
                random_state=42, n_jobs=-1
            )
    
    def prepare_features(self, exposure_df: pd.DataFrame, 
                        window_months: int = 12) -> pd.DataFrame:
        """
        노출량 예측을 위한 특성 생성
        """
        features_list = []
        
        # 기업별로 시계열 특성 생성
        for company_id in exposure_df['company_id'].unique():
            company_data = exposure_df[exposure_df['company_id'] == company_id].sort_values('month')
            
            for i in range(window_months, len(company_data)):
                feature_dict = {'company_id': company_id}
                
                # 과거 window_months 개월의 노출량 패턴
                past_data = company_data.iloc[i-window_months:i]
                
                feature_dict.update({
                    'avg_exposure_12m': past_data['net_exposure'].mean(),
                    'std_exposure_12m': past_data['net_exposure'].std(),
                    'trend_exposure': self._calculate_trend(past_data['net_exposure']),
                    'seasonal_pattern': self._get_seasonal_pattern(past_data),
                    'volatility_exposure': past_data['net_exposure'].std() / (abs(past_data['net_exposure'].mean()) + 1e-8),
                })
                
                # 최근 3개월 패턴
                recent_data = past_data.tail(3)
                feature_dict.update({
                    'recent_avg_3m': recent_data['net_exposure'].mean(),
                    'recent_trend_3m': self._calculate_trend(recent_data['net_exposure']),
                    'momentum': (recent_data['net_exposure'].iloc[-1] - recent_data['net_exposure'].iloc[0]) / 3,
                })
                
                # 타겟: 미래 노출량
                target_idx = min(i + self.horizon_months - 1, len(company_data) - 1)
                feature_dict['target_exposure'] = company_data.iloc[target_idx]['net_exposure']
                feature_dict['target_month'] = company_data.iloc[target_idx]['month']
                
                features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """시계열 트렌드 계산"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]  # 기울기
    
    def _get_seasonal_pattern(self, data: pd.DataFrame) -> float:
        """계절성 패턴 특성"""
        if len(data) < 4:
            return 0.0
        
        # 분기별 평균 노출량의 변동성
        data = data.copy()  # 경고 방지
        data['quarter'] = data['month'].dt.quarter
        quarterly_avg = data.groupby('quarter')['net_exposure'].mean()
        
        return quarterly_avg.std() if len(quarterly_avg) > 1 else 0.0
    
    def train(self, exposure_df: pd.DataFrame) -> Dict[str, float]:
        """노출량 예측 모델 학습"""
        logger.info(f"Training exposure forecaster for {self.horizon_months}-month horizon")
        
        # 특성 준비
        feature_df = self.prepare_features(exposure_df)
        
        if len(feature_df) == 0:
            logger.warning("No training data available for exposure forecasting")
            return {}
        
        X = feature_df.drop(['company_id', 'target_exposure', 'target_month'], axis=1)
        y = feature_df['target_exposure']
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # 성능 평가
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': np.mean(np.abs(y_test - y_pred_test)),
        }
        
        logger.info(f"Exposure forecasting training completed. Test R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def predict_future_exposure(self, exposure_df: pd.DataFrame) -> pd.DataFrame:
        """미래 노출량 예측"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # 각 기업별 최신 데이터로 예측
        predictions = []
        
        for company_id in exposure_df['company_id'].unique():
            company_data = exposure_df[exposure_df['company_id'] == company_id].sort_values('month')
            
            if len(company_data) >= 12:  # 최소 12개월 데이터 필요
                # 최근 12개월 데이터로 특성 생성
                recent_data = company_data.tail(12)
                
                feature_dict = {
                    'avg_exposure_12m': recent_data['net_exposure'].mean(),
                    'std_exposure_12m': recent_data['net_exposure'].std(),
                    'trend_exposure': self._calculate_trend(recent_data['net_exposure']),
                    'seasonal_pattern': self._get_seasonal_pattern(recent_data),
                    'volatility_exposure': recent_data['net_exposure'].std() / (abs(recent_data['net_exposure'].mean()) + 1e-8),
                    'recent_avg_3m': recent_data.tail(3)['net_exposure'].mean(),
                    'recent_trend_3m': self._calculate_trend(recent_data.tail(3)['net_exposure']),
                    'momentum': (recent_data['net_exposure'].iloc[-1] - recent_data['net_exposure'].iloc[-3]) / 2,
                }
                
                X_pred = pd.DataFrame([feature_dict])
                predicted_exposure = self.model.predict(X_pred)[0]
                
                # 예측 월 계산
                last_month = company_data['month'].max()
                future_month = last_month + pd.DateOffset(months=self.horizon_months)
                
                predictions.append({
                    'company_id': company_id,
                    'predicted_month': future_month,
                    'predicted_exposure': predicted_exposure,
                    'current_exposure': company_data['net_exposure'].iloc[-1],
                    'exposure_change': predicted_exposure - company_data['net_exposure'].iloc[-1]
                })
        
        return pd.DataFrame(predictions)


def train_ml_models(exposure_df: pd.DataFrame, market_df: pd.DataFrame, 
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    전체 ML 모델 학습 파이프라인
    """
    results = {}
    
    # 설정 읽기
    ml_config = config.get('ml_models', {})
    
    if not ml_config.get('enabled', False):
        logger.info("ML models disabled in config")
        return results
    
    # 1. 헤지 비율 예측 모델
    hedge_config = ml_config.get('hedge_ratio_prediction', {})
    if hedge_config.get('enabled', True):
        logger.info("Training hedge ratio prediction model...")
        
        hedge_predictor = HedgeRatioPredictor(
            model_type=hedge_config.get('model_type', 'random_forest')
        )
        
        try:
            metrics = hedge_predictor.train(exposure_df, market_df)
            results['hedge_ratio_predictor'] = {
                'model': hedge_predictor,
                'metrics': metrics,
                'feature_importance': hedge_predictor.get_feature_importance()
            }
        except Exception as e:
            logger.error(f"Failed to train hedge ratio predictor: {e}")
    
    # 2. 노출량 예측 모델
    exposure_config = ml_config.get('exposure_forecasting', {})
    if exposure_config.get('enabled', True):
        logger.info("Training exposure forecasting model...")
        
        exposure_forecaster = ExposureForecastor(
            model_type=exposure_config.get('model_type', 'random_forest'),
            horizon_months=exposure_config.get('horizon_months', 3)
        )
        
        try:
            metrics = exposure_forecaster.train(exposure_df)
            results['exposure_forecaster'] = {
                'model': exposure_forecaster,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Failed to train exposure forecaster: {e}")
    
    return results