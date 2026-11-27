"""트리 기반 모델로 기업·월별 헤지 비율과 몇 달 뒤 순노출을 예측한다."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

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

# 이보다 표본이 적으면 학습·검증 분할이나 교차검증을 하지 않는다
MIN_TRAIN_SAMPLES = 50
MIN_CV_SAMPLES = 25

# 노출 규모 구간(_get_exposure_tier)별 규칙 기반 헤지 비율
RULE_RATIO_BY_TIER = {1: 0.3, 2: 0.5, 3: 0.8}


class HedgeRatioPredictor:
    """노출·시장·계절 특성으로 헤지 비율을 예측한다.

    학습 타깃은 노출 규모 구간별 규칙 비율(RULE_RATIO_BY_TIER)이므로, 모델은 그 규칙을
    특성에서 다시 배운다. 손익을 이용한 최적 비율 학습은 구현되어 있지 않다.
    """

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
        self.is_fitted = False

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
            logger.warning(f"Model {model_type} not available. Using RandomForest.")
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

    def prepare_features(self, exposure_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        features_list = []

        for _, row in exposure_df.iterrows():
            month = row['month']
            net_exposure = row['net_exposure']

            feature_dict = {
                'net_exposure_abs': abs(net_exposure),
                'net_exposure_log': np.log1p(abs(net_exposure)),
                'exposure_direction': 1 if net_exposure > 0 else -1,
                'exposure_size_tier': self._get_exposure_tier(abs(net_exposure)),
            }

            if month in market_df.index:
                market_row = market_df.loc[month]
                feature_dict.update({
                    'usd_krw_level': market_row.get('spot', 1300),
                    'volatility_20d': self._calculate_volatility(market_df, month, 20),
                    'volatility_60d': self._calculate_volatility(market_df, month, 60),
                    'carry_signal': self._calculate_carry_signal(market_df, month),
                    'momentum_signal': self._calculate_momentum(market_df, month),
                })
            else:
                # 시장 데이터가 없는 달은 고정 기본값을 쓴다
                feature_dict.update({
                    'usd_krw_level': 1300,
                    'volatility_20d': 0.015,
                    'volatility_60d': 0.015,
                    'carry_signal': 0.0,
                    'momentum_signal': 0.0,
                })

            feature_dict.update({
                'month_of_year': month.month,
                'quarter': (month.month - 1) // 3 + 1,
                'is_year_end': 1 if month.month == 12 else 0,
            })

            # 군집 결과 열이 없으면 모두 0(단일 군집)으로 둔다
            feature_dict['company_cluster'] = row.get('company_cluster', 0)

            features_list.append(feature_dict)

        return pd.DataFrame(features_list)

    def _get_exposure_tier(self, abs_exposure: float) -> int:
        """|순노출| 100만 이상 3, 10만 이상 2, 그 밖 1."""
        if abs_exposure >= 1000000:
            return 3
        elif abs_exposure >= 100000:
            return 2
        else:
            return 1

    def _calculate_volatility(self, market_df: pd.DataFrame, date: pd.Timestamp, window: int) -> float:
        """date까지 window개 관측치(파이프라인에서는 월말 값)의 환율 수익률 표준편차 × sqrt(252). 계산할 수 없으면 0.015."""
        try:
            end_idx = market_df.index.get_loc(date)
            start_idx = max(0, end_idx - window + 1)
            prices = market_df.iloc[start_idx:end_idx + 1]['spot']

            if len(prices) < 2:
                return 0.015

            returns = prices.pct_change().dropna()
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.015
        except KeyError:
            return 0.015

    def _calculate_carry_signal(self, market_df: pd.DataFrame, date: pd.Timestamp) -> float:
        """한국 금리 - 미국 금리."""
        try:
            if 'us_rate' in market_df.columns and 'kr_rate' in market_df.columns:
                return market_df.loc[date, 'kr_rate'] - market_df.loc[date, 'us_rate']
        except KeyError:
            pass
        return 0.0

    def _calculate_momentum(self, market_df: pd.DataFrame, date: pd.Timestamp) -> float:
        """20개 관측치 전 대비 환율 변화율. 이력이 부족하면 0."""
        try:
            end_idx = market_df.index.get_loc(date)
            if end_idx >= 20:
                current_price = market_df.iloc[end_idx]['spot']
                past_price = market_df.iloc[end_idx - 20]['spot']
                return (current_price - past_price) / past_price
        except KeyError:
            pass
        return 0.0

    def create_target_variable(self, exposure_df: pd.DataFrame) -> np.ndarray:
        return np.array([RULE_RATIO_BY_TIER[self._get_exposure_tier(abs(ne))]
                         for ne in exposure_df['net_exposure']])

    def train(self, exposure_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, float]:
        logger.info(f"Training hedge ratio predictor using {self.model_type}")

        X = self.prepare_features(exposure_df, market_df)
        y = self.create_target_variable(exposure_df)

        if len(X) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Insufficient data for training: {len(X)} < {MIN_TRAIN_SAMPLES}. "
                         f"Using simplified model or fallback to rule-based approach.")
            if len(X) < 10:
                raise ValueError(f"Data too small for any ML training: {len(X)} samples")

            # 표본이 적을 때 과적합을 줄이려고 트리 수와 깊이를 낮춘다
            if hasattr(self.model, 'n_estimators'):
                self.model.set_params(n_estimators=min(10, len(X)//2))
            if hasattr(self.model, 'max_depth'):
                self.model.set_params(max_depth=3)

        if len(X) >= MIN_TRAIN_SAMPLES:
            if 'month' in exposure_df.columns:
                # 행 순서대로 앞 80%를 학습, 뒤 20%를 검증에 쓴다. 월 기준으로 정렬하지 않으므로
                # 시간 순 분할이 아니며, 기업 순으로 정렬된 패널에서는 기업 단위 분할에 가깝다.
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test = X, X.iloc[:0]
            y_train, y_test = y, y[:0]
            logger.warning("Using all data for training due to small dataset size")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_, strict=True))

        y_pred_train = self.model.predict(X_train)

        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'n_samples': len(X),
        }

        if len(X_test) > 0:
            y_pred_test = self.model.predict(X_test)
            metrics.update({
                'test_r2': r2_score(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            })
        else:
            metrics.update({'test_r2': None, 'test_rmse': None})

        if len(X_train) >= MIN_CV_SAMPLES:
            try:
                cv_folds = min(5, len(X_train) // 5)
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
        """[0, 1]로 자르고 소수 셋째 자리에서 반올림한 헤지 비율. 순노출이 0인 행은 0."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X = self.prepare_features(exposure_df, market_df)
        ratios = np.round(np.clip(self.model.predict(X), 0.0, 1.0), 3)

        unique_ratios = np.unique(ratios)
        logger.info(f"ML predicted {len(unique_ratios)} unique hedge ratios. "
                   f"Range: [{ratios.min():.3f}, {ratios.max():.3f}]. "
                   f"Common values: {unique_ratios[:5] if len(unique_ratios) > 5 else unique_ratios}")

        zero_exposure_mask = (exposure_df['net_exposure'].abs() < 1e-6)
        if zero_exposure_mask.any():
            ratios[zero_exposure_mask] = 0.0
            logger.info(f"Set hedge ratio to 0 for {zero_exposure_mask.sum()} companies with zero net exposure")

        return ratios

    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도를 큰 순서로 정렬해 돌려준다."""
        if self.feature_importance_ is None:
            return {}
        return dict(sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True))


class ExposureForecaster:
    """기업별 최근 12개월 순노출 패턴으로 horizon_months 뒤 순노출을 예측한다."""

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
        else:
            logger.warning(f"Model {model_type} not available for exposure forecasting. Using RandomForest.")
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )

    def prepare_features(self, exposure_df: pd.DataFrame,
                        window_months: int = 12) -> pd.DataFrame:
        features_list = []

        for company_id in exposure_df['company_id'].unique():
            company_data = exposure_df[exposure_df['company_id'] == company_id].sort_values('month')

            for i in range(window_months, len(company_data)):
                feature_dict = {'company_id': company_id}

                past_data = company_data.iloc[i-window_months:i]

                feature_dict.update({
                    'avg_exposure_12m': past_data['net_exposure'].mean(),
                    'std_exposure_12m': past_data['net_exposure'].std(),
                    'trend_exposure': self._calculate_trend(past_data['net_exposure']),
                    'seasonal_pattern': self._get_seasonal_pattern(past_data),
                    'volatility_exposure': past_data['net_exposure'].std() / (abs(past_data['net_exposure'].mean()) + 1e-8),
                })

                recent_data = past_data.tail(3)
                feature_dict.update({
                    'recent_avg_3m': recent_data['net_exposure'].mean(),
                    'recent_trend_3m': self._calculate_trend(recent_data['net_exposure']),
                    'momentum': (recent_data['net_exposure'].iloc[-1] - recent_data['net_exposure'].iloc[0]) / 3,
                })

                # 기업 이력의 끝에 가까우면 타깃을 마지막 달로 자른다
                target_idx = min(i + self.horizon_months - 1, len(company_data) - 1)
                feature_dict['target_exposure'] = company_data.iloc[target_idx]['net_exposure']
                feature_dict['target_month'] = company_data.iloc[target_idx]['month']

                features_list.append(feature_dict)

        return pd.DataFrame(features_list)

    def _calculate_trend(self, series: pd.Series) -> float:
        """선형 회귀 기울기."""
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]

    def _get_seasonal_pattern(self, data: pd.DataFrame) -> float:
        """분기별 평균 순노출의 표준편차."""
        if len(data) < 4:
            return 0.0

        data = data.copy()
        data['quarter'] = data['month'].dt.quarter
        quarterly_avg = data.groupby('quarter')['net_exposure'].mean()

        return quarterly_avg.std() if len(quarterly_avg) > 1 else 0.0

    def train(self, exposure_df: pd.DataFrame) -> Dict[str, float]:
        logger.info(f"Training exposure forecaster for {self.horizon_months}-month horizon")

        feature_df = self.prepare_features(exposure_df)

        if len(feature_df) == 0:
            logger.warning("No training data available for exposure forecasting")
            return {}

        X = feature_df.drop(['company_id', 'target_exposure', 'target_month'], axis=1)
        y = feature_df['target_exposure']

        # 행 단위 무작위 분할이라 같은 기업의 앞뒤 시점이 학습·검증에 섞인다
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred_test = self.model.predict(X_test)

        metrics = {
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': np.mean(np.abs(y_test - y_pred_test)),
            'n_samples': len(X),
        }

        logger.info(f"Exposure forecasting training completed. Test R²: {metrics['test_r2']:.3f}")

        return metrics

    def predict_future_exposure(self, exposure_df: pd.DataFrame) -> pd.DataFrame:
        """12개월 이상 이력이 있는 기업마다 마지막 12개월로 horizon_months 뒤 순노출을 예측한다."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        predictions = []

        for company_id in exposure_df['company_id'].unique():
            company_data = exposure_df[exposure_df['company_id'] == company_id].sort_values('month')

            if len(company_data) >= 12:
                recent_data = company_data.tail(12)
                # momentum은 학습 특성(prepare_features)과 달리 2로 나눈다. 기존 산출물과 맞추려고 그대로 둔다.

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
    """설정에서 켠 모델을 학습해 {이름: {'model', 'metrics', ...}}로 돌려준다. 실패한 모델은 빠진다."""
    results = {}

    ml_config = config.get('ml_models', {})

    if not ml_config.get('enabled', False):
        logger.info("ML models disabled in config")
        return results

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

    exposure_config = ml_config.get('exposure_forecasting', {})
    if exposure_config.get('enabled', True):
        logger.info("Training exposure forecasting model...")

        exposure_forecaster = ExposureForecaster(
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
