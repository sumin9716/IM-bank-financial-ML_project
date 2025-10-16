# 주요 알고리즘 개요

## 1. 월별 환노출 산출
- `compute_monthly_exposure()` 함수는 수출·수입 금액을 월별로 합산한 뒤 순노출(Net Exposure)을 계산한다.
- 데이터는 월말 기준으로 정규화되어 이후 시장 데이터와 정합성을 확보한다.
- 부족한 필수 컬럼이 있을 경우 예외를 발생시켜 데이터 품질을 보장한다. 【F:src/fx_external_pipeline_full/exposure.py†L1-L22】

## 2. 시장 지표 기반 특징량 생성
- `compute_features()`는 일별/월별 환율과 금리를 활용해 변동성(rv20/rv60), 이동평균(ma20/ma60), 금리차 기반 캐리 신호를 월말 인덱스의 피처로 산출한다.
- 일별 데이터가 없으면 월별 데이터로 자동 대체하여 견고성을 유지한다.
- 선도이론가(CIP)를 이용한 캐리 계산으로 금리차 신호를 반영한다. 【F:src/fx_external_pipeline_full/features.py†L1-L63】【F:src/fx_external_pipeline_full/pricing.py†L1-L17】

## 3. 헤지 정책 및 ML 강화 로직
- 기본 정책(`_apply_policy_v0`)은 순노출 규모에 따라 단계별 헤지비율을 배정한다.
- 고급 정책(`_apply_policy_v1`)은 추세·변동성·캐리 신호를 결합해 헤지비율을 상향/하향 조정한다.
- ML이 활성화되면 `HedgeRatioPredictor`가 노출·시장 피처로 예측한 헤지비율을 적용하고, 신용/관계도 등 가중치와 상하한을 통해 최종 조정한다. 【F:src/fx_external_pipeline_full/policy.py†L1-L128】【F:src/fx_external_pipeline_full/ml_models.py†L1-L214】

## 4. 헤지비율 ML 예측 모델
- `HedgeRatioPredictor`는 Random Forest·XGBoost·LightGBM 등을 선택적으로 사용하며 최소 표본 수 검증과 교차검증을 포함한다.
- 노출 규모, 방향, 계절성, 시장 변동성/캐리/모멘텀 등을 입력 특징으로 사용하고, 예측 결과는 0~1 범위로 클리핑 및 소수점 3자리 반올림한다.
- 순노출이 0인 기업은 예측 후에도 헤지비율을 0으로 강제해 불필요한 거래를 방지한다. 【F:src/fx_external_pipeline_full/ml_models.py†L24-L214】

## 5. 선도환 가격 및 손익 산출
- `cip_forward_theoretical()`은 CIP 공식을 이용해 금리차를 반영한 이론 선도가격을 계산한다.
- `apply_spread()`는 호가 스프레드(bps)를 고려하여 매수/매도 방향별 선도가격을 조정한다.
- `ndf_settlement_pnl()`은 만기 고시 환율과 합의 선도가격 차이를 기준으로 NDF 손익을 산출한다. 【F:src/fx_external_pipeline_full/pricing.py†L1-L17】【F:src/fx_external_pipeline_full/pricing.py†L9-L17】

## 6. 월별 선도 헤지 백테스트 엔진
- `monthly_forward_strategy()`는 월말 체결·익월말 결제 구조를 가정하고 영업일 보정을 적용한다.
- 실제 만기일수에 따라 선도가격을 선형 조정하며, 명목금액이 임계값 미만이면 거래를 제외한다.
- 마지막 관측 월의 미체결 포지션을 별도 레코드로 저장하여 익월 손익을 추적한다. 【F:src/fx_external_pipeline_full/backtest.py†L1-L131】

## 7. 리스크 지표 (VaR·ES)
- `historical_var()`는 최근 윈도우(기본 252영업일)의 손익 분포에서 (1-α) 분위수를 산출한다.
- `expected_shortfall()`은 VaR 이하 구간의 평균 손실을 계산하여 Tail Risk를 정량화한다. 【F:src/fx_external_pipeline_full/risk.py†L1-L14】

## 8. IFRS9 헤지 유효성 테스트
- 달러 오프셋 비율과 회귀 R²를 계산해 IFRS9 기준(0.8~1.25, R²≥0.8) 충족 여부를 판정한다.
- 회귀 분석은 기울기, 표준오차, t-통계, p-값을 함께 제공해 감사를 위한 근거를 마련한다.
- 월·분기별 표본 추출 시 영업일/평균/월말 등 규칙을 선택할 수 있도록 지원한다. 【F:src/fx_external_pipeline_full/hedge_effectiveness.py†L1-L111】【F:src/fx_external_pipeline_full/hedge_effectiveness.py†L113-L181】

## 9. 고객 세분화를 위한 클러스터링
- `run_clustering()`은 표준화된 피처를 기반으로 K-Means와 GMM을 동시에 수행해 다각도의 군집 결과를 제공한다.
- 군집별 평균 특성을 요약해 세그먼트별 리스크/헤지 전략 도출에 활용할 수 있다. 【F:src/fx_external_pipeline_full/clustering.py†L1-L18】

## 10. 데이터 품질 검증 및 보정
- `validate_monthly_continuity()`는 기업별 월별 데이터의 결측 기간을 탐지하고 요약 통계를 생성한다.
- `fill_monthly_gaps()`는 월별 연속성을 확보하기 위해 결측 구간을 0·전진채움·보간 방식으로 보정한다.
- `validate_forward_maturity_consistency()`는 선도 만기일 일수의 일관성을 검증해 이상 거래를 조기에 발견한다. 【F:src/fx_external_pipeline_full/data_quality.py†L1-L134】【F:src/fx_external_pipeline_full/data_quality.py†L136-L229】
