# 기업 환노출 헤지 분석 파이프라인

기업별 월 수출·수입 금액으로 달러 순노출을 계산하고, 헤지 비율 정책(규칙 기반·트리 모델)을 적용한 1개월 선도환 헤지를 백테스트한 뒤 VaR/ES, IFRS9 헤지 효과성, 기업 군집, 옵션 이론가를 CSV와 그림으로 저장하는 파이썬 배치 파이프라인입니다.

iM뱅크 iM DiGital Banker Academy(데이터 분석 전문가 양성 과정) 6기 5조의 팀 프로젝트입니다(2025년 10월).

## 배경과 목표

수출입 기업은 달러 수입과 지출 시점이 달라 환율 변동에 노출됩니다. 팀이 기획 단계에서 정리한 문제는 중소·중견기업의 헤지 비중이 노출에 비해 낮다는 점, 그리고 은행의 선물환 견적·체결과 IFRS9 헤지회계 효과성 검증·문서화가 수작업에 의존한다는 점이었습니다.

이 프로젝트는 기업별 월 수출입 패널과 시장 지표를 연결해 다음 과정을 한 번에 실행하는 것을 목표로 했습니다.

- 기업·월 단위 순노출(수출 - 수입) 계산
- 노출 규모와 시장 신호에 따른 헤지 비율 제안
- CIP(커버드 금리평가) 선도환율로 월말 체결·익월말 정산 헤지 백테스트
- 손익 기반 VaR/ES, 기업별 미헤지 노출의 VaR/ES와 극단 시나리오 손익
- IFRS9 헤지 효과성(Dollar-offset 비율, 회귀 R²) 판정과 변경 이력 기록
- K-Means/GMM 기업 군집, Garman-Kohlhagen 통화옵션 이론가

## 처리 흐름

```
data/panel_base.csv ──► 월별 순노출 집계 ──► 월 공백 채우기(0)
                                              │
data/external/*.csv ──► 월말 환율·금리 ────────┤
 (또는 FRED/ECOS API)                          ▼
                                    헤지 비율 결정
                                    학습된 ML 모델, 없으면 v1 규칙,
                                    ML 예측 실패·특성 부족 시 v0 규칙
                                              │
                          CIP 선도환율 + 스프레드 ▼
                                    월말 체결 → 익월말 정산 백테스트
                                              │
      ┌──────────────┬──────────────┬─────────┴────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼              ▼              ▼
  기업별 요약     VaR/ES        IFRS9 효과성     K-Means/GMM     ML 성과 리포트   기업별 리스크 표
                                                                 (그림 포함)
```

진입점은 `run_pipeline.py`이고, 계산 모듈은 `src/fx_external_pipeline_full/`에 있습니다.

| 단계 | 모듈 | 내용 |
|---|---|---|
| 노출 | `exposure.py`, `data_quality.py` | 월말 기준 순노출 집계, 기업별 빠진 달 탐지·채우기, 품질 보고서 |
| 시장 데이터 | `external_loader.py`, `fred_loader.py`, `ecos_loader.py`, `holiday_calendar.py` | CSV 또는 FRED·ECOS API, 한국·미국 휴일 캐시 |
| 헤지 정책 | `policy.py`, `features.py`, `ml_models.py`, `policy_tuner.py` | v0 규칙(노출 구간별 비율), v1 규칙(변동성·추세·캐리 신호), 랜덤 포레스트 헤지 비율·노출 예측, v1 임계값 격자 탐색 |
| 가격·백테스트 | `pricing.py`, `backtest.py`, `pricing_option.py` | CIP 선도환율, 스프레드, NDF 정산 손익, 만기 일수 보정, Garman-Kohlhagen |
| 리스크 | `risk.py`, `company_risk_analytics.py` | 역사적 VaR/ES, 기업·월별 미헤지 노출 VaR/ES와 시나리오 손익 |
| 회계·거버넌스 | `hedge_effectiveness.py`, `governance.py` | Dollar-offset·회귀 통계, 분기 표본 규칙, 정책·효과성 스냅샷과 변경 이력 |
| 분석·리포트 | `clustering.py`, `ml_monitoring.py`, `reporting.py` | 군집, ML 지표·특성 중요도·그림, CSV 저장 |

## 기술 스택

Python 3.10 이상(`X | None` 타입 표기 사용), pandas 2.2 이상, NumPy, SciPy, scikit-learn(랜덤 포레스트, K-Means, GMM), XGBoost·LightGBM(선택), matplotlib, holidays, PyYAML, requests, pytest

## 실행 방법

Python 3.13 + pandas 2.3.3 / 3.1.1 환경에서 아래 명령으로 확인했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python validate_inputs.py          # 입력 CSV와 필수 열 확인
python -m pytest -q                # 단위 테스트
python run_pipeline.py             # 결과를 reports/에 저장(기존 파일을 덮어씀)
```

저장소의 결과를 그대로 두고 실행해 보려면 출력 폴더를 바꿉니다. 이 저장소의 데이터로 약 1분 20초가 걸렸습니다(Linux, Python 3.13, pandas 2.3.3).

```bash
python run_pipeline.py --reports_dir /tmp/fx_reports
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config` | `config/config.yml` | 설정 파일 |
| `--panel`, `--spot`, `--kr`, `--us` | `data/` 아래 CSV | 입력 파일 경로 |
| `--reports_dir` | `reports/` | 출력 폴더 |
| `--days` | 30 | 선도 만기(일) |
| `--spread_bps` | 10.0 | 선도환율에 더하는 스프레드(bp) |
| `--clusters` | 설정값 또는 4 | K-Means·GMM 군집 수 |
| `--tune_policy` | 꺼짐 | v1 정책 임계값 격자 탐색 → `policy_tuning_results.csv` |
| `--freeze_policy`, `--freeze_ifrs9` | 꺼짐 | 정책·효과성 스냅샷을 `reports/_artifacts/`에 저장하고 변경 이력 CSV에 한 줄 추가 |
| `--actor`, `--changelog_note`, `--ifrs9_note` | | 변경 이력에 남길 작성자·메모 |
| `--profile` | | 설정 파일 `profiles` 절의 값을 덮어씀(현재 설정에는 `profiles` 절이 없음) |

`bash run_example.sh`는 입력 확인 후 파이프라인을 실행해 `reports/`를 덮어씁니다. 입력 확인이 실패하면 `make_external_templates.py`로 `data/external/`의 합성 시장 데이터를 먼저 다시 만듭니다.

### 시장 데이터를 API로 받기

`config/config.yml`의 `data_source`를 `fred` 또는 `ecos`로 바꾸고 `FRED_API_KEY`, `ECOS_API_KEY` 환경 변수를 설정합니다. 키 설정 여부와 응답은 `python scripts/check_api_keys.py`로 확인할 수 있습니다. 키 값은 출력하지 않고, 요청이 실패했을 때의 오류 메시지에서도 키를 `***`로 가립니다. API 모드는 실제 키로 실행해 확인하지 않았습니다. FRED 응답 처리는 모의 응답 테스트(`tests/test_fred_loader.py`)로 확인했고, ECOS 응답 처리는 테스트가 없습니다.

## 결과

아래 숫자는 저장소에 커밋된 `data/`와 `reports/` 파일에서 읽은 값입니다. **시장 데이터(`data/external/*.csv`)는 실제 환율·금리가 아니라 `make_external_templates.py`가 난수 시드 42로 만든 합성 값**(USD/KRW 약 1,100원대)이므로, 손익·VaR·IFRS9 수치는 파이프라인 동작 확인용이며 실제 시장 성과를 뜻하지 않습니다.

| 항목 | 값 | 출처 |
|---|---|---|
| 입력 패널 | 3,411개 기업, 2022-01 ~ 2024-12, 17,227행 | `data/panel_base.csv` |
| 공백 채운 뒤 기업·월 | 21,441행(빈 달 4,214개를 0으로 추가) | `reports/exposure.csv` |
| 헤지 비율 분포 | 0.0: 4,349 / 0.3: 9,054 / 0.5: 6,498 / 0.8: 1,540 | `reports/exposure.csv` |
| 백테스트 거래 | 정산 13,713건(2,132개 기업), 미체결 3,379건 | `reports/pnl_by_trade.csv`, `reports/summary_company.csv`, `reports/open_positions.csv` |
| IFRS9 월(M) | Dollar-offset 0.998, R² 0.999994 → Effective(기준 0.85~1.15, R² 0.90 이상) | `reports/ifrs9_effectiveness_summary.csv` |
| IFRS9 분기(Q) | Dollar-offset 0.9997, R² 0.999995 → Effective | 같은 파일 |
| 헤지 비율 모델(랜덤 포레스트) | 검증 R² 1.000 | `reports/ml_performance_report_model_metrics.csv` |
| 3개월 뒤 순노출 예측(랜덤 포레스트) | 검증 R² 0.584, MAE 319,829 | 같은 파일, 예측 546개 기업 `reports/future_exposure_forecast.csv` |
| 기업별 리스크 표 | 3,400개 기업·17,092행, 등급 Very Low 17,065 / Low 20 / Medium 7 | `reports/company_risk_analytics.csv` |
| 노출 규모 구분 | XS 14,336 / S 1,216 / M 1,205 / L 198 / XL 137 | 같은 파일 |
| 군집(4개) | K-Means 3,367 / 40 / 1 / 3, GMM 1,554 / 242 / 1,171 / 444 | `reports/cluster_labels.csv` |

그 밖에 `reports/ml_dashboard/`에 특성 중요도, 헤지 비율 비교, 손익 요약 그림이 있고, `reports/options_summary.csv`에 만기 30·90·180일, 행사가 95·100·105%의 콜·풋 이론가가 있습니다.

### 결과를 읽을 때 주의할 점

- **헤지 비율 모델의 R² 1.0은 예측력이 아닙니다.** 학습 타깃이 |순노출| 구간(10만·100만)별 규칙 비율(0.3/0.5/0.8)이고 특성에 노출 크기와 구간이 들어 있어, 모델은 그 규칙을 그대로 다시 배웁니다(`ml_models.py`의 `RULE_RATIO_BY_TIER`).
- **IFRS9 R²가 1에 가까운 이유**: 회귀가 환율 수준과 선도환율 수준 사이에서 이뤄지고, 선도환율은 같은 환율에 (1 + 미국 금리 × 기간) / (1 + 한국 금리 × 기간)과 스프레드를 곱한 값이기 때문입니다.
- `ml_performance_report_evaluation_results.json`의 규칙 기반 평균 헤지 비율(0.051)과 ML 평균(0.336) 차이는 성능 개선이 아니라 기준이 다르기 때문입니다. 비교용 규칙 기반 정책은 `config.yml`의 `policy.v0` 값을 읽지 못하고 코드 기본값(100만·500만, 0/0.5/0.8)을 쓰며, 공백을 채우기 전 패널에 적용됩니다.
- VaR/ES(`reports/var_es_summary.csv`)는 정산월 순으로 정렬한 손익 중 **마지막 252건**으로 계산합니다(252영업일이 아님).

## 팀과 역할

5명이 함께한 교육 과정 팀 프로젝트입니다. 이 저장소의 커밋 기록에는 두 작성자가 있습니다.

- `campus4 D052`: 초기 코드 업로드와 파이프라인·ML 모델·리스크 분석 코드 대부분
- `sumin9716`: README 작성과 파일 정리, 이후 코드 정리와 리팩터링

팀원별 역할 분담은 커밋 기록만으로는 확인할 수 없어 적지 않았습니다.

## 한계

- 시장 데이터가 합성 값이라 손익·리스크·효과성 수치에 실무적 의미가 없습니다.
- 헤지 비율 모델은 규칙을 재현할 뿐이고, 손익을 이용해 최적 비율을 학습하는 부분은 구현되어 있지 않습니다.
- 노출 예측 모델은 같은 기업의 앞뒤 시점이 학습·검증에 섞이는 무작위 분할을 쓰고, 예측 시 `momentum` 특성을 학습 때와 다르게 계산합니다(`ml_models.py` 주석 참고). 예측 월은 `DateOffset`으로 더해 월말이 아닌 날짜(예: 2023-07-30)가 나옵니다.
- `company_risk_analytics.csv`의 `historical_volatility_pct`(7.37)는 월말 환율 수익률에 일별 기준인 √252를 곱해 연환산한 값입니다. 월간 기준(√12)으로 계산하면 약 1.61입니다. 헤지 비율 모델의 `volatility_20d`·`volatility_60d` 특성도 월말 값 20·60개에 √252를 곱합니다.
- `config.yml`의 일부 키(`policy.v0`·`v1`의 임계값·가중치, `ml_models`의 학습 파라미터 등)는 코드가 읽지 않습니다. 파일의 주석에 어떤 키가 쓰이는지 적어 두었습니다.
- 기업별 IFRS9 간이 판정(`ifrs9_company_effectiveness.csv`, 손익 표준편차 ≤ 평균 절댓값 × 2)은 커밋된 `reports/`를 만든 뒤에 추가된 출력이라 `reports/`에는 없고, 파이프라인을 실행하면 생깁니다.
- 데이터 품질 보고서(`data_quality_report.json`)는 공백을 채운 뒤에 만들어 빠진 달이 0으로 나옵니다. 채우기 전 공백 수는 실행 로그에만 출력됩니다.

## 폴더 구조

```
.
├── run_pipeline.py              # 파이프라인 진입점
├── validate_inputs.py           # 입력 CSV 확인
├── make_external_templates.py   # 합성 시장 데이터 생성
├── run_example.sh               # 입력 확인 → 파이프라인 실행
├── config/
│   ├── config.yml               # 데이터 소스, 정책, 리스크, IFRS9, ML 설정
│   └── logging.yml
├── data/
│   ├── panel_base.csv           # 기업·월별 수출입 금액
│   ├── external/                # 월말 USD/KRW, 한국·미국 금리(합성)
│   └── reference/holidays_cache/  # 2018~2025 한국·미국 휴일
├── src/fx_external_pipeline_full/ # 계산 모듈(위 표 참고)
├── scripts/check_api_keys.py    # FRED·ECOS API 키 확인
├── tests/                       # pytest 단위 테스트(합성 데이터)
├── reports/                     # 커밋된 실행 결과
└── docs/DEVELOPMENT.md          # 개발·유지보수 메모
```
