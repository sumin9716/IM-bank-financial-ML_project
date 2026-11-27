# 개발 메모

## 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

- Python 3.10 이상이 필요합니다. Python 3.13에서 pandas 2.3.3과 3.1.1 두 버전으로 테스트와 파이프라인을, Python 3.11(pandas 3.1.1)에서 테스트를 확인했습니다.
- pandas 하한(2.2)은 월말 빈도 별칭 `'ME'` 때문입니다.
- FRED·ECOS API 키는 환경 변수(`FRED_API_KEY`, `ECOS_API_KEY`)로만 넘기고 파일로 커밋하지 않습니다. `.env`는 `.gitignore`에 들어 있습니다.

## 확인 순서

코드를 바꾼 뒤에는 아래를 차례로 실행합니다.

```bash
python -m pytest -q
ruff check --select F,E4,E7,E9,W,B .   # ruff는 requirements에 없으므로 따로 설치
python run_pipeline.py --reports_dir /tmp/fx_reports
```

마지막 명령의 결과를 비교해 의도하지 않은 숫자 변화가 없는지 봅니다. 숫자를 바꾸지 않는 변경이라면 같은 환경에서 변경 전 코드로 만든 결과와 비교하는 것이 가장 정확합니다. 이때 다음 차이는 정상입니다.

- `data_quality_report.json`의 `timestamp`
- 랜덤 포레스트 결과(`future_exposure_forecast.csv`, `ml_performance_report_model_metrics.csv`)와 `ml_performance_report_evaluation_results.json`의 마지막 자릿수. 병렬 합산 순서 등에 따라 부동소수 오차 수준으로 달라집니다.

커밋된 `reports/`와 바로 비교하면 라이브러리 버전 차이 때문에 다음도 달라집니다.

- 손익·요약·IFRS9 회귀 통계의 부동소수 마지막 자릿수(pandas 3에서는 `exposure.csv`도 `4100000.0`이 `4099999.9999999995`가 되는 식으로 달라짐)
- `company_risk_analytics.csv`의 숫자 표기(`810000` 대 `810000.0`)와 `ml_dashboard/` PNG 파일의 바이트
- 노출 예측 지표(검증 R²: 커밋된 결과 0.5837, scikit-learn 1.7.2에서 0.5822, 1.9.1에서 0.5839)

`python run_pipeline.py`를 출력 폴더 없이 실행하면 `reports/`를 덮어씁니다. 결과를 바꾸는 변경일 때만 `reports/`를 다시 만들어 커밋합니다.

## 코드 규칙

- 파일 경로는 작업 디렉터리가 아니라 프로젝트 루트 기준으로 만듭니다(`run_pipeline.PROJECT_ROOT`, `holiday_calendar.DEFAULT_CACHE_DIR`, `company_risk_analytics.DEFAULT_SPOT_FILE`). 설정 파일의 상대 경로는 `resolve_path`로 해석합니다.
- 주석과 docstring은 한국어로, 실행 로그는 `[INFO]`/`[WARN]` 접두어를 붙인 영어로 씁니다.
- 예외는 대체 경로가 있는 곳에서만 잡고 경고를 남깁니다. 파이프라인의 선택 단계(API 일별 환율, 정책 특성, ML 학습·리포트, 기업별 리스크 표)와 헤지 정책에서 ML 예측이 실패하거나 v1 특성이 없을 때 v0 규칙으로 내려가는 부분(`policy.py`)이 여기에 해당합니다. 노출 집계, 백테스트, VaR/ES, IFRS9, 군집 계산에서는 예외를 잡지 않습니다.
- 새 계산 함수에는 `tests/`에 합성 데이터 테스트를 추가합니다. `tests/conftest.py`가 `src/`를 import 경로에 넣습니다.

## 설정 파일에서 주의할 점

`config/config.yml`에는 코드가 읽지 않는 키가 있습니다(파일 안 주석 참고). 특히 `policy.v0`·`policy.v1` 아래의 임계값·비율·가중치·상하한은 적용되지 않고, `policy.py`는 `policy` 바로 아래 키를 읽습니다. 이 값을 실제로 적용하도록 고치면 규칙 기반 헤지 비율과 ML 비교 리포트의 숫자가 바뀌므로, 그때는 `reports/`를 다시 만들고 README의 결과 표도 함께 고쳐야 합니다.

## 알려진 문제

README의 "한계" 절에 정리했습니다. 결과 숫자를 바꾸는 항목(노출 예측의 `momentum` 계산 차이, 예측 월 날짜, 규칙 기반 비교 기준)은 산출물과 README를 함께 고칠 때 처리합니다.
