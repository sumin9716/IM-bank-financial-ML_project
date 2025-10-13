# FX Hedging & Analytics Pipeline

**외환 헤징 전략 분석 및 IFRS9 효과성 테스트를 위한 머신러닝 기반 파이프라인**

## 🎯 프로젝트 개요

이 프로젝트는 금융기관의 외환 위험 관리를 위한 종합적인 분석 도구입니다. 내부 패널 데이터와 외부 경제지표를 결합하여 효과적인 헤징 전략을 수립하고, IFRS9 회계기준에 따른 헤지 효과성을 검증합니다.

### 🔧 주요 기능

- **📊 데이터 통합**: 내부 패널 데이터 + 외부 경제지표 (한국은행, FRED 연계)
- **💰 가격 모델링**: CIP 선도가격, NDF 수익률 계산, Garman-Kohlhagen 옵션 모델
- **🤖 머신러닝**: K-Means/GMM 클러스터링 기반 고객 분석
- **📈 백테스트**: 과거 데이터 기반 헤징 전략 성과 분석
- **⚖️ IFRS9 효과성**: Dollar-Offset, 회귀분석 기반 헤지 효과성 검증
- **📋 리스크 관리**: VaR, Expected Shortfall 계산
- **🎛️ 정책 최적화**: 그리드 서치 기반 헤징 비율 최적화
- **📄 자동 리포팅**: CSV 기반 분석 결과 자동 생성

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 데이터 유효성 검사 (선택사항)
python validate_inputs.py

# 기본 실행 (예제 데이터 사용)
bash run_example.sh
```

### 2. 기본 실행

```bash
# 메인 파이프라인 실행
python run_pipeline.py --config config/config.yml

# 디버그 모드로 실행 (문제 해결시)
python run_pipeline_debug.py
```

### 3. 프로파일 기반 실행

```bash
# 베이스라인 (기본값)
python run_pipeline.py --profile baseline

# 보수적 헤징 전략
python run_pipeline.py --profile conservative

# 공격적 헤징 전략
python run_pipeline.py --profile aggressive
```

## 📁 프로젝트 구조

```
ML_project/
├── config/                    # 설정 파일
│   ├── config.yml            # 메인 파이프라인 설정
│   └── logging.yml           # 로깅 설정
├── data/                     # 데이터 폴더
│   ├── panel_base.csv        # 기본 패널 데이터
│   └── external/             # 외부 데이터
│       ├── spot_usdkrw_eom.csv
│       ├── kr_rates_month.csv
│       └── us_rates_month.csv
├── src/fx_external_pipeline_full/  # 메인 소스코드
│   ├── backtest.py           # 백테스트 엔진
│   ├── clustering.py         # 머신러닝 클러스터링
│   ├── exposure.py           # 노출량 계산
│   ├── policy.py             # 헤징 정책
│   ├── pricing.py            # 가격 모델링
│   ├── pricing_option.py     # 옵션 가격 모델
│   ├── hedge_effectiveness.py # IFRS9 효과성 테스트
│   ├── risk.py               # 리스크 측정
│   ├── governance.py         # 거버넌스 관리
│   └── ...
├── tests/                    # 단위 테스트
├── reports/                  # 결과 리포트 저장소
├── run_pipeline.py           # 메인 실행 스크립트
├── run_pipeline_debug.py     # 디버그용 파이프라인
├── filter_csv.py            # CSV 파일 필터링 도구
└── validate_inputs.py        # 입력 데이터 유효성 검사
```

## 🛠️ 고급 사용법

### 정책 튜닝

헤징 비율 임계값을 자동으로 최적화합니다:

```bash
python run_pipeline.py --tune_policy
```

### IFRS9 효과성 테스트

헤지 효과성을 검증하고 결과를 저장합니다:

```bash
python run_pipeline.py --freeze_ifrs9 --ifrs9_note "Q3 2024 효과성 테스트"
```

### 거버넌스 관리

정책 변경사항을 추적하고 기록합니다:

```bash
python run_pipeline.py --freeze_policy --changelog_note "새로운 리스크 한도 적용" --actor "김담당자"
```

### 사용자 정의 파라미터

```bash
python run_pipeline.py \
    --days 60 \
    --spread_bps 15.0 \
    --clusters 6 \
    --panel data/custom_panel.csv
```

## 📊 주요 모듈 설명

### 1. 데이터 로딩 (`external_loader.py`)
- **CSV 파일**: 로컬 데이터 파일 읽기
- **FRED API**: 미국 경제 데이터 (금리, 환율)
- **ECOS API**: 한국은행 경제통계 (기준금리 등)

### 2. 노출량 계산 (`exposure.py`)
월별 외환 노출량을 고객별, 통화별로 집계합니다.

### 3. 헤징 정책 (`policy.py`)
- **정적 정책**: 고정 비율 헤징
- **동적 정책**: 변동성, 캐리 기반 조건부 헤징
- **머신러닝 기반**: 고객 클러스터별 차별화된 헤징

### 4. 가격 모델링 (`pricing.py`, `pricing_option.py`)
- **선도 가격**: Covered Interest Parity (CIP) 기반
- **옵션 가격**: Garman-Kohlhagen 모델
- **스프레드 적용**: 시장 유동성 비용 반영

### 5. 백테스트 (`backtest.py`)
과거 데이터를 사용하여 헤징 전략의 성과를 시뮬레이션합니다.

### 6. 리스크 측정 (`risk.py`)
- **VaR (Value at Risk)**: 최대 예상 손실
- **Expected Shortfall**: 극단 상황 예상 손실

### 7. IFRS9 효과성 (`hedge_effectiveness.py`)
- **Dollar Offset**: 헤지 손익과 원 노출 손익의 상쇄 비율
- **회귀분석**: R² 기반 효과성 검증
- **자동 판정**: 80-125% 기준 통과/실패 판정

### 8. 머신러닝 (`clustering.py`)
고객 특성 기반 K-Means/GMM 클러스터링으로 맞춤형 헤징 전략을 수립합니다.

## ⚙️ 설정 파일 (`config.yml`)

### 데이터 소스 설정
```yaml
data_source:
  spot: csv        # csv | fred
  rates_us: csv    # csv | fred | ecos
  rates_kr: csv    # csv | ecos
```

### 헤징 정책 설정
```yaml
policy:
  version: v0              # v0 (기본) | v1 (고급)
  size_thresholds: [0.01, 0.1]    # 노출량 임계값 (데이터에 맞게 조정)
  ratios: [0.3, 0.5, 0.8]         # 헤징 비율 [작은노출, 중간노출, 큰노출]
  features:                        # v1 버전용 고급 기능
    rv_windows: [20, 60]
    ma_windows: [20, 60]
```

### IFRS9 효과성 기준
```yaml
ifrs9:
  dollar_offset_bounds: [0.85, 1.15]
  regression_r2_threshold: 0.90
  sampling: { period: "Q", rule: "prev_business_day" }
```

## 📈 출력 결과

모든 분석 결과는 `reports/` 폴더에 CSV 형태로 저장됩니다:

- **exposure_*.csv**: 노출량 분석 결과
- **policy_*.csv**: 헤징 정책 적용 결과
- **backtest_*.csv**: 백테스트 성과 분석
- **risk_*.csv**: 리스크 측정 결과
- **ifrs9_*.csv**: IFRS9 효과성 테스트 결과
- **clustering_*.csv**: 고객 클러스터링 결과

## 🔧 API 연동 설정

### FRED API (미국 경제데이터)
```bash
export FRED_API_KEY="your_fred_api_key"
```

### ECOS API (한국은행)
```bash
export ECOS_API_KEY="your_ecos_api_key"
```

## 🧪 테스트 실행

```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 모듈 테스트
python -m pytest tests/test_pricing.py
python -m pytest tests/test_hedge_effectiveness.py
```

## ⚠️ 중요 사항

### 데이터 설정
- 현재 프로젝트는 작은 규모의 샘플 데이터에 최적화되어 있습니다
- 실제 운영 데이터 사용시 `config.yml`의 `policy.size_thresholds` 값을 적절히 조정하세요
- 헤징 정책 버전은 `v0` (단순) 또는 `v1` (고급) 중 선택 가능합니다

### 문제 해결
```bash
# 파이프라인 실행 중 오류 발생시
python run_pipeline_debug.py  # 상세 디버그 정보 확인

# 데이터 파일 검증
python validate_inputs.py     # 필수 데이터 파일 존재 확인

# CSV 파일 처리 (필요시)
python filter_csv.py         # 특정 컬럼만 추출
```

## 📋 요구사항

- Python 3.8+
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- PyYAML >= 6.0
- scipy >= 1.10
- holidays >= 0.34

## 🚨 업데이트 로그

### v2.0 (2024-10-14)
- **파이프라인 구조 전면 개선**: 전체 코드 구조 재정리 및 안정성 향상
- **디버그 기능 추가**: `run_pipeline_debug.py`로 상세한 실행 과정 추적 가능
- **설정 최적화**: 작은 규모 데이터에 맞는 정책 임계값 자동 조정
- **오류 수정**: `holiday_calendar.py`에 누락된 함수 추가
- **도구 추가**: CSV 필터링 및 데이터 검증 유틸리티 포함

### 주요 개선사항
- ✅ 파이프라인 실행 안정성 대폭 향상
- ✅ 에러 처리 및 디버깅 기능 강화  
- ✅ 실제 데이터 특성에 맞는 설정 자동화
- ✅ 사용자 친화적 문서화 및 가이드 제공

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의사항

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 남겨주세요.