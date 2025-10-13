# ML_project — FX Hedging & Analytics (CLI)

- 내부 패널 + 외부지표 결합
- CIP 선도, NDF PnL, 백테스트
- KMeans/GMM, 옵션가격(GK), IFRS9 효과성(Do/회귀)
- CSV 자동 리포트 (`reports/`)

## Quickstart
pip install -r requirements.txt
bash run_example.sh

## Profiles 사용
`config.yml`의 `profiles` 섹션으로 주요 파라미터를 일괄 전환할 수 있습니다.

```bash
# 베이스라인(기본) 프로파일
python run_pipeline.py --config config/config.yml --profile baseline

# 보수적
python run_pipeline.py --config config/config.yml --profile conservative

# 공격적
python run_pipeline.py --config config/config.yml --profile aggressive
```
