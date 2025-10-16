#!/usr/bin/env python
import os, sys
sys.path.insert(0, 'src')

print('🔑 API 키 확인:')
print(f'FRED_API_KEY: {os.getenv("FRED_API_KEY", "NOT_SET")}')
print(f'ECOS_API_KEY: {os.getenv("ECOS_API_KEY", "NOT_SET")}')

print('\n📊 FRED API 테스트:')
try:
    from fx_external_pipeline_full.fred_loader import load_usdkrw_spot_eom_from_fred
    spot_data = load_usdkrw_spot_eom_from_fred('2024-01-01', '2024-03-31')
    print(f'✅ FRED 데이터 로드 성공: {len(spot_data)} 포인트')
    if len(spot_data) > 0:
        print(f'   샘플 데이터: {spot_data.head(2).to_dict()}')
except Exception as e:
    print(f'❌ FRED 오류: {e}')

print('\n📊 ECOS API 테스트:')
try:
    from fx_external_pipeline_full.ecos_loader import load_ecos_series
    kr_data = load_ecos_series('722Y001', 'M', '2024-01', '2024-03', '')
    print(f'✅ ECOS 데이터 로드 성공: {len(kr_data)} 포인트')
    if len(kr_data) > 0:
        print(f'   샘플 데이터: {kr_data.head(2).to_dict()}')
except Exception as e:
    print(f'❌ ECOS 오류: {e}')