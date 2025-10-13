#!/usr/bin/env python
import pandas as pd

def filter_csv_columns():
    """
    final_DBSCAN.csv에서 company_id, month, export_amt, import_amt 컬럼만 남기고 나머지 제거
    """
    input_file = r"C:\Users\campus4D052\Desktop\seohyun\final_DBSCAN.csv"
    output_file = r"C:\Users\campus4D052\Desktop\seohyun\final_DBSCAN_filtered.csv"
    
    # 필요한 컬럼들
    required_columns = ['company_id', 'month', 'export_amt', 'import_amt']
    
    print("CSV 파일을 읽는 중...")
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_file)
        
        print(f"원본 파일 정보:")
        print(f"- 행 수: {len(df):,}")
        print(f"- 컬럼 수: {len(df.columns)}")
        print(f"- 원본 컬럼들: {list(df.columns)}")
        
        # 필요한 컬럼들이 존재하는지 확인
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 다음 컬럼들을 찾을 수 없습니다: {missing_columns}")
            return
        
        # 필요한 컬럼들만 선택
        df_filtered = df[required_columns].copy()
        
        print(f"\n필터링된 파일 정보:")
        print(f"- 행 수: {len(df_filtered):,}")
        print(f"- 컬럼 수: {len(df_filtered.columns)}")
        print(f"- 남은 컬럼들: {list(df_filtered.columns)}")
        
        # 데이터 미리보기
        print(f"\n데이터 미리보기:")
        print(df_filtered.head())
        
        # 필터링된 데이터를 새 파일로 저장
        df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 필터링 완료!")
        print(f"📁 저장 위치: {output_file}")
        
        # 파일 크기 비교
        import os
        original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        filtered_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"\n📊 파일 크기 비교:")
        print(f"- 원본: {original_size:.2f} MB")
        print(f"- 필터링 후: {filtered_size:.2f} MB")
        print(f"- 용량 절약: {original_size - filtered_size:.2f} MB ({((original_size - filtered_size) / original_size * 100):.1f}%)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    filter_csv_columns()