import time
import polars as pl
import random
from typing import List

def create_large_sample_data(num_rows: int = 1_000_000) -> pl.DataFrame:
    """100만 건의 샘플 데이터를 생성합니다."""
    print(f"\n📊 샘플 데이터 생성: {num_rows:,}개")
    data = {
        "open": [50000 + random.randint(-500, 500) for _ in range(num_rows)],
        "high": [55000 + random.randint(-500, 500) for _ in range(num_rows)],
        "low": [45000 + random.randint(-500, 500) for _ in range(num_rows)],
        "close": [50000 + random.randint(-500, 500) for _ in range(num_rows)],
    }
    return pl.DataFrame(data)

def test_dataframe_iteration(df: pl.DataFrame) -> (List[float], float):
    """
    방법 1: Polars DataFrame을 직접 순회 (iter_rows 사용)
    - 매 반복마다 Polars 엔진에서 Python으로 데이터를 가져오는 오버헤드가 발생합니다.
    """
    print("🚀 테스트 1: Polars DataFrame 직접 순회")
    results = []
    start_time = time.time()
    
    # iter_rows()는 행을 dict 형태로 반환하는 공식적인 반복자입니다.
    for row_dict in df.iter_rows(named=True):
        avg_price = (row_dict["open"] + row_dict["high"] + row_dict["low"] + row_dict["close"]) / 4
        results.append(avg_price)
        
    elapsed = time.time() - start_time
    print(f"   - 순회 시간: {elapsed:.4f}초")
    return results, elapsed

def test_list_dict_iteration(df: pl.DataFrame) -> (List[float], float):
    """
    방법 2: List[Dict]로 변환 후 순회
    - to_dicts() 변환에 초기 비용이 발생합니다.
    - 변환 후에는 순수 Python 객체를 다루므로 순회 속도가 빠릅니다.
    """
    print("🚚 테스트 2: List[Dict] 변환 후 순회")
    
    # 1. to_dicts() 변환 시간 측정
    convert_start = time.time()
    list_data = df.to_dicts()
    convert_time = time.time() - convert_start
    print(f"   - 변환 시간 (to_dicts): {convert_time:.4f}초")

    # 2. List[Dict] 순회하며 계산 시간 측정
    iter_start = time.time()
    results = []
    for row_dict in list_data:
        avg_price = (row_dict["open"] + row_dict["high"] + row_dict["low"] + row_dict["close"]) / 4
        results.append(avg_price)
    iter_time = time.time() - iter_start
    print(f"   - 순회 시간 (Python List): {iter_time:.4f}초")
    
    total_time = convert_time + iter_time
    return results, total_time

def run_comparison():
    """두 순회 방식의 성능 테스트를 실행하고 결과를 비교합니다."""
    print("=" * 60)
    print(" DataFrame 직접 순회 vs List[Dict] 변환 후 순회 성능 비교")
    print("=" * 60)
    
    df = create_large_sample_data()
    
    print("-" * 60)
    df_results, df_time = test_dataframe_iteration(df)
    print("-" * 60)
    list_results, list_time = test_list_dict_iteration(df)
    
    print("=" * 60)
    print("\n📈 최종 성능 비교")
    print("-" * 50)
    print(f"  - DataFrame 직접 순회:  {df_time:.4f} 초")
    print(f"  - List[Dict] 총 시간:    {list_time:.4f} 초 (변환 + 순회)")
    print("-" * 50)

    if df_time < list_time:
        speed_diff = list_time / df_time
        print(f"\n🏆 결론: DataFrame 직접 순회가 {speed_diff:.2f}배 더 빠릅니다.")
    else:
        speed_diff = df_time / list_time
        print(f"\n🏆 결론: List[Dict] 변환 후 순회가 {speed_diff:.2f}배 더 빠릅니다.")

if __name__ == "__main__":
    run_comparison()