"""
병렬 처리 디버깅 테스트
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import sys
import os

# 시스템 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from quantbt.infrastructure.data_provider.csv_data_provider import CSVDataProvider
    from quantbt.infrastructure.broker.simple_broker import SimpleBroker
    from quantbt.examples.strategies.sma_grid_strategy import SMAGridStrategy
    from quantbt.core.value_objects.backtest_config import BacktestConfig
except ImportError as e:
    print(f"Import 오류: {e}")
    print("직접 모듈 사용")
    import polars as pl


def simple_worker_test(test_data):
    """간단한 워커 테스트"""
    print(f"워커 프로세스: {mp.current_process().name}")
    print(f"테스트 데이터: {test_data}")
    return {"status": "success", "data": test_data}


def test_polars_serialization():
    """Polars DataFrame 직렬화 테스트"""
    import polars as pl
    
    # 샘플 데이터 생성
    data = pl.DataFrame({
        "timestamp": [datetime.now() - timedelta(days=i) for i in range(5)],
        "symbol": ["KRW-BTC"] * 5,
        "open": [50000.0 + i*1000 for i in range(5)],
        "high": [51000.0 + i*1000 for i in range(5)],
        "low": [49000.0 + i*1000 for i in range(5)],
        "close": [50500.0 + i*1000 for i in range(5)],
        "volume": [100.0 + i*10 for i in range(5)]
    })
    
    print("원본 데이터:")
    print(data)
    
    # 멀티프로세싱으로 전달 테스트
    with ProcessPoolExecutor(max_workers=2) as executor:
        future = executor.submit(polars_worker, data)
        result = future.result()
        print("멀티프로세싱 결과:")
        print(result)


def polars_worker(data):
    """Polars 데이터를 받는 워커"""
    print(f"워커에서 받은 데이터 타입: {type(data)}")
    print(f"데이터 크기: {len(data)}")
    return data.head(3)


async def main():
    print("🔍 병렬 처리 디버깅 테스트 시작")
    
    # 1. 기본 멀티프로세싱 테스트
    print("\n1. 기본 멀티프로세싱 테스트")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(simple_worker_test, f"test_{i}") for i in range(3)]
        for future in futures:
            result = future.result()
            print(f"결과: {result}")
    
    # 2. Polars DataFrame 직렬화 테스트
    print("\n2. Polars DataFrame 직렬화 테스트")
    try:
        test_polars_serialization()
        print("✅ Polars 직렬화 성공")
    except Exception as e:
        print(f"❌ Polars 직렬화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 데이터 로드 테스트
    print("\n3. 데이터 로드 테스트")
    try:
        data_provider = CSVDataProvider("./data")
        data = await data_provider.get_data(
            symbols=["KRW-BTC"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 31),
            timeframe="1d"
        )
        print(f"✅ 데이터 로드 성공: {len(data)} 행")
        print(f"데이터 타입: {type(data)}")
        print(f"컬럼: {data.columns}")
        
        # 데이터 직렬화 테스트
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(polars_worker, data)
            result = future.result()
            print("✅ 데이터 직렬화 성공")
            
    except Exception as e:
        print(f"❌ 데이터 로드/직렬화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 디버깅 테스트 완료")


if __name__ == "__main__":
    asyncio.run(main()) 