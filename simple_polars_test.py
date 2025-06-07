import polars as pl
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

def polars_worker(data):
    print(f'워커에서 받은 데이터 타입: {type(data)}')
    print(f'데이터 크기: {len(data)}')
    return len(data)

# 테스트 데이터
data = pl.DataFrame({
    'timestamp': [datetime.now() - timedelta(days=i) for i in range(5)],
    'symbol': ['KRW-BTC'] * 5,
    'value': [i for i in range(5)]
})

print('원본 데이터:')
print(data)

# 멀티프로세싱 테스트
try:
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(polars_worker, data)
        result = future.result()
        print(f'결과: {result}')
    print('✅ Polars 직렬화 성공')
except Exception as e:
    print(f'❌ Polars 직렬화 실패: {e}')
    import traceback
    traceback.print_exc() 