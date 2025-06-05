import asyncio
from quantbt.infrastructure.data.upbit_provider import UpbitDataProvider
from datetime import datetime
import traceback

async def test_simple():
    provider = UpbitDataProvider(cache_dir='./data/upbit_cache')
    try:
        # 간단한 API 호출 테스트
        async with provider:
            print("API 호출 테스트 시작...")
            data = await provider._fetch_candles_from_api('KRW-BTC', '4h', datetime(2024, 12, 1), datetime(2024, 12, 2))
            print(f'Data height: {data.height}')
            if data.height > 0:
                print('Success!')
                print(data.head())
            else:
                print('No data returned')
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

async def test_preload():
    provider = UpbitDataProvider(cache_dir='./data/upbit_cache')
    try:
        # preload_data 메서드 테스트
        print("preload_data 테스트 시작...")
        result = await provider.preload_data(
            symbols=['KRW-BTC'],
            start=datetime(2024, 12, 1),
            end=datetime(2024, 12, 2),
            timeframe='4h'
        )
        print(f'Result: {result}')
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

def check_cache_info():
    provider = UpbitDataProvider(cache_dir='./data/upbit_cache')
    
    # 캐시 정보 확인
    cache_info = provider.get_cache_info()
    print('=== 캐시 기본 정보 ===')
    print(f'캐시 디렉토리: {cache_info["cache_dir"]}')
    print(f'캐시 파일 수: {cache_info["cache_files_count"]:,}개')
    print(f'총 캐시 크기: {cache_info["cache_size_mb"]:.2f} MB')
    
    # 상세 캐시 정보 확인
    detailed_info = provider.get_cached_data_info()
    print(f'\n=== 상세 캐시 정보 ===')
    print(f'총 파일 수: {detailed_info["total_files"]:,}개')
    print(f'총 크기: {detailed_info["total_size_mb"]:.2f} MB')
    
    print(f'\n=== 심볼별 정보 ===')
    for symbol, timeframes in detailed_info['symbols'].items():
        for tf, info in timeframes.items():
            print(f'{symbol} ({tf}): {info["total_candles"]:,}개 캔들, {info["total_size_mb"]:.2f} MB, {len(info["files"])}개 파일')
            if info['date_range']['min'] and info['date_range']['max']:
                print(f'  데이터 범위: {info["date_range"]["min"].date()} ~ {info["date_range"]["max"].date()}')

if __name__ == "__main__":
    asyncio.run(test_simple())
    asyncio.run(test_preload())
    check_cache_info() 