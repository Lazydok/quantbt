"""
업비트 데이터 프로바이더 사용 예제

업비트 API를 활용한 멀티심볼 백테스팅 예제입니다.
캐싱 기능을 활용하여 효율적으로 데이터를 관리합니다.
"""

import asyncio
from datetime import datetime, timedelta
import polars as pl

from quantbt import (
    UpbitDataProvider,
    SimpleBacktestEngine,
    SimpleBroker,
    SimpleMovingAverageCrossStrategy,
    BacktestConfig,
    BuyAndHoldStrategy,
    RSIStrategy
)


async def upbit_multi_symbol_example():
    """업비트 멀티심볼 백테스팅 예제"""
    print("=== 업비트 멀티심볼 백테스팅 예제 ===\n")
    
    # 1. 업비트 데이터 프로바이더 생성
    print("1. 업비트 데이터 프로바이더 설정")
    upbit_provider = UpbitDataProvider(
        cache_dir="./data/upbit_cache",
        rate_limit_delay=0.1,  # API 요청 간격
        max_candles_per_request=200
    )
    
    # 사용 가능한 심볼 확인
    symbols = upbit_provider.get_symbols()
    print(f"사용 가능한 심볼 수: {len(symbols)}")
    print(f"주요 심볼들: {symbols[:10]}")
    
    # 2. 백테스팅 설정
    print("\n2. 백테스팅 설정")
    
    # 테스트할 주요 암호화폐 선택
    test_symbols = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOT"]
    print(f"테스트 심볼: {test_symbols}")
    
    # 날짜 범위 설정 (최근 30일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    config = BacktestConfig(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=10000000.0,  # 1천만원
        timeframe="1h",  # 1시간봉
        commission_rate=0.0005,  # 0.05% 수수료
        slippage_rate=0.0001
    )
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"시간프레임: {config.timeframe}")
    print(f"초기 자금: {config.initial_cash:,.0f}원")
    
    # 3. 데이터 로드 테스트
    print("\n3. 데이터 로드 테스트")
    
    async with upbit_provider:
        try:
            # 데이터 조회
            data = await upbit_provider.get_data(
                symbols=test_symbols,
                start=start_date,
                end=end_date,
                timeframe=config.timeframe
            )
            
            print(f"로드된 데이터 행수: {data.height}")
            print(f"데이터 컬럼: {data.columns}")
            
            # 심볼별 데이터 개수 확인
            symbol_counts = data.group_by("symbol").agg(pl.count().alias("count"))
            print("\n심볼별 데이터 개수:")
            for row in symbol_counts.iter_rows(named=True):
                print(f"  {row['symbol']}: {row['count']}개")
            
            # 데이터 품질 확인
            if data.height > 0:
                print(f"\n데이터 범위:")
                print(f"  시작: {data.select('timestamp').min().item()}")
                print(f"  종료: {data.select('timestamp').max().item()}")
                
                # 샘플 데이터 출력
                print(f"\n샘플 데이터 (최근 5개):")
                sample_data = data.tail(5)
                print(sample_data)
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return
    
    # 4. 캐시 정보 확인
    print("\n4. 캐시 정보")
    cache_info = upbit_provider.get_cache_info()
    print(f"캐시 디렉토리: {cache_info['cache_dir']}")
    print(f"캐시 파일 수: {cache_info['cache_files_count']}")
    print(f"캐시 크기: {cache_info['cache_size_mb']:.2f} MB")
    
    # 5. 백테스팅 실행 (이동평균 교차 전략)
    print("\n5. 이동평균 교차 전략 백테스팅")
    
    # 컴포넌트 생성
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )
    
    strategy = SimpleMovingAverageCrossStrategy(
        short_window=5,   # 단기 이동평균 5시간
        long_window=20    # 장기 이동평균 20시간
    )
    strategy.position_size_pct = 0.15  # 각 종목당 15%
    strategy.max_positions = 4  # 최대 4개 종목
    
    engine = SimpleBacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(upbit_provider)
    engine.set_broker(broker)
    
    try:
        result = await engine.run(config)
        
        print("\n=== 백테스팅 결과 ===")
        result.print_summary()
        
        # 상세 결과 출력
        print(f"\n상세 성과 지표:")
        print(f"  총 수익률: {result.total_return_pct:.2f}%")
        print(f"  연간 수익률: {result.annual_return_pct:.2f}%")
        print(f"  변동성: {result.volatility_pct:.2f}%")
        print(f"  샤프 비율: {result.sharpe_ratio:.2f}")
        print(f"  최대 낙폭: {result.max_drawdown_pct:.2f}%")
        print(f"  총 거래 수: {result.total_trades}")
        print(f"  승률: {result.win_rate_pct:.2f}%")
        
    except Exception as e:
        print(f"백테스팅 실행 실패: {e}")
    
    # 6. RSI 전략으로도 테스트
    print("\n6. RSI 전략 백테스팅")
    
    rsi_strategy = RSIStrategy(
        rsi_period=14,
        oversold=25,      # 과매도 기준 (더 보수적)
        overbought=75     # 과매수 기준 (더 보수적)
    )
    rsi_strategy.position_size_pct = 0.2  # 각 종목당 20%
    rsi_strategy.max_positions = 3  # 최대 3개 종목
    
    engine.set_strategy(rsi_strategy)
    
    try:
        rsi_result = await engine.run(config)
        
        print("\n=== RSI 전략 결과 ===")
        rsi_result.print_summary()
        
    except Exception as e:
        print(f"RSI 전략 실행 실패: {e}")


async def upbit_data_management_example():
    """업비트 데이터 관리 예제"""
    print("=== 업비트 데이터 관리 예제 ===\n")
    
    upbit_provider = UpbitDataProvider(cache_dir="./data/upbit_cache")
    
    # 1. 캐시 정보 확인
    print("1. 현재 캐시 상태:")
    cache_info = upbit_provider.get_cache_info()
    print(f"캐시 디렉토리: {cache_info['cache_dir']}")
    print(f"캐시 파일 수: {cache_info['cache_files_count']}")
    print(f"캐시 크기: {cache_info['cache_size_mb']:.2f} MB")
    
    if cache_info['symbol_cache']:
        print("\n심볼별 캐시:")
        for symbol, timeframes in cache_info['symbol_cache'].items():
            print(f"  {symbol}:")
            for tf, dates in timeframes.items():
                print(f"    {tf}: {len(dates)}일")
    
    # 2. 특정 심볼 데이터 다운로드
    print("\n2. 비트코인 1분봉 데이터 다운로드 (최근 1일)")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    async with upbit_provider:
        btc_data = await upbit_provider.get_data(
            symbols=["KRW-BTC"],
            start=start_date,
            end=end_date,
            timeframe="1m"
        )
        
        print(f"비트코인 1분봉 데이터: {btc_data.height}개")
        if btc_data.height > 0:
            print(f"가격 범위: {btc_data.select('low').min().item():,.0f} ~ {btc_data.select('high').max().item():,.0f}원")
    
    # 3. 캐시 관리
    print("\n3. 캐시 관리")
    print("특정 심볼의 캐시 삭제 예제:")
    
    # 예시: KRW-BTC의 1m 데이터만 삭제 (실제로는 주석 처리)
    # upbit_provider.clear_cache(symbol="KRW-BTC", timeframe="1m")
    
    print("전체 캐시 정리:")
    # upbit_provider.clear_cache()  # 모든 캐시 삭제
    
    print("캐시 관리 완료")


async def main():
    """메인 실행 함수"""
    try:
        # 멀티심볼 백테스팅 예제
        await upbit_multi_symbol_example()
        
        print("\n" + "="*60 + "\n")
        
        # 데이터 관리 예제
        await upbit_data_management_example()
        
    except Exception as e:
        print(f"예제 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main()) 