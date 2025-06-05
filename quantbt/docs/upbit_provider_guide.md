# 업비트 데이터 프로바이더 가이드

업비트 API를 활용한 실시간 암호화폐 데이터 백테스팅 가이드입니다.

## 🎯 주요 특징

- **실시간 API 연동**: 업비트 API를 통한 최신 분봉 데이터 조회
- **스마트 캐싱**: 한 번 다운로드한 데이터는 로컬에 캐시하여 재사용
- **멀티심볼 지원**: 여러 암호화폐를 동시에 분석
- **비동기 처리**: 효율적인 API 호출과 데이터 처리
- **자동 Rate Limiting**: API 제한에 맞춘 자동 요청 조절

## 🚀 빠른 시작

### 1. 기본 설정

```python
from quantbt import UpbitDataProvider
from datetime import datetime, timedelta

# 업비트 데이터 프로바이더 생성
upbit_provider = UpbitDataProvider(
    cache_dir="./data/upbit_cache",    # 캐시 저장 경로
    rate_limit_delay=0.1,              # API 호출 간격 (초)
    max_candles_per_request=200        # 한 번에 요청할 최대 캔들 수
)
```

### 2. 사용 가능한 심볼 확인

```python
# 업비트에서 거래 가능한 모든 KRW 마켓 조회
symbols = upbit_provider.get_symbols()
print(f"총 {len(symbols)}개 암호화폐 거래 가능")
print(f"주요 코인: {symbols[:10]}")

# 출력 예시:
# 총 245개 암호화폐 거래 가능
# 주요 코인: ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-DOT', ...]
```

### 3. 데이터 조회

```python
import asyncio

async def load_crypto_data():
    # 최근 7일간의 비트코인 1시간봉 데이터
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    async with upbit_provider:
        data = await upbit_provider.get_data(
            symbols=["KRW-BTC"],
            start=start_date,
            end=end_date,
            timeframe="1h"
        )
    
    print(f"로드된 데이터: {data.height}개 캔들")
    print(data.head())

# 실행
asyncio.run(load_crypto_data())
```

## 📊 지원 시간프레임

업비트는 다음 분봉 데이터를 지원합니다:

| 시간프레임 | 설명 | 사용 예 |
|-----------|------|--------|
| `1m` | 1분봉 | 단기 스캘핑 전략 |
| `3m` | 3분봉 | 짧은 텀 매매 |
| `5m` | 5분봉 | 인트라데이 전략 |
| `10m` | 10분봉 | 중단기 신호 |
| `15m` | 15분봉 | 일반적인 단기 전략 |
| `30m` | 30분봉 | 중기 전략 |
| `1h` | 1시간봉 | 일반적인 백테스팅 |
| `4h` | 4시간봉 | 장기 트렌드 분석 |

## 🔄 캐싱 시스템

### 캐시 작동 원리

1. **최초 API 호출**: 데이터가 없으면 업비트 API에서 조회
2. **로컬 저장**: 일별로 분할하여 Parquet 파일로 저장
3. **재사용**: 이후 동일한 데이터 요청 시 캐시에서 조회
4. **자동 업데이트**: 부족한 기간만 API에서 추가 조회

### 캐시 관리

```python
# 캐시 정보 확인
cache_info = upbit_provider.get_cache_info()
print(f"캐시 디렉토리: {cache_info['cache_dir']}")
print(f"캐시 파일 수: {cache_info['cache_files_count']}")
print(f"캐시 크기: {cache_info['cache_size_mb']:.2f} MB")

# 특정 심볼의 캐시 삭제
upbit_provider.clear_cache(symbol="KRW-BTC", timeframe="1h")

# 전체 캐시 삭제
upbit_provider.clear_cache()
```

## 💼 멀티심볼 백테스팅 예제

```python
from quantbt import (
    UpbitDataProvider,
    SimpleBacktestEngine,
    SimpleBroker,
    SimpleMovingAverageCrossStrategy,
    BacktestConfig
)

async def crypto_portfolio_backtest():
    # 1. 주요 암호화폐 포트폴리오 구성
    crypto_symbols = [
        "KRW-BTC",   # 비트코인
        "KRW-ETH",   # 이더리움
        "KRW-XRP",   # 리플
        "KRW-ADA",   # 카르다노
        "KRW-DOT"    # 폴카닷
    ]
    
    # 2. 백테스팅 설정
    config = BacktestConfig(
        symbols=crypto_symbols,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_cash=10000000,  # 1천만원
        timeframe="1h",
        commission_rate=0.0005,  # 업비트 수수료 0.05%
        slippage_rate=0.0001
    )
    
    # 3. 컴포넌트 생성
    upbit_provider = UpbitDataProvider()
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate
    )
    
    # 4. 이동평균 교차 전략
    strategy = SimpleMovingAverageCrossStrategy(
        short_window=12,  # 단기 12시간
        long_window=48   # 장기 48시간 (2일)
    )
    strategy.position_size_pct = 0.15  # 각 코인당 15%
    strategy.max_positions = 4         # 최대 4개 코인
    
    # 5. 백테스팅 실행
    engine = SimpleBacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(upbit_provider)
    engine.set_broker(broker)
    
    result = await engine.run(config)
    
    # 6. 결과 분석
    print(f"총 수익률: {result.total_return_pct:.2f}%")
    print(f"샤프 비율: {result.sharpe_ratio:.2f}")
    print(f"최대 낙폭: {result.max_drawdown_pct:.2f}%")
    
    return result

# 실행
result = asyncio.run(crypto_portfolio_backtest())
```

## 🔧 고급 설정

### API 요청 최적화

```python
# 대량 데이터 처리를 위한 설정
upbit_provider = UpbitDataProvider(
    cache_dir="./data/upbit_cache",
    rate_limit_delay=0.05,         # 빠른 요청 (주의: 너무 빠르면 제한될 수 있음)
    max_candles_per_request=200    # 최대 캔들 수
)
```

### 에러 처리

```python
async def robust_data_loading():
    async with upbit_provider:
        try:
            data = await upbit_provider.get_data(
                symbols=["KRW-BTC", "KRW-ETH"],
                start=datetime.now() - timedelta(days=7),
                end=datetime.now(),
                timeframe="1h"
            )
            return data
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            # 폴백 전략: 캐시된 데이터만 사용
            return await upbit_provider._load_cached_data(...)
```

## ⚠️ 주의사항

### API 제한

- **요청 제한**: 업비트는 분당 요청 수를 제한합니다
- **Rate Limiting**: `rate_limit_delay`를 적절히 설정하세요
- **대량 데이터**: 오래된 데이터나 대량 데이터 요청 시 시간이 오래 걸릴 수 있습니다

### 데이터 품질

- **체결 없는 시간**: 체결이 없었던 시간대는 캔들이 생성되지 않습니다
- **시장 휴장**: 업비트는 24/7 운영되지만, 시스템 점검 시간이 있을 수 있습니다
- **시간대**: 모든 시간은 UTC 기준으로 저장됩니다

### 성능 최적화

```python
# 효율적인 데이터 로딩
async def efficient_loading():
    # 1. 필요한 심볼만 선택
    symbols = ["KRW-BTC", "KRW-ETH"]  # 너무 많은 심볼은 피하세요
    
    # 2. 적절한 시간 범위
    days = 30  # 너무 긴 기간은 피하세요
    
    # 3. 적절한 시간프레임
    timeframe = "1h"  # 1분봉보다는 1시간봉이 효율적
    
    async with upbit_provider:
        data = await upbit_provider.get_data(
            symbols=symbols,
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            timeframe=timeframe
        )
    
    return data
```

## 🛠️ 트러블슈팅

### 일반적인 문제들

1. **네트워크 오류**
   ```python
   # 재시도 로직 구현
   import asyncio
   
   async def retry_request(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await func()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise e
               await asyncio.sleep(2 ** attempt)  # 지수 백오프
   ```

2. **캐시 오류**
   ```python
   # 캐시 초기화
   upbit_provider.clear_cache()
   ```

3. **메모리 부족**
   ```python
   # 데이터를 청크 단위로 처리
   async def chunk_processing():
       chunk_size = timedelta(days=7)
       current_date = start_date
       
       all_data = []
       while current_date < end_date:
           chunk_end = min(current_date + chunk_size, end_date)
           chunk_data = await upbit_provider.get_data(
               symbols=symbols,
               start=current_date,
               end=chunk_end,
               timeframe=timeframe
           )
           all_data.append(chunk_data)
           current_date = chunk_end
       
       return pl.concat(all_data)
   ```

## 📈 실전 활용 예제

### 1. 일중 스캘핑 전략

```python
# 5분봉을 이용한 단기 전략
config = BacktestConfig(
    symbols=["KRW-BTC"],
    timeframe="5m",
    start_date=datetime.now() - timedelta(days=3),
    end_date=datetime.now()
)
```

### 2. 스윙 트레이딩 전략

```python
# 4시간봉을 이용한 중기 전략
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH", "KRW-XRP"],
    timeframe="4h",
    start_date=datetime.now() - timedelta(days=60),
    end_date=datetime.now()
)
```

### 3. 포지션 트레이딩 전략

```python
# 1시간봉을 이용한 장기 전략
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now()
)
```

업비트 데이터 프로바이더를 활용하여 실제 암호화폐 시장 데이터로 백테스팅을 수행해보세요! 