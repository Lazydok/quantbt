# 튜토리얼 4: 멀티 타임프레임 전략

정교한 트레이딩 전략은 종종 여러 시간 프레임(Timeframe)을 동시에 분석합니다. 예를 들어, 장기 추세는 시간(hour) 봉으로 파악하고, 정확한 진입 시점은 분(minute) 봉으로 잡는 식입니다. 이번 튜토리얼에서는 QuantBT를 사용하여 멀티 타임프레임 전략을 구현하는 방법을 배웁니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 04_multi_timeframe_strategy.ipynb](../examples/04_multi_timeframe_strategy.ipynb)**

## 1. 멀티 타임프레임 전략의 개념

이 전략의 핵심은 긴 시간 프레임(예: 1시간 봉)의 데이터를 통해 시장의 전반적인 추세(상승/하락/횡보)를 판단하고, 짧은 시간 프레임(예: 15분 봉)의 데이터를 사용하여 실제 매수/매도 주문을 실행할 최적의 타이밍을 포착하는 것입니다. 이를 통해 '추세에 순응하며, 더 유리한 가격에 진입'하는 전략을 구사할 수 있습니다.

## 2. 백테스팅 과정

QuantBT는 멀티 타임프레임 전략을 위해 `MultiTimeframeTradingStrategy`라는 특별한 기반 클래스를 제공합니다.

### 단계 1: 멀티 타임프레임 커스텀 전략 생성

`MultiTimeframeTradingStrategy`를 상속받아 커스텀 전략을 작성합니다.

- **`__init__`**: 사용할 모든 타임프레임과 각 타임프레임에서 계산할 지표의 설정을 `timeframe_configs`에 정의합니다. 거래 신호의 기준이 되는 `primary_timeframe`도 지정합니다.
- **`_compute_indicators_for_symbol_and_timeframe`**: 각 타임프레임별로 필요한 지표를 계산하는 로직을 구현합니다.
- **`generate_signals_multi_timeframe`**: 여러 타임프레임의 데이터를 종합하여 최종 매매 신호를 생성합니다.

```python
from quantbt import MultiTimeframeTradingStrategy, Order, OrderSide, OrderType
import polars as pl

class MultiTimeframeSMAStrategy(MultiTimeframeTradingStrategy):
    def __init__(self):
        # 1. 사용할 타임프레임과 관련 설정 정의
        timeframe_configs = {
            "15m": { "sma_windows": [10, 30] }, # 단기 신호용
            "1h": { "sma_windows": [60] }      # 장기 추세 필터용
        }
        
        super().__init__(
            timeframe_configs=timeframe_configs,
            primary_timeframe="15m" # 신호 생성의 기준이 되는 타임프레임
        )
        
    def _compute_indicators_for_symbol_and_timeframe(self, symbol_data, timeframe, config):
        # 2. 각 타임프레임별 지표 계산
        data = symbol_data.sort("timestamp")
        indicators = []
        if timeframe == "15m":
            indicators.extend([
                pl.col("close").rolling_mean(config["sma_windows"][0]).alias("sma_short"),
                pl.col("close").rolling_mean(config["sma_windows"][1]).alias("sma_long")
            ])
        elif timeframe == "1h":
            indicators.append(
                pl.col("close").rolling_mean(config["sma_windows"][0]).alias("sma_trend")
            )
        return data.with_columns(indicators)
    
    def generate_signals_multi_timeframe(self, multi_current_data):
        # 3. 여러 타임프레임 데이터를 종합하여 신호 생성
        orders = []
        d15m = multi_current_data.get("15m", {})
        d1h = multi_current_data.get("1h", {})

        # 필수 지표 값 추출
        price_1h = d1h.get('close')
        sma_trend = d1h.get('sma_trend')
        sma_short = d15m.get('sma_short')
        sma_long = d15m.get('sma_long')

        if any(v is None for v in [price_1h, sma_trend, sma_short, sma_long]):
            return orders

        # 매수 조건: 1h 추세 상승 & 15m 골든 크로스
        if price_1h > sma_trend and sma_short > sma_long:
            # ... 매수 주문 로직 ...
            pass
        # 매도 조건: 15m 데드 크로스
        elif sma_short < sma_long:
            # ... 매도 주문 로직 ...
            pass
            
        return orders
```

### 단계 2: 백테스팅 실행

`BacktestConfig`의 `timeframe`은 전략에서 정의한 `primary_timeframe`과 일치시켜야 합니다. QuantBT 엔진이 나머지 타임프레임 데이터는 알아서 관리합니다.

```python
from datetime import datetime
from quantbt import BacktestEngine, BacktestConfig, UpbitDataProvider, SimpleBroker

# 1. 백테스팅 설정 (기준 타임프레임은 '15m'로 설정)
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="15m",
    initial_cash=10000000,
    commission_rate=0.001
)

# 2. 구성 요소 초기화
data_provider = UpbitDataProvider()
strategy = MultiTimeframeSMAStrategy() # 위에서 정의한 전략
broker = SimpleBroker(initial_cash=config.initial_cash)

# 3. 백테스트 엔진 실행
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. 결과 분석

결과 분석은 다른 전략들과 동일합니다.

```python
result.print_summary()
result.plot_portfolio_performance()
```

이처럼 QuantBT의 `MultiTimeframeTradingStrategy`를 사용하면, 복잡한 멀티 타임프레임 분석 기반의 전략도 체계적으로 구현하고 테스트할 수 있습니다. 다음으로는 전략의 최적 파라미터를 효율적으로 찾기 위한 병렬 탐색 기능에 대해 알아보겠습니다. 