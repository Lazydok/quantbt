# 튜토리얼 4: 멀티 타임프레임 전략

정교한 트레이딩 전략은 종종 여러 시간 프레임(Timeframe)을 동시에 분석합니다. 예를 들어, 장기 추세는 시간(hour) 봉으로 파악하고, 정확한 진입 시점은 분(minute) 봉으로 잡는 식입니다. 이번 튜토리얼에서는 QuantBT를 사용하여 멀티 타임프레임 전략을 구현하는 방법을 배웁니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 04_multi_timeframe_strategy.ipynb](../examples/04_multi_timeframe_strategy.ipynb)**

## 1. 멀티 타임프레임 전략의 개념

이 전략의 핵심은 긴 시간 프레임(예: 4시간 봉)의 데이터를 통해 시장의 전반적인 추세(상승/하락/횡보)를 판단하고, 짧은 시간 프레임(예: 15분 봉)의 데이터를 사용하여 실제 매수/매도 주문을 실행할 최적의 타이밍을 포착하는 것입니다. 이를 통해 '추세에 순응하며, 더 유리한 가격에 진입'하는 전략을 구사할 수 있습니다.

## 2. 백테스팅 과정

### 단계 1: 여러 시간 프레임의 데이터 준비

백테스팅을 위해서는 전략에 필요한 모든 시간 프레임의 데이터를 준비해야 합니다. QuantBT는 내부적으로 기본이 되는 가장 작은 시간 프레임(예: 1분 봉) 데이터를 사용하여 다른 시간 프레임(5분, 1시간 등) 데이터를 자동으로 리샘플링(resampling)합니다.

```python
import quantbt as qbt

# 백테스팅의 기준이 될 가장 작은 단위의 시간 프레임 데이터를 로드합니다.
# 예를 들어, 1시간, 15분봉을 보려면 최소 1분봉 데이터가 필요할 수 있습니다.
ohlcv = qbt.load_data("BTCUSDT", timeframe="5m")
```

### 단계 2: 멀티 타임프레임 커스텀 전략 생성

여러 시간 프레임의 지표를 함께 사용하기 위해 커스텀 전략을 작성합니다.

- **`precompute_indicators`**: 이 단계에서 `resample` 기능을 사용하여 필요한 모든 시간 프레임의 지표를 계산합니다.
- **`generate_signals`**: 현재 시점에서 각기 다른 시간 프레임의 지표 값들을 모두 조회하여 거래 로직을 구성합니다.

```python
import quantbt as qbt

class MultiTimeframeStrategy(qbt.TradingStrategy):
    def precompute_indicators(self, data):
        # 1시간봉 데이터 리샘플링 및 장기 추세선(SMA 50) 계산
        df_1h = data.resample("1h").add_indicator("sma", 50, alias="sma_long")
        
        # 15분봉 데이터 리샘플링 및 단기 신호선(SMA 20) 계산
        df_15m = data.resample("15m").add_indicator("sma", 20, alias="sma_short")

        # 계산된 지표들을 원본 데이터에 병합
        data = data.add_indicators(df_1h, df_15m)
        return data

    def generate_signals(self, data):
        orders = []
        
        # 현재 시점의 가격 및 각 타임프레임의 지표 값 가져오기
        price = data.get_price("close")
        sma_long = data.get_indicator_value("sma_long")  # 1시간봉 SMA
        sma_short = data.get_indicator_value("sma_short") # 15분봉 SMA

        # 매수 조건: 장기 추세가 상승(가격 > 1h SMA)이고, 단기적으로 가격이 조정을 받은 후 반등할 때(가격 > 15m SMA)
        if price > sma_long and price > sma_short:
            # ... 매수 주문 로직 ...
            pass
        
        # 매도 조건: ...
        
        return orders
```

### 단계 3: 백테스팅 실행

생성한 전략과 기준 데이터를 사용하여 백테스트를 실행합니다.

```python
strategy = MultiTimeframeStrategy()
result = qbt.backtest(
    strategy=strategy,
    ohlcv=ohlcv,
    # ... other parameters
)
```

## 3. 결과 분석

결과 분석은 다른 전략들과 동일합니다. `result.plot()`을 통해 거래가 발생한 시점을 시각적으로 확인하며 전략이 의도대로 작동했는지 검토할 수 있습니다.

```python
result.stats()
result.plot()
```

이처럼 QuantBT의 리샘플링과 지표 통합 기능을 사용하면, 복잡한 멀티 타임프레임 분석 기반의 전략도 체계적으로 구현하고 테스트할 수 있습니다. 다음으로는 전략의 최적 파라미터를 효율적으로 찾기 위한 병렬 탐색 기능에 대해 알아보겠습니다. 