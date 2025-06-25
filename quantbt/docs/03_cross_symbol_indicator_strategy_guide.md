# 튜토리얼 3: 크로스 심볼 지표 전략

이번 튜토리얼에서는 한 종목의 기술적 지표를 사용하여 다른 종목의 거래 신호를 생성하는 '크로스 심볼(Cross-symbol)' 전략을 구현하는 방법을 알아봅니다. 예를 들어, 여러 알트코인 중 변동성이 가장 높은 코인만 골라서 거래하는 것과 같은 고급 전략을 구현할 수 있습니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 03_cross_symbol_indicator_strategy.ipynb](../examples/03_cross_symbol_indicator_strategy.ipynb)**

## 1. 크로스 심볼 전략의 개념

크로스 심볼 전략은 여러 종목의 데이터를 종합적으로 분석하여 투자 결정을 내리는 방식입니다. 시장 전체의 분위기를 주도하는 자산(예: BTC)을 기준으로 다른 자산을 거래하거나, 여러 자산 중 특정 조건을 만족하는 대상만 선택하여 거래하는 데 사용됩니다. 이를 통해 더 정교한 신호를 포착하여 거래의 효율성을 높일 수 있습니다.

## 2. 백테스팅 과정

### 단계 1: 크로스 심볼 로직을 포함한 커스텀 전략 생성

QuantBT의 `TradingStrategy`를 상속받아 커스텀 전략을 만듭니다. 크로스 심볼 로직은 주로 지표 계산 단계에서 구현됩니다.

- **`_compute_indicators_for_symbol`**: 백테스팅 시작 전, 각 종목에 필요한 개별 지표(예: 이동평균, 변동성)를 계산합니다.
- **`_compute_cross_symbol_indicators`**: 모든 종목의 데이터가 합쳐진 상태에서, 종목 간 비교를 통해 새로운 지표(예: 변동성 순위)를 생성합니다. `polars`의 윈도우 함수 `over("timestamp")`가 핵심적인 역할을 합니다.
- **`generate_signals`**: 각 시점에서 데이터를 기반으로, 크로스 심볼 지표(변동성 순위)를 조건으로 사용하여 거래 주문을 생성합니다.

```python
import polars as pl
from quantbt import TradingStrategy, Order, OrderSide, OrderType

class VolatilityBasedStrategy(TradingStrategy):
    def __init__(self, volatility_window=14, **kwargs):
        super().__init__(**kwargs)
        self.volatility_window = volatility_window

    def _compute_indicators_for_symbol(self, symbol_data):
        # 각 심볼의 이동평균과 변동성을 계산
        data = symbol_data.sort("timestamp")
        volatility = data["close"].pct_change().rolling_std(window_size=self.volatility_window)
        # ... 다른 지표들 ...
        return data.with_columns([
            volatility.alias("volatility"),
            # ...
        ])

    def _compute_cross_symbol_indicators(self, data: pl.DataFrame):
        # 타임스탬프별로 모든 종목의 변동성 순위를 계산
        return data.with_columns(
            pl.col("volatility")
            .rank("ordinal", descending=True)
            .over("timestamp")
            .alias("vol_rank")
        )

    def generate_signals(self, data: dict):
        orders = []
        # 변동성 순위가 1위인 종목만 거래
        if data.get('vol_rank') == 1:
            # ... 매수/매도 조건 ...
            # if condition:
            #     orders.append(Order(...))
        return orders
```

### 단계 2: 백테스팅 실행

생성한 커스텀 전략을 사용하여 백테스팅을 실행합니다. `BacktestConfig`에 분석할 모든 종목(`"KRW-BTC"`, `"KRW-ETH"`)을 지정합니다.

```python
from datetime import datetime
from quantbt import BacktestEngine, BacktestConfig, UpbitDataProvider, SimpleBroker

# 1. 백테스팅 설정
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000000,
    commission_rate=0.001
)

# 2. 구성 요소 초기화
data_provider = UpbitDataProvider()
strategy = VolatilityBasedStrategy() # 위에서 정의한 전략
broker = SimpleBroker(initial_cash=config.initial_cash)

# 3. 백테스트 엔진 설정 및 실행
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. 결과 분석

결과 분석은 일반적인 백테스팅과 동일합니다. `result.trades`를 통해 어떤 종목이 어떤 시점에 거래되었는지 확인할 수 있습니다.

```python
# 전체 포트폴리오 성과 요약
result.print_summary()

# 포트폴리오 수익 곡선 시각화
result.plot_portfolio_performance()

# 거래 내역 확인
print(result.trades)
```

이 가이드를 통해 서로 다른 종목을 연계하는 복잡한 전략도 QuantBT로 손쉽게 구현하고 검증할 수 있음을 확인했습니다. 다음 장에서는 여러 시간 프레임을 동시에 분석하는 멀티 타임프레임 전략을 살펴보겠습니다. 