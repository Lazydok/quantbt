# 튜토리얼 2: 멀티 심볼 전략 백테스팅

이 튜토리얼에서는 여러 종목(심볼)으로 구성된 포트폴리오를 동시에 백테스팅하는 방법을 알아봅니다. QuantBT는 여러 자산을 아우르는 전략을 손쉽게 테스트할 수 있는 강력한 기능을 제공합니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 02_multi_symbol_strategy.ipynb](../examples/02_multi_symbol_strategy.ipynb)**

## 1. 멀티 심볼 전략의 개념

멀티 심볼 전략은 여러 종목의 가격 데이터를 동시에 분석하여 각 종목에 대한 매매 신호를 독립적으로 또는 연관지어 생성합니다. 이를 통해 자산 분산 효과를 노리거나, 여러 시장의 기회를 포착하는 등의 전략을 구사할 수 있습니다.

## 2. 백테스팅 과정

멀티 심볼 백테스팅은 단일 심볼 백테스팅과 매우 유사한 구조를 가집니다. 가장 큰 차이점은 `BacktestConfig`에 여러 종목의 심볼을 리스트로 전달하는 것입니다.

### 단계 1: 필요한 모듈 임포트

백테스팅에 필요한 주요 클래스들을 `quantbt` 라이브러리에서 직접 임포트합니다.

```python
from datetime import datetime
from quantbt import (
    BacktestEngine,
    BacktestConfig,
    UpbitDataProvider,
    SimpleBroker,
    TradingStrategy,
    Order, OrderSide, OrderType
)
```

### 단계 2: 멀티 심볼 전략 정의

여러 종목을 처리할 수 있는 전략을 정의합니다. 기본 `TradingStrategy`를 상속받아 각 심볼 데이터에 대해 독립적으로 신호를 생성하도록 구현할 수 있습니다. 이 예제에서는 간단한 이동평균 교차 전략을 사용합니다.

```python
class MultiSymbolSMAStrategy(TradingStrategy):
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30):
        super().__init__()
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma

    def _compute_indicators_for_symbol(self, symbol_data):
        data = symbol_data.sort("timestamp")
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}")
        ])

    def generate_signals(self, current_data: dict):
        orders = []
        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')

        if buy_sma is None or sell_sma is None:
            return orders

        current_positions = self.get_current_positions()
        
        # 각 종목에 대해 독립적으로 매매 신호 생성
        if current_price > buy_sma and symbol not in current_positions:
            # ... 매수 주문 로직 ...
        elif current_price < sell_sma and symbol in current_positions:
            # ... 매도 주문 로직 ...
            
        return orders
```

### 단계 3: 백테스팅 설정 및 실행

`BacktestConfig`의 `symbols` 속성에 `["KRW-BTC", "KRW-ETH"]`와 같이 원하는 종목의 리스트를 전달합니다.

```python
# 1. 데이터 프로바이더 설정
data_provider = UpbitDataProvider()

# 2. 백테스팅 설정 (BTC와 ETH 동시)
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000,
    commission_rate=0.001,
    slippage_rate=0.0
)

# 3. 전략 및 브로커 초기화
strategy = MultiSymbolSMAStrategy(buy_sma=10, sell_sma=30)
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate
)

# 4. 백테스트 엔진 실행
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

QuantBT 엔진은 루프를 돌며 각 시점의 데이터를 모든 종목에 대해 전략에 전달하고, 생성된 주문을 처리합니다.

## 3. 결과 분석

백테스팅 결과에는 포트폴리오 전체의 성과와 각 종목별 성과가 모두 포함됩니다.

```python
import polars as pl

# 포트폴리오 전체의 종합 성과
result.print_summary()
result.plot_portfolio_performance()

# 개별 종목의 성과 분석
btc_trades = result.trades.filter(pl.col("symbol") == "KRW-BTC")
eth_trades = result.trades.filter(pl.col("symbol") == "KRW-ETH")

print("=== BTC Trades ===")
print(btc_trades)

print("=== ETH Trades ===")
print(eth_trades)
```

이처럼 QuantBT를 사용하면 복잡한 멀티 심볼 포트폴리오 전략도 간결한 코드로 테스트하고 분석할 수 있습니다. 다음 튜토리얼에서는 한 종목의 지표를 다른 종목 거래에 활용하는 '크로스 심볼' 전략에 대해 알아보겠습니다. 