# 튜토리얼 2: 멀티 심볼 전략 백테스팅

이 튜토리얼에서는 여러 종목(심볼)으로 구성된 포트폴리오를 동시에 백테스팅하는 방법을 알아봅니다. QuantBT는 여러 자산을 아우르는 전략을 손쉽게 테스트할 수 있는 강력한 기능을 제공합니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 02_multi_symbol_strategy.ipynb](../examples/02_multi_symbol_strategy.ipynb)**

## 1. 멀티 심볼 전략의 개념

멀티 심볼 전략은 여러 종목의 가격 데이터를 동시에 분석하여 각 종목에 대한 매매 신호를 독립적으로 또는 연관지어 생성합니다. 이를 통해 자산 분산 효과를 노리거나, 여러 시장의 기회를 포착하는 등의 전략을 구사할 수 있습니다.

## 2. 백테스팅 과정

### 단계 1: 데이터 불러오기

백테스팅에 사용할 여러 종목의 데이터를 로드합니다. `load_data` 함수에 종목 리스트를 전달하면 됩니다.

```python
import quantbt as qbt

# BTC와 ETH 데이터를 동시에 로드
data = qbt.load_data(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframe="1h"
)
```

### 단계 2: 멀티 심볼 전략 설정

전략은 각 심볼에 대해 독립적으로 실행될 수 있도록 설계되어야 합니다. `backtest` 함수에 종목 리스트를 전달하여 멀티 심볼 백테스팅을 활성화합니다.

```python
from quantbt.strategies import RSIStrategy

# 각 종목에 동일한 RSI 전략 적용
strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)

# 백테스팅 실행
result = qbt.backtest(
    strategy=strategy,
    ohlcv=data,
    symbols=["BTCUSDT", "ETHUSDT"], # 대상 심볼 지정
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_cash=10000,
)
```

QuantBT 엔진은 루프를 돌며 각 시점의 데이터를 모든 종목에 대해 전략에 전달하고, 생성된 주문을 처리합니다.

## 3. 결과 분석

백테스팅 결과에는 포트폴리오 전체의 성과와 각 종목별 성과가 모두 포함됩니다.

```python
# 포트폴리오 전체의 종합 성과
result.stats()
result.plot()

# 개별 종목의 성과 분석
btc_trades = result.trades.filter(pl.col("symbol") == "BTCUSDT")
eth_trades = result.trades.filter(pl.col("symbol") == "ETHUSDT")

print("=== BTC Trades ===")
print(btc_trades)

print("=== ETH Trades ===")
print(eth_trades)
```

이처럼 QuantBT를 사용하면 복잡한 멀티 심볼 포트폴리오 전략도 간결한 코드로 테스트하고 분석할 수 있습니다. 다음 튜토리얼에서는 한 종목의 지표를 다른 종목 거래에 활용하는 '크로스 심볼' 전략에 대해 알아보겠습니다. 