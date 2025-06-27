# Tutorial 2: Multi-Symbol Strategy Backtesting

This tutorial shows how to backtest portfolios composed of multiple symbols simultaneously. QuantBT provides powerful capabilities to easily test strategies that span multiple assets.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 02_multi_symbol_strategy.ipynb](../examples/02_multi_symbol_strategy.ipynb)**

## 1. Multi-Symbol Strategy Concept

Multi-symbol strategies analyze price data from multiple symbols simultaneously to generate trading signals for each symbol either independently or in correlation. This allows strategies to benefit from asset diversification or capture opportunities across multiple markets.

## 2. Backtesting Process

Multi-symbol backtesting has a very similar structure to single-symbol backtesting. The main difference is passing a list of multiple symbol names to the `symbols` attribute in `BacktestConfig`.

### Step 1: Import Required Modules

Import the main classes needed for backtesting directly from the `quantbt` library.

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

### Step 2: Define Multi-Symbol Strategy

Define a strategy that can handle multiple symbols. You can inherit from the base `TradingStrategy` to independently generate signals for each symbol's data. This example uses a simple moving average crossover strategy.

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
        
        # Generate trading signals independently for each symbol
        if current_price > buy_sma and symbol not in current_positions:
            # ... Buy order logic ...
        elif current_price < sell_sma and symbol in current_positions:
            # ... Sell order logic ...
            
        return orders
```

### Step 3: Configure and Run Backtesting

Pass a list of desired symbols like `["KRW-BTC", "KRW-ETH"]` to the `symbols` attribute of `BacktestConfig`.

```python
# 1. Set up data provider
data_provider = UpbitDataProvider()

# 2. Configure backtesting (BTC and ETH simultaneously)
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000,
    commission_rate=0.001,
    slippage_rate=0.0
)

# 3. Initialize strategy and broker
strategy = MultiSymbolSMAStrategy(buy_sma=10, sell_sma=30)
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate
)

# 4. Run backtest engine
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

The QuantBT engine loops through each time point, passing data for all symbols to the strategy, and processes the generated orders.

## 3. Result Analysis

Backtesting results include both overall portfolio performance and individual symbol performance.

```python
import polars as pl

# Overall portfolio performance
result.print_summary()
result.plot_portfolio_performance()

# Individual symbol performance analysis
btc_trades = result.trades.filter(pl.col("symbol") == "KRW-BTC")
eth_trades = result.trades.filter(pl.col("symbol") == "KRW-ETH")

print("=== BTC Trades ===")
print(btc_trades)

print("=== ETH Trades ===")
print(eth_trades)
```

With QuantBT, you can test and analyze complex multi-symbol portfolio strategies with concise code. In the next tutorial, we'll explore 'cross-symbol' strategies that use indicators from one symbol to trade another symbol. 