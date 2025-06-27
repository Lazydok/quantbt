# Tutorial 3: Cross-Symbol Indicator Strategy

This tutorial explores how to implement 'cross-symbol' strategies that use technical indicators from one symbol to generate trading signals for another symbol. For example, you can implement advanced strategies like trading only the most volatile coin among multiple altcoins.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 03_cross_symbol_indicator_strategy.ipynb](../examples/03_cross_symbol_indicator_strategy.ipynb)**

## 1. Cross-Symbol Strategy Concept

Cross-symbol strategies make investment decisions by comprehensively analyzing data from multiple symbols. They can be used to trade other assets based on market-leading assets (e.g., BTC), or to select and trade only targets that meet specific conditions among multiple assets. This allows for capturing more sophisticated signals and improving trading efficiency.

## 2. Backtesting Process

### Step 1: Create Custom Strategy with Cross-Symbol Logic

Create a custom strategy by inheriting from QuantBT's `TradingStrategy`. Cross-symbol logic is mainly implemented in the indicator calculation phase.

- **`_compute_indicators_for_symbol`**: Before backtesting starts, calculates individual indicators (e.g., moving averages, volatility) for each symbol.
- **`_compute_cross_symbol_indicators`**: With all symbols' data combined, generates new indicators (e.g., volatility ranking) through inter-symbol comparisons. Polars' window function `over("timestamp")` plays a crucial role.
- **`generate_signals`**: At each time point, generates trading orders using cross-symbol indicators (volatility ranking) as conditions based on the data.

```python
import polars as pl
from quantbt import TradingStrategy, Order, OrderSide, OrderType

class VolatilityBasedStrategy(TradingStrategy):
    def __init__(self, volatility_window=14, **kwargs):
        super().__init__(**kwargs)
        self.volatility_window = volatility_window

    def _compute_indicators_for_symbol(self, symbol_data):
        # Calculate moving averages and volatility for each symbol
        data = symbol_data.sort("timestamp")
        volatility = data["close"].pct_change().rolling_std(window_size=self.volatility_window)
        # ... other indicators ...
        return data.with_columns([
            volatility.alias("volatility"),
            # ...
        ])

    def _compute_cross_symbol_indicators(self, data: pl.DataFrame):
        # Calculate volatility ranking for all symbols by timestamp
        return data.with_columns(
            pl.col("volatility")
            .rank("ordinal", descending=True)
            .over("timestamp")
            .alias("vol_rank")
        )

    def generate_signals(self, data: dict):
        orders = []
        # Trade only symbols with volatility rank #1
        if data.get('vol_rank') == 1:
            # ... buy/sell conditions ...
            # if condition:
            #     orders.append(Order(...))
        return orders
```

### Step 2: Execute Backtesting

Execute backtesting using the custom strategy created above. Specify all symbols to analyze (`"KRW-BTC"`, `"KRW-ETH"`) in `BacktestConfig`.

```python
from datetime import datetime
from quantbt import BacktestEngine, BacktestConfig, UpbitDataProvider, SimpleBroker

# 1. Configure backtesting
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000000,
    commission_rate=0.001
)

# 2. Initialize components
data_provider = UpbitDataProvider()
strategy = VolatilityBasedStrategy() # Strategy defined above
broker = SimpleBroker(initial_cash=config.initial_cash)

# 3. Set up and run backtest engine
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. Result Analysis

Result analysis is the same as regular backtesting. You can check which symbols were traded at which time points through `result.trades`.

```python
# Overall portfolio performance summary
result.print_summary()

# Visualize portfolio performance curve
result.plot_portfolio_performance()

# Check trade history
print(result.trades)
```

Through this guide, you've confirmed that even complex strategies linking different symbols can be easily implemented and validated with QuantBT. In the next chapter, we'll explore multi-timeframe strategies that analyze multiple time frames simultaneously. 