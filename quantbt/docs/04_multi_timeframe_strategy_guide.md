# Tutorial 4: Multi-Timeframe Strategy

Sophisticated trading strategies often analyze multiple timeframes simultaneously. For example, capturing long-term trends with hourly bars and finding precise entry points with minute bars. In this tutorial, you'll learn how to implement multi-timeframe strategies using QuantBT.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 04_multi_timeframe_strategy.ipynb](../examples/04_multi_timeframe_strategy.ipynb)**

## 1. Multi-Timeframe Strategy Concept

The core of this strategy is to determine the overall market trend (upward/downward/sideways) through data from longer timeframes (e.g., 1-hour bars) and capture optimal timing for actual buy/sell orders using data from shorter timeframes (e.g., 15-minute bars). This allows strategies to 'follow the trend while entering at more favorable prices'.

## 2. Backtesting Process

QuantBT provides a special base class called `MultiTimeframeTradingStrategy` for multi-timeframe strategies.

### Step 1: Create Multi-Timeframe Custom Strategy

Create a custom strategy by inheriting from `MultiTimeframeTradingStrategy`.

- **`__init__`**: Define all timeframes to use and settings for indicators to calculate for each timeframe in `timeframe_configs`. Also specify the `primary_timeframe` that serves as the basis for trading signals.
- **`_compute_indicators_for_symbol_and_timeframe`**: Implement logic to calculate necessary indicators for each timeframe.
- **`generate_signals_multi_timeframe`**: Generate final trading signals by synthesizing data from multiple timeframes.

```python
from quantbt import MultiTimeframeTradingStrategy, Order, OrderSide, OrderType
import polars as pl

class MultiTimeframeSMAStrategy(MultiTimeframeTradingStrategy):
    def __init__(self):
        # 1. Define timeframes to use and related settings
        timeframe_configs = {
            "15m": { "sma_windows": [10, 30] }, # For short-term signals
            "1h": { "sma_windows": [60] }      # For long-term trend filter
        }
        
        super().__init__(
            timeframe_configs=timeframe_configs,
            primary_timeframe="15m" # Base timeframe for signal generation
        )
        
    def _compute_indicators_for_symbol_and_timeframe(self, symbol_data, timeframe, config):
        # 2. Calculate indicators for each timeframe
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
        # 3. Generate signals by synthesizing multi-timeframe data
        orders = []
        d15m = multi_current_data.get("15m", {})
        d1h = multi_current_data.get("1h", {})

        # Extract essential indicator values
        price_1h = d1h.get('close')
        sma_trend = d1h.get('sma_trend')
        sma_short = d15m.get('sma_short')
        sma_long = d15m.get('sma_long')

        if any(v is None for v in [price_1h, sma_trend, sma_short, sma_long]):
            return orders

        # Buy condition: 1h uptrend & 15m golden cross
        if price_1h > sma_trend and sma_short > sma_long:
            # ... Buy order logic ...
            pass
        # Sell condition: 15m death cross
        elif sma_short < sma_long:
            # ... Sell order logic ...
            pass
            
        return orders
```

### Step 2: Execute Backtesting

The `timeframe` in `BacktestConfig` must match the `primary_timeframe` defined in the strategy. The QuantBT engine automatically manages the other timeframe data.

```python
from datetime import datetime
from quantbt import BacktestEngine, BacktestConfig, UpbitDataProvider, SimpleBroker

# 1. Configure backtesting (set base timeframe to '15m')
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="15m",
    initial_cash=10000000,
    commission_rate=0.001
)

# 2. Initialize components
data_provider = UpbitDataProvider()
strategy = MultiTimeframeSMAStrategy() # Strategy defined above
broker = SimpleBroker(initial_cash=config.initial_cash)

# 3. Run backtest engine
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. Result Analysis

Result analysis is the same as other strategies.

```python
result.print_summary()
result.plot_portfolio_performance()
```

Using QuantBT's `MultiTimeframeTradingStrategy`, you can systematically implement and test even complex multi-timeframe analysis-based strategies. Next, we'll explore parallel search functionality to efficiently find optimal parameters for strategies. 