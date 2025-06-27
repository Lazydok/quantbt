# Tutorial 1: Simple Strategy Backtesting

This tutorial guides you through the process of backtesting a simple moving average crossover strategy for a single symbol using QuantBT. You'll learn the most basic usage of QuantBT.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 01_simple_strategy.ipynb](../examples/01_simple_strategy.ipynb)**

## 1. Basic Concepts

- **Data Provider**: Provides price data (OHLCV) needed for backtesting. Supports various sources including CSV files, databases, and real-time APIs.
- **Strategy**: Logic that generates buy/sell signals. Defines indicator calculations and signal generation rules.
- **Broker**: Acts as a virtual exchange, accepting and executing orders while managing the portfolio.
- **Backtest Engine**: Connects the data provider, strategy, and broker to orchestrate and execute the entire backtesting process.

## 2. Backtesting Process

### Step 1: Import Required Modules

Import the main classes needed for backtesting directly from the `quantbt` library.

```python
from datetime import datetime
from quantbt import (
    BacktestEngine,
    BacktestConfig,
    UpbitDataProvider,
    SimpleBroker,
    SimpleSMAStrategy,
)
```

### Step 2: Define Backtesting Configuration

Use `BacktestConfig` to set the basic conditions for backtesting. This includes symbols to analyze, time period, timeframe, initial capital, commission and slippage rates, etc.

```python
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000,
    commission_rate=0.001,
    slippage_rate=0.0, # Slippage rate
    save_portfolio_history=True,
)
```

### Step 3: Initialize Components

Initialize each component (data provider, strategy, broker) needed for backtesting.

```python
# Set up data provider (using Upbit data)
data_provider = UpbitDataProvider()

# Select strategy (Simple moving average crossover strategy)
# Buy when 10-day MA crosses above 30-day MA, sell when it crosses below
strategy = SimpleSMAStrategy(buy_sma=10, sell_sma=30)

# Set up broker
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
)
```

### Step 4: Run Backtest Engine

Create a `BacktestEngine`, configure it with the components created above, and call the `run` method to execute the backtesting.

```python
# Set up and run backtesting engine
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. Result Analysis

Once backtesting is complete, you can check various performance metrics and trade history through the `result` object.

```python
# Print summary statistics
result.print_summary()

# Visualize portfolio performance (recommended in Jupyter Notebook environment)
result.plot_portfolio_performance()
```

### Key Performance Metrics

- **Total Return**: Total return rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum decline from peak
- **Win Rate**: Trading win rate

We hope this guide has helped you understand the basic usage flow of QuantBT. In the next tutorial, we'll cover more complex multi-symbol strategies.