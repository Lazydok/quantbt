# Tutorial 6: Bayesian Parameter Optimization

Grid search from the previous tutorial can be inefficient because it randomly tests all parameter combinations. 'Bayesian Optimization' is an intelligent search method to find better parameters with fewer attempts. It probabilistically predicts the next parameters to test based on previous test results, intensively exploring areas expected to perform best.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 06_bayesian_optimization.ipynb](../examples/06_bayesian_optimization.ipynb)**

## 1. Bayesian Optimization Concept

Bayesian optimization aims to find parameters that maximize the value of an 'Objective Function'.
1.  **Initial Sampling**: Randomly select a few points in the parameter space and execute the objective function (backtesting).
2.  **Probabilistic Modeling**: Create a probabilistic model (usually Gaussian Process) that predicts performance in unexplored spaces based on current (parameter, performance) data.
3.  **Acquisition Function**: Use the probabilistic model to calculate parameters with the highest 'probability of exceeding current best performance in the next attempt'.
4.  **Iteration**: Perform actual backtesting with parameters found in step 3, add results to the data in step 2 to update the model. Repeat this process for a predetermined number of times.

This approach reduces tests on meaningless parameter combinations and efficiently finds optimal solutions by focusing on promising areas.

## 2. Bayesian Optimization Process

QuantBT provides integrated Bayesian optimization functionality through the `BayesianParameterOptimizer` class. It runs in parallel on a Ray cluster, and users only need to define the strategy to optimize and parameter space.

### Step 1: Define Optimization Target (Strategy Class, Parameter Space)

First, define the strategy class to optimize and specify the search ranges for parameters using `ParameterSpace`.

```python
from quantbt.core.interfaces.strategy import TradingStrategy
from quantbt.ray.optimization.parameter_space import ParameterSpace

# 1. Define strategy class to optimize
class SimpleSMAStrategy(TradingStrategy):
    def __init__(self, buy_sma: int, sell_sma: int, position_size_pct: float):
        super().__init__()
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.position_size_pct = position_size_pct
    # ... (indicator calculation and signal generation logic internally identical) ...

# 2. Define parameter search space
param_config = {
    'buy_sma': (10, 100),            # Integer between 10 and 100
    'sell_sma': (20, 200),           # Integer between 20 and 200
    'position_size_pct': (0.5, 1.0)  # Float between 0.5 and 1.0
}
param_space = ParameterSpace.from_dict(param_config)
```

### Step 2: Create Optimizer and Execute

Create an instance by passing strategy class, parameter space, and backtest configuration to `BayesianParameterOptimizer`, then call the `optimize` method asynchronously.

```python
from quantbt.ray.bayesian_parameter_optimizer import BayesianParameterOptimizer
from quantbt.core.value_objects.backtest_config import BacktestConfig
import asyncio
from datetime import datetime

# 1. Basic backtest configuration
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe="4h",
    initial_cash=10_000_000
)

# 2. Create optimizer
optimizer = BayesianParameterOptimizer(
    strategy_class=SimpleSMAStrategy,
    param_space=param_space,
    config=config,
    num_actors=8, # Number of CPU cores to use
    n_initial_points=10 # Number of initial random explorations
)

# 3. Execute optimization asynchronously
async def run_optimization():
    results = await optimizer.optimize(
        objective_metric='sharpe_ratio', # Target metric for optimization
        n_iter=100 # Total number of optimization attempts
    )
    return results

# Run async function (in Jupyter Notebook, you can use await directly)
all_results = asyncio.run(run_optimization())
```

### Step 3: Result Analysis

Once optimization is complete, the `optimize` function returns a list containing results for all attempts. You can analyze this list to find the parameters that showed the best performance.

```python
# Find result with highest Sharpe ratio
best_result = max(all_results, key=lambda x: x['result'].get('sharpe_ratio', -999))

print("=== Best Parameters Found ===")
print(f"Params: {best_result['params']}")
print(f"Sharpe Ratio: {best_result['result']['sharpe_ratio']:.4f}")
print(f"Total Return: {best_result['result']['total_return']:.4f}")
```

Bayesian optimization can find excellent parameters with far fewer iterations compared to grid search, significantly reducing optimization time for complex strategies. 