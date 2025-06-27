# Tutorial 5: Parallel Parameter Search (Grid Search)

Strategy performance greatly varies depending on parameters (e.g., moving average periods, RSI values, etc.). The process of testing various values to find the optimal parameter combination is called 'parameter optimization'. This tutorial introduces how to dramatically reduce optimization time by testing multiple parameter combinations in parallel across multiple CPU cores simultaneously.

> You can check the complete code and run it directly from the Jupyter Notebook link below.
>
> 👉 **[Example Notebook Link: 05_parallel_search.ipynb](../examples/05_parallel_search.ipynb)**

## 1. Parameter Optimization and Parallel Processing

**Grid Search** is the most basic optimization method that defines ranges of values to test for each parameter and tests all possible combinations one by one to find the combination that yields the best performance. However, when the number of combinations increases, testing can take an enormous amount of time.

QuantBT dramatically reduces optimization time through integration with the **Ray** library by distributing this process across multiple CPU cores for simultaneous execution (parallel processing). While it may seem somewhat complex due to direct Ray utilization, it enables building a very powerful and flexible optimization environment.

## 2. Parallel Search Process

### Step 1: Initialize Ray Cluster

Set up and start a Ray cluster for parallel processing. `RayClusterManager` manages this process.

```python
from quantbt.ray import RayClusterManager

# Cluster configuration (number of CPU cores to use, etc.)
ray_cluster_config = { "num_cpus": 8 } 
cluster_manager = RayClusterManager(ray_cluster_config)
cluster_manager.initialize_cluster()
```

### Step 2: Define Optimization Target (Strategy, Parameters)

Define the strategy and parameter combinations to test.

```python
from quantbt import TradingStrategy
import numpy as np
from itertools import product

# 1. Strategy to optimize (e.g., SimpleSMAStrategy)
class SimpleSMAStrategy(TradingStrategy):
    def __init__(self, buy_sma: int, sell_sma: int):
        super().__init__()
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
    # ... (indicator calculation and signal generation logic) ...

# 2. Create parameter grid to test
param_grid = {
    'buy_sma': np.arange(10, 21, 5),  # [10, 15, 20]
    'sell_sma': np.arange(30, 51, 10) # [30, 40, 50]
}
param_combinations = list(product(param_grid['buy_sma'], param_grid['sell_sma']))
# Result: [(10, 30), (10, 40), ..., (20, 50)]
```

### Step 3: Data Sharing and Actor Preparation

Load backtesting data once and set it up so all parallel workers (Actors) share it in memory. This is a key step to maximize efficiency.

```python
from quantbt.ray import RayDataManager
from quantbt.ray.backtest_actor import BacktestActor

# 1. Load data into Ray's shared memory through data manager
data_manager = RayDataManager.remote()
data_ref = data_manager.load_real_data.remote(...)

# 2. Create BacktestActors to perform parallel work
# Each Actor has a reference (data_ref) to the shared data
num_actors = cluster_manager.get_available_resources()['cpu']
actors = [BacktestActor.remote(f"actor_{i}", shared_data_ref=data_ref) for i in range(num_actors)]
```

### Step 4: Execute Parallel Backtesting

Assign parameter combinations to the prepared Actors one by one to execute backtesting asynchronously.

```python
import asyncio

# Create list of tasks for each Actor to perform
tasks = []
for i, (buy_sma, sell_sma) in enumerate(param_combinations):
    actor = actors[i % len(actors)] # Distribute cyclically among Actors
    params = {"buy_sma": buy_sma, "sell_sma": sell_sma}
    task = actor.execute_backtest.remote(params, SimpleSMAStrategy)
    tasks.append(task)

# Wait for all tasks to complete and collect results
all_results = await asyncio.gather(*tasks)
```

### Step 5: Analyze Optimal Parameters and Shutdown Cluster

Once all tests are complete, analyze the collected results to find the parameter combination that showed the highest performance. You must shutdown the cluster when work is finished.

```python
# Find result with highest Sharpe ratio
best_result = max(all_results, key=lambda res: res.get('sharpe_ratio', 0))

print("\n=== Best Parameters Found ===")
print(f"Best Params: {best_result['params']}")
print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")

# Shutdown cluster
cluster_manager.shutdown_cluster()
```

Using QuantBT's Ray integration features, you can perform time-consuming parameter optimization tasks very efficiently. In the next tutorial, we'll explore 'Bayesian optimization', a more intelligent optimization method. 