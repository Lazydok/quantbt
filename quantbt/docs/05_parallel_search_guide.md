# 튜토리얼 5: 파라미터 병렬 탐색 (Grid Search)

전략의 성과는 파라미터(예: 이동평균 기간, RSI 값 등)에 따라 크게 달라집니다. 최적의 파라미터 조합을 찾기 위해 다양한 값들을 테스트하는 과정을 '파라미터 최적화'라고 합니다. 이 튜토리얼에서는 여러 파라미터 조합을 여러 CPU 코어에서 병렬로 동시에 테스트하여 최적화 시간을 획기적으로 단축하는 방법을 소개합니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 05_parallel_search.ipynb](../examples/05_parallel_search.ipynb)**

## 1. 파라미터 최적화와 병렬 처리

**그리드 서치(Grid Search)**는 파라미터마다 테스트할 값의 범위를 정하고, 가능한 모든 조합을 하나씩 테스트하여 가장 좋은 성과를 내는 조합을 찾는 가장 기본적인 최적화 방법입니다. 하지만 조합의 수가 많아지면 테스트에 엄청난 시간이 소요될 수 있습니다.

QuantBT는 **Ray** 라이브러리와의 통합을 통해 이 과정을 여러 CPU 코어에 분산시켜 동시에 실행(병렬 처리)함으로써 최적화 시간을 획기적으로 줄여줍니다. Ray를 직접 활용하므로 다소 복잡해 보일 수 있지만, 매우 강력하고 유연한 최적화 환경을 구축할 수 있습니다.

## 2. 병렬 탐색 과정

### 단계 1: Ray 클러스터 초기화

병렬 처리를 위해 Ray 클러스터를 설정하고 시작합니다. `RayClusterManager`가 이 과정을 관리합니다.

```python
from quantbt.ray import RayClusterManager

# 클러스터 설정 (사용할 CPU 코어 수 등)
ray_cluster_config = { "num_cpus": 8 } 
cluster_manager = RayClusterManager(ray_cluster_config)
cluster_manager.initialize_cluster()
```

### 단계 2: 최적화 대상 정의 (전략, 파라미터)

테스트할 전략과 파라미터 조합을 정의합니다.

```python
from quantbt import TradingStrategy
import numpy as np
from itertools import product

# 1. 최적화할 전략 (예: SimpleSMAStrategy)
class SimpleSMAStrategy(TradingStrategy):
    def __init__(self, buy_sma: int, sell_sma: int):
        super().__init__()
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
    # ... (지표 계산 및 신호 생성 로직) ...

# 2. 테스트할 파라미터 그리드 생성
param_grid = {
    'buy_sma': np.arange(10, 21, 5),  # [10, 15, 20]
    'sell_sma': np.arange(30, 51, 10) # [30, 40, 50]
}
param_combinations = list(product(param_grid['buy_sma'], param_grid['sell_sma']))
# 결과: [(10, 30), (10, 40), ..., (20, 50)]
```

### 단계 3: 데이터 공유 및 Actor 준비

백테스트에 사용할 데이터를 한 번만 로드하여 모든 병렬 작업(Actor)이 메모리에서 공유하도록 설정합니다. 이는 효율성을 극대화하는 핵심 단계입니다.

```python
from quantbt.ray import RayDataManager
from quantbt.ray.backtest_actor import BacktestActor

# 1. 데이터 매니저를 통해 데이터를 Ray의 공유 메모리에 로드
data_manager = RayDataManager.remote()
data_ref = data_manager.load_real_data.remote(...)

# 2. 병렬 작업을 수행할 BacktestActor들을 생성
# 각 Actor는 공유 데이터의 참조(data_ref)를 가짐
num_actors = cluster_manager.get_available_resources()['cpu']
actors = [BacktestActor.remote(f"actor_{i}", shared_data_ref=data_ref) for i in range(num_actors)]
```

### 단계 4: 병렬 백테스트 실행

준비된 Actor들에게 파라미터 조합을 하나씩 할당하여 백테스트를 비동기적으로 실행시킵니다.

```python
import asyncio

# 각 Actor가 수행할 작업 리스트 생성
tasks = []
for i, (buy_sma, sell_sma) in enumerate(param_combinations):
    actor = actors[i % len(actors)] # Actor들에게 순환 분배
    params = {"buy_sma": buy_sma, "sell_sma": sell_sma}
    task = actor.execute_backtest.remote(params, SimpleSMAStrategy)
    tasks.append(task)

# 모든 작업이 끝날 때까지 대기하고 결과 수집
all_results = await asyncio.gather(*tasks)
```

### 단계 5: 최적 파라미터 분석 및 클러스터 종료

모든 테스트가 완료되면, 수집된 결과를 분석하여 가장 높은 성과를 보인 파라미터 조합을 찾습니다. 작업이 끝나면 반드시 클러스터를 종료해야 합니다.

```python
# 샤프 지수가 가장 높은 결과 찾기
best_result = max(all_results, key=lambda res: res.get('sharpe_ratio', 0))

print("\n=== Best Parameters Found ===")
print(f"Best Params: {best_result['params']}")
print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")

# 클러스터 종료
cluster_manager.shutdown_cluster()
```

이처럼 QuantBT의 Ray 통합 기능을 활용하면, 시간이 많이 소요되는 파라미터 최적화 작업을 매우 효율적으로 수행할 수 있습니다. 다음 튜토리얼에서는 더 지능적인 최적화 방법인 '베이지안 최적화'에 대해 알아보겠습니다. 