# 튜토리얼 6: 베이지안 파라미터 최적화

이전 튜토리얼의 그리드 서치는 모든 파라미터 조합을 무작위로 테스트하기 때문에 비효율적일 수 있습니다. '베이지안 최적화(Bayesian Optimization)'는 더 적은 시도로 더 나은 파라미터를 찾기 위한 지능적인 탐색 방법입니다. 이전 테스트 결과를 바탕으로 다음 테스트할 파라미터를 확률적으로 예측하여 가장 성능이 좋을 것으로 기대되는 영역을 집중적으로 탐색합니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 06_bayesian_optimization.ipynb](../examples/06_bayesian_optimization.ipynb)**

## 1. 베이지안 최적화의 개념

베이지안 최적화는 '목표 함수(Objective Function)'의 값을 최대화하는 파라미터를 찾는 것을 목표로 합니다.
1.  **초기 샘플링**: 파라미터 공간에서 몇 개의 포인트를 무작위로 선택하여 목표 함수(백테스팅)를 실행합니다.
2.  **확률 모델링**: 현재까지의 (파라미터, 성능) 데이터를 바탕으로 아직 탐색하지 않은 공간의 성능을 예측하는 확률 모델(보통 가우시안 프로세스)을 만듭니다.
3.  **획득 함수 (Acquisition Function)**: 확률 모델을 사용하여 '다음 시도에서 현재 최고 성능을 넘어설 확률'이 가장 높은 파라미터를 계산합니다.
4.  **반복**: 3에서 찾은 파라미터로 실제 백테스팅을 수행하고, 결과를 2의 데이터에 추가하여 모델을 업데이트합니다. 이 과정을 정해진 횟수만큼 반복합니다.

이 방식을 통해 무의미한 파라미터 조합에 대한 테스트를 줄이고, 가능성 높은 영역에 집중하여 효율적으로 최적해를 찾아갑니다.

## 2. 베이지안 최적화 과정

QuantBT는 `BayesianParameterOptimizer` 클래스를 통해 베이지안 최적화 기능을 통합적으로 제공합니다. Ray 클러스터 위에서 병렬로 실행되며, 사용자는 최적화할 전략과 파라미터 공간만 정의하면 됩니다.

### 단계 1: 최적화 대상 정의 (전략 클래스, 파라미터 공간)

먼저 최적화할 전략 클래스를 정의하고, 파라미터들의 탐색 범위를 `ParameterSpace`를 사용하여 지정합니다.

```python
from quantbt.core.interfaces.strategy import TradingStrategy
from quantbt.ray.optimization.parameter_space import ParameterSpace

# 1. 최적화할 전략 클래스 정의
class SimpleSMAStrategy(TradingStrategy):
    def __init__(self, buy_sma: int, sell_sma: int, position_size_pct: float):
        super().__init__()
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.position_size_pct = position_size_pct
    # ... (지표 계산 및 신호 생성 로직은 내부적으로 동일) ...

# 2. 파라미터 탐색 공간 정의
param_config = {
    'buy_sma': (10, 100),            # 10에서 100 사이의 정수
    'sell_sma': (20, 200),           # 20에서 200 사이의 정수
    'position_size_pct': (0.5, 1.0)  # 0.5에서 1.0 사이의 실수
}
param_space = ParameterSpace.from_dict(param_config)
```

### 단계 2: 최적화기 생성 및 실행

`BayesianParameterOptimizer`에 전략 클래스, 파라미터 공간, 백테스트 설정을 전달하여 인스턴스를 생성한 후, `optimize` 메소드를 비동기적으로 호출합니다.

```python
from quantbt.ray.bayesian_parameter_optimizer import BayesianParameterOptimizer
from quantbt.core.value_objects.backtest_config import BacktestConfig
import asyncio
from datetime import datetime

# 1. 기본 백테스트 설정
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe="4h",
    initial_cash=10_000_000
)

# 2. 최적화기 생성
optimizer = BayesianParameterOptimizer(
    strategy_class=SimpleSMAStrategy,
    param_space=param_space,
    config=config,
    num_actors=8, # 사용할 CPU 코어 수
    n_initial_points=10 # 초기 랜덤 탐색 횟수
)

# 3. 최적화 비동기 실행
async def run_optimization():
    results = await optimizer.optimize(
        objective_metric='sharpe_ratio', # 최적화 목표 지표
        n_iter=100 # 총 최적화 시도 횟수
    )
    return results

# 비동기 함수 실행 (Jupyter Notebook 등에서는 await 바로 사용 가능)
all_results = asyncio.run(run_optimization())
```

### 단계 3: 결과 분석

최적화가 완료되면 `optimize` 함수는 모든 시도에 대한 결과를 담은 리스트를 반환합니다. 이 리스트를 분석하여 가장 좋은 성능을 보인 파라미터를 찾을 수 있습니다.

```python
# 샤프 지수가 가장 높은 결과 찾기
best_result = max(all_results, key=lambda x: x['result'].get('sharpe_ratio', -999))

print("=== Best Parameters Found ===")
print(f"Params: {best_result['params']}")
print(f"Sharpe Ratio: {best_result['result']['sharpe_ratio']:.4f}")
print(f"Total Return: {best_result['result']['total_return']:.4f}")
```

베이지안 최적화는 그리드 서치에 비해 훨씬 적은 반복으로도 우수한 파라미터를 찾아낼 수 있어, 복잡한 전략의 최적화 시간을 크게 단축시켜 줍니다. 