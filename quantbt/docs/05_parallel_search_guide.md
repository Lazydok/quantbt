# 튜토리얼 5: 파라미터 병렬 탐색 (Grid Search)

전략의 성과는 파라미터(예: 이동평균 기간, RSI 값 등)에 따라 크게 달라집니다. 최적의 파라미터 조합을 찾기 위해 다양한 값들을 테스트하는 과정을 '파라미터 최적화'라고 합니다. 이 튜토리얼에서는 여러 파라미터 조합을 병렬로 동시에 테스트하여 최적화 시간을 단축하는 방법을 소개합니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 05_parallel_search.ipynb](../examples/05_parallel_search.ipynb)**

## 1. 파라미터 최적화와 병렬 처리

**그리드 서치(Grid Search)**는 파라미터마다 테스트할 값의 범위를 정하고, 가능한 모든 조합을 하나씩 테스트하여 가장 좋은 성과를 내는 조합을 찾는 가장 기본적인 최적화 방법입니다. 하지만 조합의 수가 많아지면 테스트에 엄청난 시간이 소요될 수 있습니다.

QuantBT는 **Ray** 라이브러리와의 통합을 통해 이 과정을 여러 CPU 코어에 분산시켜 동시에 실행(병렬 처리)함으로써 최적화 시간을 획기적으로 줄여줍니다.

## 2. 병렬 탐색 과정

### 단계 1: 최적화할 함수 정의

먼저, 단일 파라미터 조합에 대한 백테스팅을 수행하고 원하는 평가 지표(예: 총수익률, 샤프 지수)를 반환하는 함수를 정의합니다. 이 함수는 병렬로 실행될 작업 단위가 됩니다.

```python
import quantbt as qbt
from quantbt.strategies import SMAStrategy

def run_sma_backtest(params):
    """단일 SMA 전략 백테스트 실행 함수"""
    fast_sma, slow_sma = params
    
    # 전략 생성 및 백테스트 실행
    strategy = SMAStrategy(fast_sma=fast_sma, slow_sma=slow_sma)
    result = qbt.backtest(
        strategy=strategy,
        # ... other backtest settings ...
    )
    
    # 최적화 목표 지표 반환
    return {
        "params": params,
        "sharpe_ratio": result.sharpe_ratio,
        "total_return": result.total_return
    }
```

### 단계 2: 파라미터 조합 생성

테스트할 파라미터들의 범위를 지정하여 모든 조합의 리스트를 생성합니다.

```python
import numpy as np

# 단기 이평선: 5부터 20까지 5씩 증가
fast_sma_range = np.arange(5, 25, 5) 
# 장기 이평선: 30부터 60까지 10씩 증가
slow_sma_range = np.arange(30, 70, 10) 

# 모든 파라미터 조합 생성
param_combinations = [
    (fast, slow) 
    for fast in fast_sma_range 
    for slow in slow_sma_range 
    if fast < slow
]
# 예: [(5, 30), (5, 40), ..., (20, 60)]
```

### 단계 3: 병렬 최적화 실행

`qbt.parallel_test` 함수를 사용하여 생성된 파라미터 조합들을 병렬로 테스트합니다. Ray가 자동으로 작업을 분산시켜 처리합니다.

```python
# 병렬 테스트 실행
all_results = qbt.parallel_test(
    run_sma_backtest,
    param_combinations
)

# 결과 출력
for res in all_results:
    print(f"Params: {res['params']}, Sharpe Ratio: {res['sharpe_ratio']:.2f}")
```

## 3. 최적 파라미터 분석

모든 테스트가 완료되면, 수집된 결과 `all_results`를 분석하여 가장 높은 성과를 보인 파라미터 조합을 찾습니다.

```python
# 샤프 지수가 가장 높은 결과 찾기
best_result = max(all_results, key=lambda x: x['sharpe_ratio'])

print("\n=== Best Parameters Found ===")
print(f"Best Params: {best_result['params']}")
print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
print(f"Total Return: {best_result['total_return']:.2%}")
```

이처럼 QuantBT의 병렬 처리 기능을 활용하면, 시간이 많이 소요되는 파라미터 최적화 작업을 효율적으로 수행할 수 있습니다. 다음 튜토리얼에서는 더 지능적인 최적화 방법인 '베이지안 최적화'에 대해 알아보겠습니다. 