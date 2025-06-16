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

QuantBT는 `BayesianParameterOptimizer` 클래스를 통해 베이지안 최적화 기능을 제공합니다.

### 단계 1: 목표 함수 정의

최적화의 대상이 되는 목표 함수를 정의합니다. 이 함수는 파라미터를 인자로 받아 백테스팅을 수행하고, 최대화하려는 값(예: `sharpe_ratio`)을 반환해야 합니다. **주의: 베이지안 최적화 라이브러리는 보통 최소값을 찾도록 설계되어 있으므로, 최대화를 위해 수익률이나 샤프 지수에 -1을 곱하여 반환하는 것이 일반적입니다.**

```python
def objective_function(fast_sma, slow_sma):
    """베이지안 최적화를 위한 목표 함수"""
    # 파라미터 유효성 검사
    if fast_sma >= slow_sma:
        return -1e9 # 무효한 조합에 대해 매우 낮은 값 반환
        
    result = qbt.backtest(...) # 백테스팅 실행
    
    # 샤프 지수를 최대화하는 것이 목표
    return result.sharpe_ratio
```

### 단계 2: 파라미터 탐색 공간 정의

최적화할 각 파라미터의 탐색 범위를 정의합니다.

```python
param_space = {
    'fast_sma': (10, 50),     # 10에서 50 사이의 정수
    'slow_sma': (50, 200)     # 50에서 200 사이의 정수
}
```

### 단계 3: 최적화 실행

`BayesianParameterOptimizer`를 사용하여 최적화를 실행합니다.

```python
from quantbt.ray import BayesianParameterOptimizer

# 최적화기 생성
optimizer = BayesianParameterOptimizer(
    objective_function,
    param_space
)

# 최적화 실행 (초기 5회 랜덤 탐색, 25회 최적화 시도)
best_params, all_results = optimizer.run_optimization(
    init_points=5, 
    n_iter=25
)
```

## 3. 결과 분석

최적화가 완료되면 가장 좋은 성능을 보인 파라미터와 전체 탐색 과정을 확인할 수 있습니다.

```python
print("=== Best Parameters Found ===")
print(f"Fast SMA: {best_params['fast_sma']}")
print(f"Slow SMA: {best_params['slow_sma']}")

# 전체 결과 데이터프레임으로 보기
print(all_results.df)
```

베이지안 최적화는 그리드 서치에 비해 훨씬 적은 반복으로도 우수한 파라미터를 찾아낼 수 있어, 복잡한 전략의 최적화 시간을 크게 단축시켜 줍니다. 