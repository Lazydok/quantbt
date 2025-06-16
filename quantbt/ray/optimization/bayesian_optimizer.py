from typing import Dict, Any, List, Union
import numpy as np
from skopt import Optimizer
from quantbt.ray.optimization.parameter_space import ParameterSpace

class BayesianOptimizer:
    """
    scikit-optimize를 사용한 베이지안 최적화를 관리하는 클래스.

    이 클래스는 파라미터 공간(ParameterSpace)을 입력받아 최적화 과정을
    담당하며, 다음 탐색 지점을 제안하고(ask), 결과를 받아 모델을
    업데이트하는(tell) 인터페이스를 제공합니다.
    """

    def __init__(self,
                 space: ParameterSpace,
                 n_initial_points: int = 10,
                 acq_func: str = 'gp_hedge',
                 **skopt_kwargs):
        """
        BayesianOptimizer를 초기화합니다.

        Args:
            space (ParameterSpace): 최적화를 수행할 파라미터 공간.
            n_initial_points (int): 초기 랜덤 탐색 지점의 수.
            acq_func (str): 획득 함수(Acquisition Function).
            **skopt_kwargs: skopt.Optimizer에 전달할 추가 인자.
        """
        self.space = space.space
        self.param_names = space.dimension_names
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        
        self.optimizer = Optimizer(
            dimensions=self.space,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            **skopt_kwargs
        )

    def ask(self, n_points: int = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        평가할 다음 파라미터 포인트를 제안받습니다.
        n_points가 지정되면 해당 개수만큼의 포인트를 리스트로 반환합니다.
        """
        if n_points is None:
            # skopt는 파라미터 값의 리스트를 반환
            points = self.optimizer.ask(n_points=1)
            # 파라미터 이름과 매핑하여 딕셔너리로 변환
            return dict(zip(self.param_names, points[0]))
        else:
            points_list = self.optimizer.ask(n_points=n_points)
            return [dict(zip(self.param_names, p)) for p in points_list]

    def tell(self, params: Union[Dict[str, Any], List[Dict[str, Any]]], score: Union[float, List[float]]):
        """
        파라미터 포인트에 대한 평가 결과를 옵티마이저에 알립니다.
        배치(리스트) 또는 단일 포인트(딕셔너리) 입력을 모두 처리합니다.
        """
        if isinstance(params, list):
            # Batch update
            param_values = [[p[name] for name in self.param_names] for p in params]
            self.optimizer.tell(param_values, score)
        else:
            # Single point update
            param_values = [params[name] for name in self.param_names]
            self.optimizer.tell(param_values, score)

    def get_best_params(self) -> Dict[str, Any]:
        """
        현재까지의 최적 파라미터를 반환합니다.

        Returns:
            Dict[str, Any]: 최적 파라미터 딕셔너리.
        """
        best_values = self.optimizer.Xi[np.argmin(self.optimizer.yi)]
        return dict(zip(self.param_names, best_values))

    def get_best_score(self) -> float:
        """
        현재까지의 최고 점수를 반환합니다.

        Returns:
            float: 최고 점수.
        """
        return np.min(self.optimizer.yi)

    def get_result(self):
        """
        # ... existing code ...
        """