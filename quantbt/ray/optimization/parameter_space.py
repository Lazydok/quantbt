from typing import List, Optional, Dict, Any, Tuple, Union
from skopt.space import Dimension, Space, Real, Integer, Categorical

class ParameterSpace:
    """
    scikit-optimize의 파라미터 공간을 관리하는 클래스.

    이 클래스는 skopt.space.Dimension 객체 리스트를 받아
    파라미터 공간을 생성하고 관리하는 역할을 합니다.
    """

    def __init__(self, dimensions: List[Dimension]):
        """
        ParameterSpace를 초기화합니다.

        Args:
            dimensions: skopt.space.Dimension 객체의 리스트.
                        (e.g., [Real(0.1, 0.9), Integer(10, 100)])
        
        Raises:
            TypeError: dimensions 리스트에 Dimension 객체가 아닌 다른 타입이 포함된 경우.
        """
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError("모든 차원(dimensions)은 skopt.space.Dimension 객체여야 합니다.")
            
        self.dimensions = dimensions
        self.space = Space(dimensions)
        self._name_to_dimension = {dim.name: dim for dim in dimensions}

    def get_dimension(self, name: str) -> Optional[Dimension]:
        """
        이름으로 특정 차원(dimension) 객체를 반환합니다.

        Args:
            name: 찾고자 하는 차원의 이름.

        Returns:
            skopt.space.Dimension 객체. 해당 이름의 차원이 없으면 None을 반환합니다.
        """
        return self._name_to_dimension.get(name)

    @property
    def dimension_names(self) -> List[str]:
        """
        파라미터 공간에 포함된 모든 차원의 이름 리스트를 반환합니다.
        """
        return list(self._name_to_dimension.keys())

    def __len__(self) -> int:
        """
        파라미터 공간의 차원 수를 반환합니다.
        """
        return len(self.dimensions)

    @classmethod
    def from_dict(cls, param_dict: Dict[str, Union[Tuple, List]]) -> 'ParameterSpace':
        """
        딕셔너리에서 ParameterSpace 객체를 생성하는 클래스 메서드.

        Args:
            param_dict: 파라미터 이름과 범위를 정의한 딕셔너리.
                - ('low', 'high'): Integer 또는 Real
                - ('low', 'high', 'prior'): Real
                - ['cat1', 'cat2', ...]: Categorical

        Returns:
            ParameterSpace: 생성된 ParameterSpace 객체.
        """
        dimensions = []
        for name, value in param_dict.items():
            if isinstance(value, (list, tuple)):
                # 리스트는 Categorical로 처리
                if isinstance(value, list):
                    dimensions.append(Categorical(value, name=name))
                # 튜플은 타입에 따라 Real 또는 Integer로 처리
                elif isinstance(value, tuple):
                    if len(value) == 2 and all(isinstance(v, int) for v in value):
                        dimensions.append(Integer(value[0], value[1], name=name))
                    elif len(value) >= 2 and any(isinstance(v, float) for v in value):
                        dimensions.append(Real(value[0], value[1], name=name, prior=value[2] if len(value) > 2 else 'uniform'))
                    else:
                         dimensions.append(Categorical(value, name=name))

            else:
                raise TypeError(f"파라미터 '{name}'의 타입이 유효하지 않습니다. 튜플 또는 리스트여야 합니다.")
        
        return cls(dimensions) 