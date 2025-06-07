"""
그리드 서치 설정

병렬 백테스팅을 위한 그리드 서치 설정을 담는 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Type, Callable
from itertools import product
from .backtest_config import BacktestConfig


@dataclass(frozen=True)
class GridSearchConfig:
    """그리드 서치 설정"""
    
    # 기본 백테스트 설정
    base_config: BacktestConfig
    
    # 전략 설정
    strategy_class: Type  # 전략 클래스
    strategy_params: Dict[str, List[Any]]  # 파라미터명: 값 리스트 매핑
    fixed_params: Optional[Dict[str, Any]] = None  # 고정 파라미터들
    
    # 병렬 처리 설정
    max_workers: Optional[int] = None    # None이면 CPU 코어 수 사용
    batch_size: int = 10                # 배치 처리 크기
    
    # 최적화 설정
    optimization_metric: str = "calmar_ratio"  # 최적화 기준 지표
    min_trades: int = 10                       # 최소 거래 횟수 (필터링용)
    
    # 결과 저장 설정
    save_detailed_results: bool = False        # 모든 결과 저장 여부
    save_top_n: int = 10                      # 상위 N개 결과만 상세 저장
    
    # 메타데이터
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def total_combinations(self) -> int:
        """총 파라미터 조합 수"""
        if not self.strategy_params:
            return 0
        
        total = 1
        for param_values in self.strategy_params.values():
            total *= len(param_values)
        return total
    
    @property
    def parameter_combinations(self) -> List[Dict[str, Any]]:
        """모든 파라미터 조합 생성"""
        if not self.strategy_params:
            return [{}]
        
        # 파라미터 이름과 값들 추출
        param_names = list(self.strategy_params.keys())
        param_values = list(self.strategy_params.values())
        
        # 모든 조합 생성
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # 고정 파라미터 추가
            if self.fixed_params:
                param_dict.update(self.fixed_params)
            
            # 유효성 검사 (필요시 하위 클래스에서 오버라이드)
            if self._is_valid_combination(param_dict):
                combinations.append(param_dict)
        
        return combinations
    
    @property 
    def valid_combinations(self) -> int:
        """유효한 조합 수"""
        return len(self.parameter_combinations)
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """파라미터 조합 유효성 검사 (하위 클래스에서 오버라이드 가능)"""
        return True
    
    def get_batch_combinations(self, batch_idx: int) -> List[Dict[str, Any]]:
        """배치별 파라미터 조합 반환"""
        combinations = self.parameter_combinations
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(combinations))
        return combinations[start_idx:end_idx]
    
    @property
    def total_batches(self) -> int:
        """총 배치 수"""
        import math
        return math.ceil(self.valid_combinations / self.batch_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "base_config": self.base_config.to_dict(),
            "strategy_class": self.strategy_class.__name__ if self.strategy_class else None,
            "strategy_params": self.strategy_params,
            "fixed_params": self.fixed_params or {},
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "optimization_metric": self.optimization_metric,
            "min_trades": self.min_trades,
            "save_detailed_results": self.save_detailed_results,
            "save_top_n": self.save_top_n,
            "total_combinations": self.total_combinations,
            "valid_combinations": self.valid_combinations,
            "total_batches": self.total_batches,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def create_basic(
        cls,
        strategy_class: Type,
        symbols: List[str], 
        start_date: datetime,
        end_date: datetime,
        strategy_params: Dict[str, List[Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        initial_cash: float = 10_000_000,
        **kwargs
    ) -> "GridSearchConfig":
        """기본 그리드 서치 설정 생성"""
        
        # 기본 백테스트 설정
        base_config = BacktestConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            initial_cash=initial_cash,
            commission_rate=0.001,
            slippage_rate=0.001,
            save_portfolio_history=False,  # 그리드 서치에서는 기본적으로 히스토리 저장 안함
            **kwargs
        )
        
        return cls(
            base_config=base_config,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            fixed_params=fixed_params
        )


class SMAGridSearchConfig(GridSearchConfig):
    """SMA 전략 전용 그리드 서치 설정"""
    
    def _is_valid_combination(self, params: Dict[str, Any]) -> bool:
        """SMA 전략의 유효성 검사: buy_sma <= sell_sma"""
        buy_sma = params.get('buy_sma')
        sell_sma = params.get('sell_sma')
        
        if buy_sma is not None and sell_sma is not None:
            return buy_sma <= sell_sma
        
        return True
    
    @classmethod
    def create_sma_config(
        cls,
        strategy_class: Type,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        buy_sma_range: List[int] = None,
        sell_sma_range: List[int] = None,
        initial_cash: float = 10_000_000,
        **kwargs
    ) -> "SMAGridSearchConfig":
        """SMA 전략용 편의 생성자"""
        
        # 기본 범위 설정
        if buy_sma_range is None:
            buy_sma_range = [5, 10, 15, 20]
        if sell_sma_range is None:
            sell_sma_range = [10, 20, 30, 40]
        
        strategy_params = {
            'buy_sma': buy_sma_range,
            'sell_sma': sell_sma_range
        }
        
        return cls.create_basic(
            strategy_class=strategy_class,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            initial_cash=initial_cash,
            **kwargs
        ) 