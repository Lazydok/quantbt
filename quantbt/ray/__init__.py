"""
QuantBT Ray 기반 분산 백테스팅 최적화 모듈

Ray를 활용한 고성능 분산 백테스팅 시스템
"""

from .cluster_manager import RayClusterManager
from .data_manager import RayDataManager
from .backtest_actor import BacktestActor
from .result_aggregator import RayResultAggregator, StatisticsCalculator, ReportGenerator
from .quantbt_engine_adapter import QuantBTEngineAdapter

__all__ = [
    'RayClusterManager',
    'RayDataManager',
    'BacktestActor',
    'RayResultAggregator',
    'StatisticsCalculator',
    'ReportGenerator',
    'QuantBTEngineAdapter'
] 