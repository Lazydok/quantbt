"""
QuantBT - 고성능 퀀트 트레이딩 백테스팅 엔진

모듈화되고 확장 가능한 백테스팅 프레임워크
"""

__version__ = "0.1.0"
__author__ = "QuantBT Team"

# 현재 구현된 엔티티들만 export
from .core.entities.market_data import MarketData, MarketDataBatch, MultiTimeframeDataBatch
from .core.entities.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from .core.entities.position import Position, Portfolio
from .core.entities.trade import Trade

# 유틸리티
from .core.utils.timeframe import TimeframeUtils

# 전략 기본 클래스
from .core.interfaces.strategy import StrategyBase, TradingStrategy, BacktestContext

# Dict 기반 고성능 전략 (신규 추가)
from .core.strategies.dict_based import DictTradingStrategy

# 백테스팅 설정 및 결과
from .core.value_objects.backtest_config import BacktestConfig
from .core.value_objects.backtest_result import BacktestResult
from .core.value_objects.grid_search_config import GridSearchConfig
from .core.value_objects.grid_search_result import GridSearchResult, GridSearchSummary

# 백테스팅 엔진 - Dict Native가 기본!
from .core.interfaces.backtest_engine import IBacktestEngine, BacktestEngineBase
from .infrastructure.engine.base_engine import BacktestEngine  # 기본 엔진

# 데이터 제공자
from .infrastructure.data.csv_provider import CSVDataProvider
from .infrastructure.data.upbit_provider import UpbitDataProvider

# 브로커
from .infrastructure.brokers.simple_broker import SimpleBroker

# 샘플 전략
from .examples.strategies.sma_dict_strategy import SimpleSMAStrategyDict

__all__ = [
    # 엔티티
    "MarketData",
    "MarketDataBatch", 
    "MultiTimeframeDataBatch",
    "Order",
    "OrderType",
    "OrderSide", 
    "OrderStatus",
    "TimeInForce",
    "Position",
    "Portfolio",
    "Trade",
    
    # 유틸리티
    "TimeframeUtils",
    
    # 전략
    "StrategyBase",
    "TradingStrategy",
    "MultiTimeframeTradingStrategy",
    "BacktestContext",
    
    
    "DictTradingStrategy",
    
    # 백테스팅
    "BacktestConfig",
    "BacktestResult",
    "GridSearchConfig",
    "SMAGridSearchConfig", 
    "GridSearchResult",
    "GridSearchSummary",
    "IBacktestEngine",
    "BacktestEngineBase",
    "BacktestEngine",  # 기본 엔진
    
    # 인프라스트럭처
    "CSVDataProvider",
    "UpbitDataProvider",
    "SimpleBroker",
    
    "SimpleSMAStrategyDict",
] 