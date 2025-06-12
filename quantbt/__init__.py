"""
QuantBT - 고성능 퀀트 트레이딩 백테스팅 엔진

모듈화되고 확장 가능한 백테스팅 프레임워크
"""

__version__ = "0.1.0"
__author__ = "QuantBT Team"

# 현재 구현된 엔티티들만 export
from .core.entities.market_data import MarketData
from .core.entities.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from .core.entities.position import Position, Portfolio
from .core.entities.trade import Trade

# 유틸리티
from .core.utils.timeframe import TimeframeUtils

# 전략 기본 클래스
from .core.interfaces.strategy import StrategyBase, TradingStrategy, MultiTimeframeTradingStrategy, BacktestContext

# 백테스팅 설정 및 결과
from .core.value_objects.backtest_config import BacktestConfig
from .core.value_objects.backtest_result import BacktestResult
from .core.value_objects.grid_search_config import GridSearchConfig
from .core.value_objects.grid_search_result import GridSearchResult, GridSearchSummary

# 백테스팅 엔진
from .core.interfaces.backtest_engine import IBacktestEngine, BacktestEngineBase
from .infrastructure.engine.base_engine import BacktestEngine

# 데이터 제공자
from .infrastructure.data.csv_provider import CSVDataProvider
from .infrastructure.data.upbit_provider import UpbitDataProvider

# 브로커
from .infrastructure.brokers.simple_broker import SimpleBroker

# 샘플 전략
from .examples.strategies.simple_sma_strategy import SimpleSMAStrategy

__all__ = [
    # 엔티티
    "MarketData",
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
    
    # 백테스팅
    "BacktestConfig",
    "BacktestResult",
    "GridSearchConfig",
    "GridSearchResult",
    "GridSearchSummary",
    "IBacktestEngine",
    "BacktestEngineBase",
    "BacktestEngine",
    
    # 인프라스트럭처
    "CSVDataProvider",
    "UpbitDataProvider",
    "SimpleBroker",
    
    # 샘플 전략
    "SimpleSMAStrategy",
] 