"""
Core Interfaces

핵심 인터페이스들을 정의합니다.
"""

from .backtest_engine import IBacktestEngine, BacktestEngineBase

__all__ = [
    "IBacktestEngine",
    "BacktestEngineBase",
] 