"""
백테스팅 엔진 인터페이스

백테스팅 실행을 위한 핵심 인터페이스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Callable, Any
import asyncio

from .strategy import IStrategy
from .data_provider import IDataProvider
from .broker import IBroker
from ..value_objects.backtest_config import BacktestConfig
from ..value_objects.backtest_result import BacktestResult


class IBacktestEngine(Protocol):
    """백테스팅 엔진 인터페이스"""
    
    def set_strategy(self, strategy: IStrategy) -> None:
        """전략 설정
        
        Args:
            strategy: 백테스팅에 사용할 전략
        """
        ...
    
    def set_data_provider(self, data_provider: IDataProvider) -> None:
        """데이터 제공자 설정
        
        Args:
            data_provider: 시장 데이터 제공자
        """
        ...
    
    def set_broker(self, broker: IBroker) -> None:
        """브로커 설정
        
        Args:
            broker: 주문 실행 브로커
        """
        ...
    
    async def run(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            백테스팅 결과
        """
        ...
    
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """진행률 콜백 추가
        
        Args:
            callback: 진행률 콜백 함수 (progress: float, message: str)
        """
        ...


class BacktestEngineBase(ABC):
    """백테스팅 엔진 기본 클래스"""
    
    def __init__(self, name: str = "BacktestEngine"):
        self.name = name
        self.strategy: Optional[IStrategy] = None
        self.data_provider: Optional[IDataProvider] = None
        self.broker: Optional[IBroker] = None
        self.progress_callbacks: list[Callable[[float, str], None]] = []
        self._is_running = False
        
    def set_strategy(self, strategy: IStrategy) -> None:
        """전략 설정"""
        self.strategy = strategy
        
    def set_data_provider(self, data_provider: IDataProvider) -> None:
        """데이터 제공자 설정"""
        self.data_provider = data_provider
        
    def set_broker(self, broker: IBroker) -> None:
        """브로커 설정"""
        self.broker = broker
        
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """진행률 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: float, message: str) -> None:
        """진행률 알림"""
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    def _validate_components(self) -> None:
        """컴포넌트 유효성 검증"""
        if self.strategy is None:
            raise ValueError("Strategy not set")
        if self.data_provider is None:
            raise ValueError("Data provider not set")
        if self.broker is None:
            raise ValueError("Broker not set")
    
    @abstractmethod
    async def _execute_backtest(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행 - 서브클래스에서 구현"""
        pass
    
    async def run(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행"""
        if self._is_running:
            raise RuntimeError("Backtest is already running")
        
        try:
            self._is_running = True
            
            # 컴포넌트 유효성 검증
            self._validate_components()
            
            # 백테스팅 실행
            result = await self._execute_backtest(config)
            
            return result
        finally:
            self._is_running = False
    
    @property
    def is_running(self) -> bool:
        """실행 중 여부"""
        return self._is_running 