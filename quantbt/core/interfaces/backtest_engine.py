"""
백테스팅 엔진 인터페이스

백테스팅 실행을 위한 핵심 인터페이스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Callable, Any
import asyncio
from tqdm import tqdm

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
                pass  # 콜백 에러는 조용히 무시
    
    def create_progress_bar(self, total: int, desc: str = "Processing") -> tqdm:
        """tqdm 진행률 바 생성
        
        Args:
            total: 전체 항목 수
            desc: 진행률 바 설명
            
        Returns:
            tqdm 진행률 바 객체
        """
        return tqdm(total=total, desc=desc, unit="item", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    def update_progress_bar(self, pbar: tqdm, message: str = "") -> None:
        """tqdm 진행률 바 업데이트 및 콜백 알림
        
        Args:
            pbar: tqdm 진행률 바 객체
            message: 추가 메시지
        """
        pbar.update(1)
        if pbar.total and pbar.total > 0:
            progress = pbar.n / pbar.total
            self._notify_progress(progress, message or pbar.desc)
    
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
        """백테스팅 실행 - 서브클래스에서 구현
        
        예시 사용법:
            # 데이터 개수 확인
            data_count = len(market_data)
            
            # tqdm 진행률 바 생성
            pbar = self.create_progress_bar(data_count, "백테스팅 진행")
            
            try:
                for i, data_point in enumerate(market_data):
                    # 백테스팅 로직 수행
                    await self._process_data_point(data_point)
                    
                    # 진행률 업데이트
                    self.update_progress_bar(pbar, f"처리중... {i+1}/{data_count}")
                    
            finally:
                pbar.close()
        """
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