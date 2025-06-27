"""
Backtesting Engine Interface

Defines core interfaces for backtesting execution.
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
    """Backtesting engine interface"""
    
    def set_strategy(self, strategy: IStrategy) -> None:
        """Set strategy
        
        Args:
            strategy: Strategy to use for backtesting
        """
        ...
    
    def set_data_provider(self, data_provider: IDataProvider) -> None:
        """Set data provider
        
        Args:
            data_provider: Market data provider
        """
        ...
    
    def set_broker(self, broker: IBroker) -> None:
        """Set broker
        
        Args:
            broker: Order execution broker
        """
        ...
    
    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run backtesting
        
        Args:
            config: Backtesting configuration
            
        Returns:
            Backtesting result
        """
        ...
    
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Add progress callback
        
        Args:
            callback: Progress callback function (progress: float, message: str)
        """
        ...


class BacktestEngineBase(ABC):
    """Base backtesting engine class"""
    
    def __init__(self, name: str = "BacktestEngine"):
        self.name = name
        self.strategy: Optional[IStrategy] = None
        self.data_provider: Optional[IDataProvider] = None
        self.broker: Optional[IBroker] = None
        self.progress_callbacks: list[Callable[[float, str], None]] = []
        self._is_running = False
        
    def set_strategy(self, strategy: IStrategy) -> None:
        """Set strategy"""
        self.strategy = strategy
        
    def set_data_provider(self, data_provider: IDataProvider) -> None:
        """Set data provider"""
        self.data_provider = data_provider
        
    def set_broker(self, broker: IBroker) -> None:
        """Set broker"""
        self.broker = broker
        
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Add progress callback"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: float, message: str) -> None:
        """Notify progress"""
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                pass  # Silently ignore callback errors
    
    def create_progress_bar(self, total: int, desc: str = "Processing", disable: bool = False) -> tqdm:
        """Create tqdm progress bar
        
        Args:
            total: Total number of items
            desc: Progress bar description
            disable: Whether to disable progress bar
            
        Returns:
            tqdm progress bar object
        """
        return tqdm(total=total, desc=desc, unit="item", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   disable=disable)
    
    def update_progress_bar(self, pbar: tqdm, message: str = "") -> None:
        """Update tqdm progress bar and notify callbacks
        
        Args:
            pbar: tqdm progress bar object
            message: Additional message
        """
        pbar.update(1)
        if message:
            pbar.set_description_str(message)
        
        if pbar.total and pbar.total > 0:
            progress = pbar.n / pbar.total
            current_desc = message if message else ""
            self._notify_progress(progress, current_desc)
    
    def _validate_components(self) -> None:
        """Validate components"""
        if self.strategy is None:
            raise ValueError("Strategy not set")
        if self.data_provider is None:
            raise ValueError("Data provider not set")
        if self.broker is None:
            raise ValueError("Broker not set")
    
    @abstractmethod
    def _execute_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute backtesting - to be implemented by subclasses
        
        Example usage:
            # Check data count
            data_count = len(market_data)
            
            # Create tqdm progress bar
            pbar = self.create_progress_bar(data_count, "Backtesting Progress")
            
            try:
                for i, data_point in enumerate(market_data):
                    # Perform backtesting logic
                    self._process_data_point(data_point)
                    
                    # Update progress
                    self.update_progress_bar(pbar, f"Processing... {i+1}/{data_count}")
                    
            finally:
                pbar.close()
        """
        pass
    
    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run backtesting"""
        if self._is_running:
            raise RuntimeError("Backtest is already running")
        
        try:
            self._is_running = True
            
            # Validate components
            self._validate_components()
            
            # Execute backtesting
            result = self._execute_backtest(config)
            
            return result
        finally:
            self._is_running = False
    
    @property
    def is_running(self) -> bool:
        """Whether running"""
        return self._is_running 