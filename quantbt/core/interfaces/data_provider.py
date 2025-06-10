"""
데이터 제공자 인터페이스

데이터 제공과 관련된 핵심 인터페이스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, AsyncIterator, Optional, Dict
from datetime import datetime
import polars as pl

from ..entities.market_data import MarketDataBatch

# 멀티타임프레임 지원을 위한 import 추가
try:
    from ..utils.timeframe import TimeframeUtils
    from ..entities.market_data import MultiTimeframeDataBatch
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False


class IDataProvider(Protocol):
    """데이터 제공자 인터페이스"""
    
    async def get_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> pl.DataFrame:
        """시장 데이터 조회
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframe: 시간 프레임
            
        Returns:
            시장 데이터 DataFrame
        """
        ...
    
    async def get_data_stream(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> AsyncIterator[MarketDataBatch]:
        """시장 데이터 스트림
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframe: 시간 프레임
            
        Yields:
            시간순으로 정렬된 시장 데이터 배치
        """
        ...
    
    async def get_multi_timeframe_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframes: List[str],
        base_timeframe: str = "1m"
    ) -> Dict[str, pl.DataFrame]:
        """멀티타임프레임 데이터 조회
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframes: 목표 타임프레임 리스트 (예: ["5m", "1h", "1D"])
            base_timeframe: 기준 타임프레임 (리샘플링 소스)
            
        Returns:
            타임프레임별 데이터 딕셔너리 {"1m": df, "5m": df, "1h": df}
        """
        ...
    
    async def get_multi_timeframe_stream(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframes: List[str],
        base_timeframe: str = "1m"
    ) -> AsyncIterator:
        """멀티타임프레임 데이터 스트림
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframes: 목표 타임프레임 리스트
            base_timeframe: 기준 타임프레임
            
        Yields:
            시간순으로 정렬된 멀티타임프레임 데이터 배치
        """
        ...
    
    def get_symbols(self) -> List[str]:
        """사용 가능한 심볼 목록"""
        ...
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """심볼 유효성 검증"""
        ...
    
    def supports_timeframe(self, timeframe: str) -> bool:
        """타임프레임 지원 여부 확인"""
        ...


class DataProviderBase(ABC):
    """데이터 제공자 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self._available_symbols: Optional[List[str]] = None
    
    def get_symbols(self) -> List[str]:
        """사용 가능한 심볼 목록"""
        if self._available_symbols is None:
            self._available_symbols = self._load_available_symbols()
        return self._available_symbols
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """심볼 유효성 검증"""
        available_symbols = set(self.get_symbols())
        return all(symbol in available_symbols for symbol in symbols)
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """시간 프레임 유효성 검증"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        return timeframe.lower() in [tf.lower() for tf in valid_timeframes]
    
    def supports_timeframe(self, timeframe: str) -> bool:
        """타임프레임 지원 여부 확인"""
        return self.validate_timeframe(timeframe)
    
    def validate_date_range(self, start: datetime, end: datetime) -> bool:
        """날짜 범위 유효성 검증"""
        return start <= end and start <= datetime.now()
    
    @abstractmethod
    def _load_available_symbols(self) -> List[str]:
        """사용 가능한 심볼 로드 - 서브클래스에서 구현"""
        pass
    
    @abstractmethod
    async def _fetch_raw_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> pl.DataFrame:
        """원시 데이터 조회 - 서브클래스에서 구현"""
        pass
    
    async def get_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> pl.DataFrame:
        """시장 데이터 조회"""
        # 유효성 검증
        if not self.validate_symbols(symbols):
            raise ValueError(f"Invalid symbols: {symbols}")
        
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        if not self.validate_date_range(start, end):
            raise ValueError(f"Invalid date range: {start} to {end}")
        
        # 데이터 조회
        raw_data = await self._fetch_raw_data(symbols, start, end, timeframe)
        
        # 데이터 후처리
        return self._post_process_data(raw_data)
    
    def _post_process_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """데이터 후처리"""
        # 기본적인 후처리 (정렬, 중복 제거 등)
        return data.sort(["symbol", "timestamp"]).unique()
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """시간 프레임 정규화"""
        return timeframe.upper()
    
    async def get_multi_timeframe_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframes: List[str],
        base_timeframe: str = "1m"
    ) -> Dict[str, pl.DataFrame]:
        """멀티타임프레임 데이터 조회"""
        if not MULTI_TIMEFRAME_AVAILABLE:
            raise ImportError("MultiTimeframe components not available. Please ensure TimeframeUtils is properly installed.")
        
        # 유효성 검증
        if not self.validate_symbols(symbols):
            raise ValueError(f"Invalid symbols: {symbols}")
        
        for tf in [base_timeframe] + timeframes:
            if not self.validate_timeframe(tf):
                raise ValueError(f"Invalid timeframe: {tf}")
        
        if not self.validate_date_range(start, end):
            raise ValueError(f"Invalid date range: {start} to {end}")
        
        # 기준 타임프레임 데이터 조회
        base_data = await self._fetch_raw_data(symbols, start, end, base_timeframe)
        base_data = self._post_process_data(base_data)
        
        # 멀티타임프레임 데이터 생성
        result = {}
        
        # TimeframeUtils를 사용하여 효율적인 멀티 타임프레임 데이터 생성
        result = TimeframeUtils.create_multi_timeframe_data(
            base_data=base_data,
            timeframes=timeframes,
            base_timeframe=base_timeframe
        )
        
        return result