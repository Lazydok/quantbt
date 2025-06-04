"""
데이터 제공자 인터페이스

데이터 제공과 관련된 핵심 인터페이스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, AsyncIterator, Optional
from datetime import datetime
import polars as pl

from ..entities.market_data import MarketDataBatch


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
    
    def get_symbols(self) -> List[str]:
        """사용 가능한 심볼 목록"""
        ...
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """심볼 유효성 검증"""
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
    
    async def get_data_stream(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> AsyncIterator[MarketDataBatch]:
        """시장 데이터 스트림"""
        # 전체 데이터 조회
        full_data = await self.get_data(symbols, start, end, timeframe)
        
        # 시간순으로 정렬
        sorted_data = full_data.sort("timestamp")
        
        # 각 시점별로 배치 생성
        unique_timestamps = sorted_data.select("timestamp").unique().sort("timestamp")
        
        for timestamp_row in unique_timestamps.iter_rows(named=True):
            timestamp = timestamp_row["timestamp"]
            
            # 해당 시점까지의 누적 데이터
            batch_data = sorted_data.filter(pl.col("timestamp") <= timestamp)
            
            yield MarketDataBatch(
                data=batch_data,
                symbols=symbols,
                timeframe=timeframe
            )
    
    def _post_process_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """데이터 후처리"""
        # 기본적인 후처리 (정렬, 중복 제거 등)
        return data.sort(["symbol", "timestamp"]).unique()
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """시간 프레임 정규화"""
        return timeframe.upper() 