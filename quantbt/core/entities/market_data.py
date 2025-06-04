"""
시장 데이터 엔티티

시장 데이터를 표현하는 핵심 엔티티들을 정의합니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, List
import polars as pl


@dataclass(frozen=True)
class MarketData:
    """단일 시점의 시장 데이터"""
    
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "metadata": self.metadata or {}
        }
    
    @property
    def ohlc(self) -> tuple[float, float, float, float]:
        """OHLC 튜플 반환"""
        return (self.open, self.high, self.low, self.close)
    
    @property
    def typical_price(self) -> float:
        """일반적 가격 (HLC/3)"""
        return (self.high + self.low + self.close) / 3


@dataclass
class MarketDataBatch:
    """배치 형태의 시장 데이터"""
    
    data: pl.DataFrame
    symbols: List[str]
    timeframe: str
    
    def __post_init__(self) -> None:
        """데이터 유효성 검증"""
        required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """특정 심볼의 최신 데이터 조회"""
        symbol_data = self.data.filter(pl.col("symbol") == symbol)
        if symbol_data.height == 0:
            return None
        
        latest = symbol_data.row(-1, named=True)
        return MarketData(
            timestamp=latest["timestamp"],
            symbol=latest["symbol"],
            open=latest["open"],
            high=latest["high"],
            low=latest["low"],
            close=latest["close"],
            volume=latest["volume"]
        )
    
    def get_latest_with_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """특정 심볼의 최신 데이터 조회 (지표 포함)"""
        symbol_data = self.data.filter(pl.col("symbol") == symbol)
        if symbol_data.height == 0:
            return None
        
        latest = symbol_data.row(-1, named=True)
        return dict(latest)
    
    def get_symbol_data(self, symbol: str) -> pl.DataFrame:
        """특정 심볼의 모든 데이터 조회"""
        return self.data.filter(pl.col("symbol") == symbol)
    
    def get_price_dict(self, price_type: str = "close") -> Dict[str, float]:
        """심볼별 가격 딕셔너리 반환"""
        if price_type not in ["open", "high", "low", "close"]:
            raise ValueError(f"Invalid price_type: {price_type}")
        
        latest_data = self.data.group_by("symbol").last()
        return {
            row["symbol"]: row[price_type] 
            for row in latest_data.iter_rows(named=True)
        }
    
    def get_indicator_dict(self, indicator_name: str) -> Dict[str, float]:
        """심볼별 지표 값 딕셔너리 반환"""
        if indicator_name not in self.data.columns:
            raise ValueError(f"Indicator '{indicator_name}' not found in data")
        
        latest_data = self.data.group_by("symbol").last()
        return {
            row["symbol"]: row[indicator_name] 
            for row in latest_data.iter_rows(named=True)
            if row[indicator_name] is not None
        }
    
    @property
    def timestamp_range(self) -> tuple[datetime, datetime]:
        """시간 범위 반환"""
        timestamps = self.data.select("timestamp").to_series()
        return timestamps.min(), timestamps.max()
    
    @property
    def unique_symbols(self) -> List[str]:
        """고유한 심볼 목록"""
        return self.data.select("symbol").unique().to_series().to_list()
    
    @property
    def indicator_columns(self) -> List[str]:
        """지표 컬럼 목록 반환"""
        base_columns = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        return [col for col in self.data.columns if col not in base_columns] 