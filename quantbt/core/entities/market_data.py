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


@dataclass
class MultiTimeframeDataBatch:
    """멀티타임프레임 마켓데이터 배치
    
    여러 타임프레임의 데이터를 동시에 관리하고 접근할 수 있는 클래스
    """
    
    timeframe_data: Dict[str, pl.DataFrame]  # {"1m": df, "5m": df, "1h": df}
    symbols: List[str]
    primary_timeframe: str = "1m"  # 주요 타임프레임
    
    def __post_init__(self) -> None:
        """데이터 유효성 검증"""
        if not self.timeframe_data:
            raise ValueError("timeframe_data cannot be empty")
        
        if self.primary_timeframe not in self.timeframe_data:
            # primary_timeframe이 없으면 가장 작은 타임프레임을 primary로 설정
            from ..utils.timeframe import TimeframeUtils
            valid_timeframes = []
            for tf in self.timeframe_data.keys():
                if TimeframeUtils.validate_timeframe(tf):
                    valid_timeframes.append((tf, TimeframeUtils.get_timeframe_minutes(tf)))
            
            if valid_timeframes:
                # 가장 작은 타임프레임을 primary로 설정
                self.primary_timeframe = min(valid_timeframes, key=lambda x: x[1])[0]
            else:
                self.primary_timeframe = list(self.timeframe_data.keys())[0]
        
        # 각 타임프레임 데이터의 필수 컬럼 검증
        required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        for timeframe, data in self.timeframe_data.items():
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {timeframe} data: {missing_columns}")

    @property
    def data(self) -> pl.DataFrame:
        """호환성을 위한 data 속성 - 주요 타임프레임의 데이터 반환"""
        return self.timeframe_data[self.primary_timeframe]
    
    def get_timeframe_data(self, timeframe: str) -> Optional[pl.DataFrame]:
        """특정 타임프레임의 전체 데이터 반환
        
        Args:
            timeframe: 조회할 타임프레임
            
        Returns:
            해당 타임프레임의 DataFrame 또는 None
        """
        return self.timeframe_data.get(timeframe)
    
    def get_timeframe_latest(self, timeframe: str, symbol: str) -> Optional[Dict[str, Any]]:
        """특정 타임프레임의 특정 심볼 최신 데이터 반환
        
        Args:
            timeframe: 조회할 타임프레임
            symbol: 조회할 심볼
            
        Returns:
            최신 데이터 딕셔너리 또는 None
        """
        data = self.timeframe_data.get(timeframe)
        if data is None:
            return None
        
        symbol_data = data.filter(pl.col("symbol") == symbol)
        if symbol_data.height == 0:
            return None
        
        latest = symbol_data.row(-1, named=True)
        return dict(latest)
    
    def get_timeframe_indicator(self, timeframe: str, indicator: str, symbol: str) -> Optional[float]:
        """특정 타임프레임의 특정 심볼 지표값 반환
        
        Args:
            timeframe: 조회할 타임프레임
            indicator: 지표명
            symbol: 심볼명
            
        Returns:
            지표값 또는 None
        """
        latest_data = self.get_timeframe_latest(timeframe, symbol)
        if latest_data is None:
            return None
        
        return latest_data.get(indicator)
    
    def get_timeframe_price(self, timeframe: str, symbol: str, price_type: str = "close") -> Optional[float]:
        """특정 타임프레임의 특정 심볼 가격 반환
        
        Args:
            timeframe: 조회할 타임프레임
            symbol: 심볼명
            price_type: 가격 유형 (open, high, low, close)
            
        Returns:
            가격 또는 None
        """
        if price_type not in ["open", "high", "low", "close"]:
            raise ValueError(f"Invalid price_type: {price_type}")
        
        latest_data = self.get_timeframe_latest(timeframe, symbol)
        if latest_data is None:
            return None
        
        return latest_data.get(price_type)
    
    def get_primary_batch(self) -> MarketDataBatch:
        """주요 타임프레임의 MarketDataBatch 반환
        
        Returns:
            주요 타임프레임의 MarketDataBatch
        """
        primary_data = self.timeframe_data[self.primary_timeframe]
        return MarketDataBatch(
            data=primary_data,
            symbols=self.symbols,
            timeframe=self.primary_timeframe
        )
    
    def get_timeframe_batch(self, timeframe: str) -> Optional[MarketDataBatch]:
        """특정 타임프레임의 MarketDataBatch 반환
        
        Args:
            timeframe: 조회할 타임프레임
            
        Returns:
            해당 타임프레임의 MarketDataBatch 또는 None
        """
        data = self.timeframe_data.get(timeframe)
        if data is None:
            return None
        
        return MarketDataBatch(
            data=data,
            symbols=self.symbols,
            timeframe=timeframe
        )
    
    def get_cross_timeframe_signal(
        self, 
        symbol: str, 
        long_tf: str, 
        short_tf: str, 
        indicator: str
    ) -> Optional[Dict[str, float]]:
        """크로스 타임프레임 신호 비교
        
        Args:
            symbol: 심볼명
            long_tf: 장기 타임프레임
            short_tf: 단기 타임프레임  
            indicator: 비교할 지표명
            
        Returns:
            {"long_value": float, "short_value": float, "ratio": float} 또는 None
        """
        long_value = self.get_timeframe_indicator(long_tf, indicator, symbol)
        short_value = self.get_timeframe_indicator(short_tf, indicator, symbol)
        
        if long_value is None or short_value is None:
            return None
        
        ratio = short_value / long_value if long_value != 0 else 0
        
        return {
            "long_value": long_value,
            "short_value": short_value,
            "ratio": ratio
        }
    
    @property
    def available_timeframes(self) -> List[str]:
        """사용 가능한 타임프레임 목록"""
        return list(self.timeframe_data.keys())
    
    @property
    def timestamp_range(self) -> tuple[datetime, datetime]:
        """전체 데이터의 시간 범위 (주요 타임프레임 기준)"""
        primary_data = self.timeframe_data[self.primary_timeframe]
        timestamps = primary_data.select("timestamp").to_series()
        return timestamps.min(), timestamps.max()
    
    def get_indicator_summary(self, indicator: str, symbol: str) -> Dict[str, Optional[float]]:
        """모든 타임프레임에서 특정 지표의 현재값 요약
        
        Args:
            indicator: 지표명
            symbol: 심볼명
            
        Returns:
            {timeframe: indicator_value} 딕셔너리
        """
        summary = {}
        for timeframe in self.timeframe_data.keys():
            summary[timeframe] = self.get_timeframe_indicator(timeframe, indicator, symbol)
        
        return summary
    
    def get_price_summary(self, symbol: str, price_type: str = "close") -> Dict[str, Optional[float]]:
        """모든 타임프레임에서 특정 가격의 현재값 요약
        
        Args:
            symbol: 심볼명
            price_type: 가격 유형
            
        Returns:
            {timeframe: price_value} 딕셔너리
        """
        summary = {}
        for timeframe in self.timeframe_data.keys():
            summary[timeframe] = self.get_timeframe_price(timeframe, symbol, price_type)
        
        return summary 