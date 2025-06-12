"""
시장 데이터 엔티티

시장 데이터를 표현하는 핵심 엔티티들을 정의합니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any


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