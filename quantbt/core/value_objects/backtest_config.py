"""
백테스팅 설정

백테스팅 실행을 위한 설정 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class BacktestConfig:
    """백테스팅 설정"""
    
    # 기본 설정
    start_date: datetime
    end_date: datetime
    initial_cash: float
    symbols: List[str]
    
    # 데이터 설정
    timeframe: str = "1D"
    data_source: str = "csv"
    
    # 브로커 설정
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    
    # 실행 설정
    benchmark_symbol: Optional[str] = None
    save_trades: bool = True
    save_portfolio_history: bool = True
    
    # 추가 설정
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """설정 유효성 검증"""
        if self.start_date >= self.end_date:
            raise ValueError("시작일은 종료일보다 이전이어야 합니다")
        
        if self.initial_cash <= 0:
            raise ValueError("초기 자본은 0보다 커야 합니다")
        
        if not self.symbols:
            raise ValueError("최소 하나의 심볼이 필요합니다")
        
        if self.commission_rate < 0:
            raise ValueError("수수료율은 0 이상이어야 합니다")
        
        if self.slippage_rate < 0:
            raise ValueError("슬리피지율은 0 이상이어야 합니다")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "data_source": self.data_source,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "benchmark_symbol": self.benchmark_symbol,
            "save_trades": self.save_trades,
            "save_portfolio_history": self.save_portfolio_history,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """딕셔너리에서 생성"""
        return cls(
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_cash=data["initial_cash"],
            symbols=data["symbols"],
            timeframe=data.get("timeframe", "1D"),
            data_source=data.get("data_source", "csv"),
            commission_rate=data.get("commission_rate", 0.001),
            slippage_rate=data.get("slippage_rate", 0.0001),
            benchmark_symbol=data.get("benchmark_symbol"),
            save_trades=data.get("save_trades", True),
            save_portfolio_history=data.get("save_portfolio_history", True),
            metadata=data.get("metadata")
        )
    
    @property
    def duration_days(self) -> int:
        """백테스팅 기간 (일수)"""
        return (self.end_date - self.start_date).days
    
    @property
    def symbol_count(self) -> int:
        """심볼 개수"""
        return len(self.symbols) 