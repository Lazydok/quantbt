"""
백테스팅 설정

백테스팅 실행을 위한 설정 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: str
    initial_cash: float
    commission_rate: float
    slippage_rate: float
    is_multi_timeframe: bool = False
    data_dir: Optional[str] = None
    save_portfolio_history: bool = False

    def to_dict(self):
        """
        Ray 액터에 전달할 수 있는 직렬화 가능한 딕셔너리를 반환합니다.
        """
        return {
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timeframe": self.timeframe,
            "initial_cash": self.initial_cash,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "is_multi_timeframe": self.is_multi_timeframe,
            "save_portfolio_history": self.save_portfolio_history,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        딕셔너리로부터 BacktestConfig 객체를 생성합니다.
        """
        return cls(
            symbols=data["symbols"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            timeframe=data["timeframe"],
            initial_cash=data["initial_cash"],
            commission_rate=data["commission_rate"],
            slippage_rate=data["slippage_rate"],
            is_multi_timeframe=data.get("is_multi_timeframe", False),
            data_dir=data.get("data_dir"),
            save_portfolio_history=data.get("save_portfolio_history", False)
        ) 