"""
백테스팅 결과

백테스팅 실행 결과를 담는 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import polars as pl

from ..entities.trade import Trade
from ..entities.position import Portfolio
from .backtest_config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    """백테스팅 결과"""
    
    # 기본 정보
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    
    # 성과 지표
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 최종 상태
    final_portfolio: Portfolio
    final_equity: float
    
    # 상세 데이터 (선택적)
    trades: Optional[List[Trade]] = None
    portfolio_history: Optional[pl.DataFrame] = None
    equity_curve: Optional[pl.DataFrame] = None
    
    # 추가 메타데이터
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """실행 시간 (초)"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.final_equity - self.config.initial_cash
    
    @property
    def total_return_pct(self) -> float:
        """총 수익률 (%)"""
        return self.total_return * 100
    
    @property
    def annual_return_pct(self) -> float:
        """연간 수익률 (%)"""
        return self.annual_return * 100
    
    @property
    def volatility_pct(self) -> float:
        """변동성 (%)"""
        return self.volatility * 100
    
    @property
    def max_drawdown_pct(self) -> float:
        """최대 낙폭 (%)"""
        return self.max_drawdown * 100
    
    @property
    def win_rate_pct(self) -> float:
        """승률 (%)"""
        return self.win_rate * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            
            # 성과 지표
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annual_return": self.annual_return,
            "annual_return_pct": self.annual_return_pct,
            "volatility": self.volatility,
            "volatility_pct": self.volatility_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            
            # 거래 통계
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "win_rate_pct": self.win_rate_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            
            # 최종 상태
            "final_portfolio": self.final_portfolio.to_dict(),
            "final_equity": self.final_equity,
            "total_pnl": self.total_pnl,
            
            # 메타데이터
            "metadata": self.metadata or {}
        }
        
        # 거래 내역 포함 (선택적)
        if self.trades:
            result["trades"] = [trade.to_dict() for trade in self.trades]
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """요약 정보 반환"""
        return {
            "기간": f"{self.config.start_date.date()} ~ {self.config.end_date.date()}",
            "초기자본": f"{self.config.initial_cash:,.0f}",
            "최종자본": f"{self.final_equity:,.0f}",
            "총수익률": f"{self.total_return_pct:.2f}%",
            "연간수익률": f"{self.annual_return_pct:.2f}%",
            "변동성": f"{self.volatility_pct:.2f}%",
            "샤프비율": f"{self.sharpe_ratio:.2f}",
            "최대낙폭": f"{self.max_drawdown_pct:.2f}%",
            "총거래수": self.total_trades,
            "승률": f"{self.win_rate_pct:.1f}%",
            "수익인수": f"{self.profit_factor:.2f}",
            "실행시간": f"{self.duration:.2f}초"
        }
    
    def print_summary(self) -> None:
        """요약 정보 출력"""
        print("=" * 50)
        print("백테스팅 결과 요약")
        print("=" * 50)
        
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key:12}: {value}")
        
        print("=" * 50) 