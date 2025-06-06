"""
백테스팅 설정

백테스팅 실행을 위한 설정 값 객체입니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


@dataclass(frozen=True)
class BacktestConfig:
    """백테스팅 설정"""
    
    # 기본 설정
    start_date: datetime
    end_date: datetime
    initial_cash: float
    symbols: List[str]
    
    # 데이터 설정 - 멀티 타임프레임 지원
    timeframes: Optional[List[str]] = None  # 멀티 타임프레임 리스트
    timeframe: Optional[str] = None         # 하위 호환성용 단일 타임프레임
    primary_timeframe: Optional[str] = None # 주요 타임프레임 (기준점)
    data_source: str = "csv"
    
    # 브로커 설정
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    
    # 실행 설정
    benchmark_symbol: Optional[str] = None
    save_trades: bool = True
    save_portfolio_history: bool = True
    
    # 시각화 설정
    visualization_mode: bool = False  # True일 때 상세 데이터 수집 (메모리 사용량 증가)
    
    # 추가 설정
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """설정 유효성 검증 및 정규화"""
        # 타임프레임 정규화
        self._normalize_timeframes()
        
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
    
    def _normalize_timeframes(self) -> None:
        """타임프레임 정규화 및 하위 호환성 처리"""
        # timeframes와 timeframe 둘 다 None인 경우 기본값 설정
        if self.timeframes is None and self.timeframe is None:
            object.__setattr__(self, 'timeframes', ["1D"])
            object.__setattr__(self, 'primary_timeframe', "1D")
            object.__setattr__(self, 'timeframe', "1D")
            return
        
        # timeframes가 None이고 timeframe이 있는 경우 (하위 호환성)
        if self.timeframes is None and self.timeframe is not None:
            object.__setattr__(self, 'timeframes', [self.timeframe])
            object.__setattr__(self, 'primary_timeframe', self.timeframe)
            return
        
        # timeframes가 있는 경우
        if self.timeframes is not None:
            if not self.timeframes:
                raise ValueError("timeframes는 빈 리스트일 수 없습니다")
            
            # 중복 제거 및 정렬
            from ..utils.timeframe import TimeframeUtils
            valid_timeframes = []
            
            for tf in self.timeframes:
                if TimeframeUtils.validate_timeframe(tf):
                    normalized_tf = TimeframeUtils._normalize_timeframe(tf)
                    if normalized_tf not in valid_timeframes:
                        valid_timeframes.append(normalized_tf)
                else:
                    raise ValueError(f"지원되지 않는 타임프레임: {tf}")
            
            # 시간 순으로 정렬 (작은 것부터)
            timeframe_minutes = [(tf, TimeframeUtils.get_timeframe_minutes(tf)) for tf in valid_timeframes]
            timeframe_minutes.sort(key=lambda x: x[1])
            sorted_timeframes = [tf for tf, _ in timeframe_minutes]
            
            object.__setattr__(self, 'timeframes', sorted_timeframes)
            
            # primary_timeframe 설정
            if self.primary_timeframe is None:
                # 가장 작은 타임프레임을 primary로 설정
                object.__setattr__(self, 'primary_timeframe', sorted_timeframes[0])
            elif self.primary_timeframe not in sorted_timeframes:
                raise ValueError(f"primary_timeframe '{self.primary_timeframe}'가 timeframes에 없습니다")
            
            # 하위 호환성을 위해 timeframe도 설정 (항상 primary_timeframe으로 덮어쓰기)
            object.__setattr__(self, 'timeframe', self.primary_timeframe)
    
    @property
    def is_multi_timeframe(self) -> bool:
        """멀티 타임프레임 백테스팅 여부"""
        return self.timeframes is not None and len(self.timeframes) > 1
    
    @property
    def base_timeframe(self) -> str:
        """기준 타임프레임 (가장 작은 타임프레임)"""
        return self.timeframes[0] if self.timeframes else "1D"
    
    @property
    def highest_timeframe(self) -> str:
        """최상위 타임프레임 (가장 큰 타임프레임)"""
        return self.timeframes[-1] if self.timeframes else "1D"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "timeframe": self.timeframe,
            "primary_timeframe": self.primary_timeframe,
            "data_source": self.data_source,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "benchmark_symbol": self.benchmark_symbol,
            "save_trades": self.save_trades,
            "save_portfolio_history": self.save_portfolio_history,
            "visualization_mode": self.visualization_mode,
            "metadata": self.metadata or {},
            "is_multi_timeframe": self.is_multi_timeframe,
            "base_timeframe": self.base_timeframe,
            "highest_timeframe": self.highest_timeframe,
            "timeframe_count": self.timeframe_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """딕셔너리에서 생성"""
        return cls(
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_cash=data["initial_cash"],
            symbols=data["symbols"],
            timeframes=data.get("timeframes"),
            timeframe=data.get("timeframe"),
            primary_timeframe=data.get("primary_timeframe"),
            data_source=data.get("data_source", "csv"),
            commission_rate=data.get("commission_rate", 0.001),
            slippage_rate=data.get("slippage_rate", 0.0001),
            benchmark_symbol=data.get("benchmark_symbol"),
            save_trades=data.get("save_trades", True),
            save_portfolio_history=data.get("save_portfolio_history", True),
            visualization_mode=data.get("visualization_mode", False),
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
    
    @property
    def timeframe_count(self) -> int:
        """타임프레임 개수"""
        return len(self.timeframes) if self.timeframes else 1 