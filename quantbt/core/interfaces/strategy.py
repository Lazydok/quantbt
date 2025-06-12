"""
전략 인터페이스

전략과 관련된 핵심 인터페이스와 기본 클래스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional
from datetime import datetime
import polars as pl

from ..entities.order import Order
from ..entities.trade import Trade


class BacktestContext:
    """백테스팅 컨텍스트"""
    
    def __init__(self, initial_cash: float, symbols: List[str]):
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.current_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}


class IStrategy(Protocol):
    """전략 인터페이스"""
    
    def precompute_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """지표 사전 계산
        
        Args:
            data: OHLCV 원본 데이터
            
        Returns:
            지표가 추가된 데이터프레임
        """
        ...
    
    def initialize(self, context: BacktestContext) -> None:
        """전략 초기화
        
        Args:
            context: 백테스팅 컨텍스트
        """
        ...
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성
        
        Args:
            current_data: 현재 시점의 시장 데이터 (Dict 형태)
            
        Returns:
            생성할 주문 리스트
        """
        ...
    
    def on_order_fill(self, trade: Trade) -> None:
        """주문 체결 시 호출
        
        Args:
            trade: 체결된 거래 정보
        """
        ...
    
    def finalize(self, context: BacktestContext) -> None:
        """백테스팅 종료 시 호출
        
        Args:
            context: 백테스팅 컨텍스트
        """
        ...


class StrategyBase(ABC):
    """전략 기본 클래스"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.context: Optional[BacktestContext] = None
        self.state: Dict[str, Any] = {}
        self.indicator_columns: List[str] = []  # 추가된 지표 컬럼 추적
        
    def precompute_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """지표 사전 계산 - 2단계 처리 구조
        
        1단계: 심볼별 기본 지표 계산 (_compute_indicators_for_symbol)
        2단계: 심볼간 비교 지표 계산 (_compute_cross_symbol_indicators)
        """
        # 1단계: 심볼별 기본 지표 계산
        enriched_data = data.group_by("symbol").map_groups(
            lambda group: self._compute_indicators_for_symbol(group)
        )
        
        # 2단계: 심볼간 비교 지표 계산 (동일 시점 보장!)
        enriched_data = self._compute_cross_symbol_indicators(enriched_data)
        
        # 시간순 정렬
        return enriched_data.sort(["timestamp", "symbol"])
    
    @abstractmethod
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """심볼별 지표 계산 - 서브클래스에서 구현
        
        Args:
            symbol_data: 단일 심볼의 OHLCV 데이터
            
        Returns:
            지표가 추가된 데이터프레임
        """
        pass
    
    @abstractmethod 
    def _compute_cross_symbol_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """심볼간 비교 지표 계산 - 서브클래스에서 구현
        
        Args:
            data: 모든 심볼의 지표가 계산된 데이터프레임
            
        Returns:
            심볼간 비교 지표가 추가된 데이터프레임
        """
        pass
        
    def initialize(self, context: BacktestContext) -> None:
        """기본 초기화"""
        self.context = context
        self.reset_state()
    
    def on_order_fill(self, trade: Trade) -> None:
        """주문 체결 처리"""
        # 기본적으로는 아무것도 하지 않음
        # 서브클래스에서 필요시 오버라이드
        pass
    
    def finalize(self, context: BacktestContext) -> None:
        """종료 처리"""
        # 기본적으로는 아무것도 하지 않음
        # 서브클래스에서 필요시 오버라이드
        pass
    
    def reset_state(self) -> None:
        """상태 초기화"""
        self.state = {}
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        return self.config.get(key, default)
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """디버그 로그 출력"""
        if self.get_config_value('debug', False):
            print(f"[{self.name}] {message}", kwargs)
    
    # Polars 기반 벡터 연산 지표 계산 유틸리티
    def calculate_sma(self, prices: pl.Series, window: int) -> pl.Series:
        """단순 이동평균 계산"""
        return prices.rolling_mean(window_size=window)
    
    def calculate_ema(self, prices: pl.Series, span: int) -> pl.Series:
        """지수 이동평균 계산"""
        return prices.ewm_mean(span=span)
    
    def calculate_rsi(self, prices: pl.Series, period: int = 14) -> pl.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.map_elements(lambda x: max(x, 0))
        loss = delta.map_elements(lambda x: -min(x, 0))
        
        avg_gain = gain.ewm_mean(span=period)
        avg_loss = loss.ewm_mean(span=period)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TradingStrategy(StrategyBase):
    """단일 타임프레임 거래 전략
    
    단일 타임프레임에서 동작하는 기본 거래 전략 클래스입니다.
    하이브리드 방식: Polars 벡터 연산(지표) + Dict Native(신호 생성)
    """
    
    def __init__(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        position_size_pct: float = 0.1,
        max_positions: int = 10
    ):
        super().__init__(name, config)
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        
        # 브로커 연결 (백테스팅 엔진에서 설정)
        self.broker = None
    
    def _compute_cross_symbol_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """심볼간 비교 지표 계산
        
        단일 타임프레임에서는 기본적으로 변경 없이 반환
        필요한 경우 서브클래스에서 오버라이드
        """
        return data
    
    # 브로커 연동 메서드들
    def set_broker(self, broker) -> None:
        """브로커 설정"""
        self.broker = broker
    
    def get_current_positions(self) -> Dict[str, float]:
        """현재 포지션 조회"""
        if self.broker:
            portfolio = self.broker.get_portfolio()
            positions = {}
            for symbol, position in portfolio.positions.items():
                if position.quantity > 0:
                    positions[symbol] = position.quantity
            return positions
        return {}
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 총 가치 조회"""
        if self.broker:
            portfolio = self.broker.get_portfolio()
            return portfolio.equity
        return 0.0
    
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        portfolio_value: float
    ) -> float:
        """포지션 크기 계산"""
        target_value = portfolio_value * self.position_size_pct
        return target_value / price
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성 - 서브클래스에서 구현
        
        Args:
            current_data: 현재 시점의 시장 데이터 (Dict 형태)
                - symbol, timestamp, open, high, low, close, volume
                - 사전 계산된 지표들 (sma_10, sma_20, rsi_14 등)
        
        Returns:
            생성할 주문 리스트
        """
        # 기본 구현: 아무것도 하지 않음
        # 서브클래스에서 구체적인 전략 로직 구현
        return []
    
    
    def is_new_position_order(self, order: Order) -> bool:
        """새 포지션 생성 주문인지 확인"""
        current_positions = self.get_current_positions()
        current_qty = current_positions.get(order.symbol, 0)
        
        # 현재 포지션이 없거나, 반대 방향 주문인 경우 새 포지션
        if current_qty == 0:
            return True
        elif (current_qty > 0 and order.quantity < 0) or (current_qty < 0 and order.quantity > 0):
            return True
        
        return False


class MultiTimeframeTradingStrategy(TradingStrategy):
    """다중 타임프레임 거래 전략 기반 클래스
    
    여러 타임프레임의 데이터를 동시에 활용하는 거래 전략입니다.
    순차 캐싱 시스템을 통해 효율적인 멀티 타임프레임 분석을 제공합니다.
    
    특징:
    - O(1) 복잡도의 멀티 타임프레임 데이터 접근
    - 타임프레임별 독립적인 지표 계산
    - 크로스 타임프레임 신호 생성 지원
    
    Example:
        class MyStrategy(MultiTimeframeTradingStrategy):
            def __init__(self):
                super().__init__(
                    name="MultiTF",
                    timeframe_configs={
                        "1m": {"sma_windows": [10, 20]},
                        "5m": {"rsi_period": 14}
                    },
                    primary_timeframe="1m"
                )
    """
    
    def __init__(
        self,
        name: str,
        timeframe_configs: Dict[str, Dict[str, Any]],
        primary_timeframe: str = "1m",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.timeframe_configs = timeframe_configs
        self.primary_timeframe = primary_timeframe
        
        # 타임프레임별 지표 컬럼 추적
        self.timeframe_indicator_columns: Dict[str, List[str]] = {}
    
    def precompute_indicators_multi_timeframe(
        self, 
        multi_data: Dict[str, pl.DataFrame]
    ) -> Dict[str, pl.DataFrame]:
        """다중 타임프레임 지표 사전 계산
        
        Args:
            multi_data: 타임프레임별 OHLCV 데이터
                {"1m": DataFrame, "5m": DataFrame, ...}
        
        Returns:
            타임프레임별 지표가 계산된 데이터
        """
        enriched_multi_data = {}
        
        for timeframe, data in multi_data.items():
            if timeframe in self.timeframe_configs:
                config = self.timeframe_configs[timeframe]
                enriched_data = self._compute_indicators_for_timeframe(data, timeframe, config)
                enriched_multi_data[timeframe] = enriched_data
            else:
                # 설정이 없는 타임프레임은 그대로 전달
                enriched_multi_data[timeframe] = data
        
        return enriched_multi_data
    
    def _compute_indicators_for_timeframe(
        self, 
        data: pl.DataFrame, 
        timeframe: str, 
        config: Dict[str, Any]
    ) -> pl.DataFrame:
        """특정 타임프레임에 대한 지표 계산"""
        # 심볼별 지표 계산
        enriched_data = data.group_by("symbol").map_groups(
            lambda group: self._compute_indicators_for_symbol_and_timeframe(group, timeframe, config)
        )
        
        # 시간순 정렬
        return enriched_data.sort(["timestamp", "symbol"])
    
    @abstractmethod 
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str, 
        config: Dict[str, Any]
    ) -> pl.DataFrame:
        """심볼별 타임프레임별 지표 계산 - 서브클래스에서 구현
        
        Args:
            symbol_data: 단일 심볼의 특정 타임프레임 OHLCV 데이터
            timeframe: 타임프레임 ("1m", "5m" 등)
            config: 해당 타임프레임의 설정
            
        Returns:
            지표가 추가된 데이터프레임
        """
        pass
    
    @abstractmethod
    def generate_signals_multi_timeframe(
        self, 
        multi_current_data: Dict[str, Dict[str, Any]]
    ) -> List[Order]:
        """다중 타임프레임 기반 신호 생성 - 서브클래스에서 구현
        
        Args:
            multi_current_data: 타임프레임별 현재 시점 데이터
                {
                    "1m": {timestamp, symbol, close, sma_10, sma_20, volume_ratio, ...},
                    "5m": {timestamp, symbol, close, sma_5, rsi_14, volatility, ...}
                }
        
        Returns:
            생성할 주문 리스트
        """
        pass
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """단일 타임프레임 방식 호출 시 처리"""
        # Multi timeframe 전략에서는 사용하지 않음
        # 호환성을 위해 빈 리스트 반환
        return []
    
    # 단일 타임프레임 메서드들은 사용하지 않으므로 기본 구현으로 유지
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 지표 계산 - Multi timeframe에서는 사용하지 않음"""
        return symbol_data
    
    @property
    def available_timeframes(self) -> List[str]:
        """사용 가능한 타임프레임 목록"""
        return list(self.timeframe_configs.keys())
    
    @property
    def is_multi_timeframe_strategy(self) -> bool:
        """다중 타임프레임 전략 여부"""
        return True