"""
전략 인터페이스

전략과 관련된 핵심 인터페이스와 기본 클래스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional, Union
from datetime import datetime
import polars as pl

from ..entities.market_data import MarketDataBatch, MultiTimeframeDataBatch
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
    
    def on_data(self, data: MarketDataBatch) -> List[Order]:
        """데이터 수신 시 호출
        
        Args:
            data: 시장 데이터 배치 (지표 포함)
            
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
        """지표 사전 계산 - 기본 구현"""
        # 심볼별로 그룹화하여 지표 계산
        enriched_data = data.group_by("symbol").map_groups(
            lambda group: self._compute_indicators_for_symbol(group)
        )
        
        # 시간순 정렬
        return enriched_data.sort(["timestamp", "symbol"])
    
    @abstractmethod
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """심볼별 지표 계산 - 서브클래스에서 구현"""
        pass
        
    def initialize(self, context: BacktestContext) -> None:
        """기본 초기화"""
        self.context = context
        self.reset_state()
    
    @abstractmethod
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 서브클래스에서 구현"""
        pass
    
    def on_data(self, data: MarketDataBatch) -> List[Order]:
        """데이터 처리"""
        # 상태 업데이트
        if self.context:
            latest_time = data.data.select("timestamp").max().item()
            self.context.current_time = latest_time
        
        # 신호 생성 (지표는 이미 계산됨)
        orders = self.generate_signals(data)
        
        # 주문 후처리 (validation, risk management 등)
        return self.post_process_orders(orders, data)
    
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
    
    def post_process_orders(
        self, 
        orders: List[Order], 
        data: MarketDataBatch
    ) -> List[Order]:
        """주문 후처리 (검증, 리스크 관리 등)"""
        validated_orders = []
        
        for order in orders:
            if self.validate_order(order, data):
                validated_orders.append(order)
        
        return validated_orders
    
    def validate_order(self, order: Order, data: MarketDataBatch) -> bool:
        """주문 유효성 검증"""
        # 기본 검증 로직
        if order.quantity <= 0:
            return False
        
        if order.symbol not in data.symbols:
            return False
        
        # 추가 검증 로직은 서브클래스에서 구현 가능
        return True
    
    def get_current_price(self, symbol: str, data: MarketDataBatch) -> Optional[float]:
        """현재 가격 조회"""
        latest_data = data.get_latest(symbol)
        return latest_data.close if latest_data else None
    
    def get_indicator_value(self, symbol: str, indicator_name: str, data: MarketDataBatch) -> Optional[float]:
        """지표 값 조회"""
        latest_data = data.get_latest_with_indicators(symbol)
        if latest_data and indicator_name in latest_data:
            return latest_data[indicator_name]
        return None
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        return self.config.get(key, default)
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """디버그 로그"""
        # 실제 로깅 시스템과 연동 가능
        timestamp = self.context.current_time if self.context else datetime.now()
        print(f"[{timestamp}] {self.name}: {message}", kwargs)
    
    # 유틸리티 메서드들
    def calculate_sma(self, prices: pl.Series, window: int) -> pl.Series:
        """단순 이동평균 계산"""
        return prices.rolling_mean(window_size=window)
    
    def calculate_ema(self, prices: pl.Series, span: int) -> pl.Series:
        """지수 이동평균 계산"""
        return prices.ewm_mean(span=span)
    
    def calculate_rsi(self, prices: pl.Series, period: int = 14) -> pl.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)
        
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class TradingStrategy(StrategyBase):
    """일반적인 트레이딩 전략 기본 클래스"""
    
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
        
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        portfolio_value: float
    ) -> float:
        """포지션 크기 계산"""
        target_value = portfolio_value * self.position_size_pct
        return target_value / price if price > 0 else 0.0
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 가치 조회"""
        # 브로커가 연결되어 있으면 실제 값 조회
        if hasattr(self, '_broker') and self._broker:
            return self._broker.get_portfolio().equity
        # 아니면 컨텍스트의 초기 자본 사용
        elif self.context:
            return self.context.initial_cash
        # 최후의 수단으로 설정값 사용
        return self.get_config_value("initial_cash", 100000.0)
    
    def get_current_positions(self) -> Dict[str, float]:
        """현재 포지션 조회"""
        # 브로커가 연결되어 있으면 실제 포지션 조회
        if hasattr(self, '_broker') and self._broker:
            portfolio = self._broker.get_portfolio()
            positions = {}
            for symbol, position in portfolio.active_positions.items():
                positions[symbol] = position.quantity
            return positions
        # 아니면 상태에서 조회
        return self.state.get("positions", {})
    
    def set_broker(self, broker) -> None:
        """브로커 연결"""
        self._broker = broker
    
    def validate_order(self, order: Order, data: MarketDataBatch) -> bool:
        """트레이딩 전략용 주문 검증"""
        if not super().validate_order(order, data):
            return False
        
        # 최대 포지션 수 체크
        current_positions = len(self.get_current_positions())
        if current_positions >= self.max_positions and self.is_new_position_order(order):
            return False
        
        return True
    
    def is_new_position_order(self, order: Order) -> bool:
        """새로운 포지션 생성 주문인지 확인"""
        positions = self.get_current_positions()
        return order.symbol not in positions or positions[order.symbol] == 0


class MultiTimeframeTradingStrategy(TradingStrategy):
    """멀티타임프레임 지원 트레이딩 전략 기본 클래스"""
    
    def __init__(
        self,
        name: str,
        timeframes: List[str],
        config: Optional[Dict[str, Any]] = None,
        position_size_pct: float = 0.1,
        max_positions: int = 10,
        primary_timeframe: Optional[str] = None
    ):
        """멀티타임프레임 전략 초기화
        
        Args:
            name: 전략명
            timeframes: 사용할 타임프레임 리스트 (예: ["1m", "5m", "1h"])
            config: 설정 딕셔너리
            position_size_pct: 포지션 크기 비율
            max_positions: 최대 포지션 수
            primary_timeframe: 주요 타임프레임 (None이면 가장 작은 것 자동 선택)
        """
        super().__init__(name, config, position_size_pct, max_positions)
        
        self.timeframes = timeframes or ["1m"]
        self.primary_timeframe = primary_timeframe
        
        # 타임프레임별 지표 컬럼 추적
        self.timeframe_indicators: Dict[str, List[str]] = {}
        
        # 유효성 검증
        from ..utils.timeframe import TimeframeUtils
        valid_timeframes = TimeframeUtils.filter_valid_timeframes(self.timeframes)
        if not valid_timeframes:
            raise ValueError(f"No valid timeframes provided: {self.timeframes}")
        
        self.timeframes = valid_timeframes
        
        # primary_timeframe 설정
        if self.primary_timeframe is None or self.primary_timeframe not in self.timeframes:
            # 가장 작은 타임프레임을 primary로 설정
            timeframe_minutes = [(tf, TimeframeUtils.get_timeframe_minutes(tf)) for tf in self.timeframes]
            self.primary_timeframe = min(timeframe_minutes, key=lambda x: x[1])[0]
    
    def precompute_indicators_multi_timeframe(
        self, 
        timeframe_data: Dict[str, pl.DataFrame]
    ) -> Dict[str, pl.DataFrame]:
        """멀티타임프레임 지표 사전 계산
        
        Args:
            timeframe_data: 타임프레임별 원시 데이터 딕셔너리
            
        Returns:
            타임프레임별 지표가 추가된 데이터 딕셔너리
        """
        enriched_data = {}
        
        for timeframe, data in timeframe_data.items():
            if timeframe in self.timeframes:
                # 해당 타임프레임에 대한 지표 계산
                enriched_df = self._compute_indicators_for_timeframe(data, timeframe)
                enriched_data[timeframe] = enriched_df
                
                # 지표 컬럼 추적
                base_columns = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
                indicator_cols = [col for col in enriched_df.columns if col not in base_columns]
                self.timeframe_indicators[timeframe] = indicator_cols
        
        return enriched_data
    
    def _compute_indicators_for_timeframe(
        self, 
        data: pl.DataFrame, 
        timeframe: str
    ) -> pl.DataFrame:
        """특정 타임프레임에 대한 지표 계산
        
        Args:
            data: 해당 타임프레임의 원시 데이터
            timeframe: 타임프레임
            
        Returns:
            지표가 추가된 데이터프레임
        """
        # 심볼별로 그룹화하여 지표 계산
        enriched_data = data.group_by("symbol").map_groups(
            lambda group: self._compute_indicators_for_symbol_and_timeframe(group, timeframe)
        )
        
        # 시간순 정렬
        return enriched_data.sort(["timestamp", "symbol"])
    
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str
    ) -> pl.DataFrame:
        """심볼별, 타임프레임별 지표 계산 - 서브클래스에서 구현
        
        Args:
            symbol_data: 특정 심볼의 데이터
            timeframe: 타임프레임
            
        Returns:
            지표가 추가된 데이터프레임
        """
        # 기본 구현: 기존 단일 타임프레임 메서드 호출
        return self._compute_indicators_for_symbol(symbol_data)
    
    def generate_signals_multi_timeframe(
        self, 
        multi_data: MultiTimeframeDataBatch
    ) -> List[Order]:
        """멀티타임프레임 데이터 기반 신호 생성 - 서브클래스에서 구현
        
        Args:
            multi_data: 멀티타임프레임 데이터 배치
            
        Returns:
            주문 리스트
        """
        # 기본 구현: primary 타임프레임으로 단일 타임프레임 신호 생성
        primary_batch = multi_data.get_primary_batch()
        if primary_batch:
            return self.generate_signals(primary_batch)
        return []
    
    def on_data_multi_timeframe(self, multi_data: MultiTimeframeDataBatch) -> List[Order]:
        """멀티타임프레임 데이터 처리
        
        Args:
            multi_data: 멀티타임프레임 데이터 배치
            
        Returns:
            주문 리스트
        """
        # 상태 업데이트
        if self.context:
            start_time, end_time = multi_data.timestamp_range
            self.context.current_time = end_time
        
        # 멀티타임프레임 신호 생성
        orders = self.generate_signals_multi_timeframe(multi_data)
        
        # 주문 후처리 (primary 타임프레임 배치 사용)
        primary_batch = multi_data.get_primary_batch()
        if primary_batch:
            return self.post_process_orders(orders, primary_batch)
        
        return orders
    
    def on_data(self, data: Union[MarketDataBatch, MultiTimeframeDataBatch]) -> List[Order]:
        """데이터 처리 - 단일/멀티 타임프레임 모두 지원
        
        Args:
            data: 시장 데이터 배치
            
        Returns:
            주문 리스트
        """
        if isinstance(data, MultiTimeframeDataBatch):
            return self.on_data_multi_timeframe(data)
        else:
            return super().on_data(data)
    
    def get_timeframe_price(
        self, 
        symbol: str, 
        timeframe: str, 
        price_type: str = "close",
        data: Optional[MultiTimeframeDataBatch] = None
    ) -> Optional[float]:
        """특정 타임프레임의 가격 조회
        
        Args:
            symbol: 심볼명
            timeframe: 타임프레임
            price_type: 가격 유형
            data: 멀티타임프레임 데이터 (None이면 현재 상태에서 조회)
            
        Returns:
            가격 또는 None
        """
        if data:
            return data.get_timeframe_price(timeframe, symbol, price_type)
        # 현재 상태에서 조회하는 로직은 추후 구현
        return None
    
    def get_timeframe_indicator(
        self,
        symbol: str,
        timeframe: str, 
        indicator: str,
        data: Optional[MultiTimeframeDataBatch] = None
    ) -> Optional[float]:
        """특정 타임프레임의 지표값 조회
        
        Args:
            symbol: 심볼명
            timeframe: 타임프레임
            indicator: 지표명
            data: 멀티타임프레임 데이터 (None이면 현재 상태에서 조회)
            
        Returns:
            지표값 또는 None
        """
        if data:
            return data.get_timeframe_indicator(timeframe, indicator, symbol)
        # 현재 상태에서 조회하는 로직은 추후 구현
        return None
    
    def get_cross_timeframe_signal(
        self,
        symbol: str,
        long_tf: str,
        short_tf: str,
        indicator: str,
        data: Optional[MultiTimeframeDataBatch] = None
    ) -> Optional[Dict[str, float]]:
        """크로스 타임프레임 신호 비교
        
        Args:
            symbol: 심볼명
            long_tf: 장기 타임프레임
            short_tf: 단기 타임프레임
            indicator: 지표명
            data: 멀티타임프레임 데이터
            
        Returns:
            신호 딕셔너리 또는 None
        """
        if data:
            return data.get_cross_timeframe_signal(symbol, long_tf, short_tf, indicator)
        return None
    
    def get_indicator_summary(
        self,
        symbol: str,
        indicator: str,
        data: Optional[MultiTimeframeDataBatch] = None
    ) -> Dict[str, Optional[float]]:
        """모든 타임프레임에서 지표 요약
        
        Args:
            symbol: 심볼명
            indicator: 지표명
            data: 멀티타임프레임 데이터
            
        Returns:
            타임프레임별 지표값 딕셔너리
        """
        if data:
            return data.get_indicator_summary(indicator, symbol)
        return {}
    
    @property
    def supported_timeframes(self) -> List[str]:
        """지원되는 타임프레임 목록"""
        return self.timeframes.copy()
    
    def log_debug(self, message: str, timeframe: Optional[str] = None, **kwargs: Any) -> None:
        """디버그 로그 (타임프레임 정보 포함)"""
        timestamp = self.context.current_time if self.context else datetime.now()
        tf_info = f"[{timeframe}]" if timeframe else ""
        print(f"[{timestamp}] {self.name}{tf_info}: {message}", kwargs) 