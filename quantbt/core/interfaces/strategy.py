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
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 기본 구현 (Dict Native 자동 변환)
        
        서브클래스에서 generate_signals_dict() 구현 시 자동으로 호출됨
        """
        # 서브클래스에서 generate_signals_dict() 구현했는지 확인
        # 기본 구현이 아닌 경우 Dict 방식 우선 사용
        if hasattr(self, 'generate_signals_dict') and \
           self.__class__.generate_signals_dict != TradingStrategy.generate_signals_dict:
            # 첫 번째 심볼의 데이터로 Dict Native 방식 호출
            for symbol in data.symbols:
                current_data = self._convert_batch_to_dict_basic(data, symbol)
                if current_data:
                    return self.generate_signals_dict(current_data)
        
        # 기본 동작: 빈 주문 리스트 반환
        return []
    
    def _convert_batch_to_dict_basic(self, data: MarketDataBatch, symbol: str) -> Optional[Dict[str, Any]]:
        """MarketDataBatch에서 특정 심볼의 최신 데이터를 Dict로 변환 (기본 버전)"""
        try:
            latest_data = data.get_latest(symbol)
            if latest_data:
                return {
                    'symbol': symbol,
                    'timestamp': latest_data.timestamp,
                    'open': latest_data.open,
                    'high': latest_data.high,
                    'low': latest_data.low,
                    'close': latest_data.close,
                    'volume': latest_data.volume
                }
        except:
            pass
        return None
    
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
    """트레이딩 전략 클래스
    
    특징:
    - 지표 계산: Polars 벡터연산
    - 신호 생성: Dict Native 방식
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
        self._broker = None
        self._legacy_warning_shown = False
    
    # === 1. 브로커 연결 ===
    def set_broker(self, broker) -> None:
        """브로커 설정"""
        self._broker = broker
    
    # === 2. 포지션 관리 ===
    def get_current_positions(self) -> Dict[str, float]:
        """현재 포지션 조회"""
        if self._broker:
            portfolio = self._broker.get_portfolio()
            positions = {}
            for symbol, position in portfolio.positions.items():
                if hasattr(position, 'quantity') and position.quantity != 0:
                    positions[symbol] = position.quantity
            return positions
        return {}
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 가치 조회"""
        if self._broker:
            return self._broker.get_portfolio().equity
        return self.get_config_value("initial_cash", 100000.0)
    
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        portfolio_value: float
    ) -> float:
        """포지션 크기 계산"""
        target_value = portfolio_value * self.position_size_pct
        return target_value / price if price > 0 else 0.0
    
    def generate_signals_dict(self, current_data: Dict[str, Any], 
                            historical_data: Optional[List[Dict[str, Any]]] = None) -> List[Order]:
        """Dict 기반 신호 생성 (서브클래스에서 구현)
        
        Args:
            current_data: 현재 캔들 데이터 (지표 포함)
            historical_data: 과거 데이터 (옵션)
            
        Returns:
            생성된 주문 리스트
        """
        # 기본 구현: 빈 주문 리스트 반환
        # 서브클래스에서 이 메서드를 구현하면 최고 성능으로 작동
        return []
    
    
    def _convert_batch_to_dict(self, data: MarketDataBatch, symbol: str) -> Optional[Dict[str, Any]]:
        """MarketDataBatch에서 특정 심볼의 최신 데이터를 Dict로 변환"""
        try:
            latest_data = data.get_latest(symbol)
            if latest_data:
                result = {
                    'symbol': symbol,
                    'timestamp': latest_data.timestamp,
                    'open': latest_data.open,
                    'high': latest_data.high,
                    'low': latest_data.low,
                    'close': latest_data.close,
                    'volume': latest_data.volume
                }
                
                # 지표 데이터 추가 (만약 있다면)
                if hasattr(latest_data, '_additional_fields'):
                    result.update(latest_data._additional_fields)
                
                # 지표 컬럼들 추가 (만약 있다면)
                for col in self.indicator_columns:
                    if hasattr(latest_data, col):
                        result[col] = getattr(latest_data, col)
                
                return result
        except Exception as e:
            print(f"Warning: Failed to convert batch to dict for {symbol}: {e}")
        return None
    
    def calculate_sma_dict(self, prices: List[float], window: int) -> List[float]:
        """Dict 기반 단순이동평균 계산"""
        sma_values = []
        for i in range(len(prices)):
            if i < window - 1:
                sma_values.append(None)
            else:
                window_prices = prices[i - window + 1:i + 1]
                sma = sum(window_prices) / len(window_prices)
                sma_values.append(sma)
        return sma_values
    
    def calculate_ema_dict(self, prices: List[float], span: int) -> List[float]:
        """Dict 기반 지수이동평균 계산"""
        if not prices:
            return []
        
        alpha = 2.0 / (span + 1)
        ema_values = [None] * len(prices)
        
        # 첫 번째 유효한 값으로 초기화
        first_idx = 0
        while first_idx < len(prices) and prices[first_idx] is None:
            first_idx += 1
        
        if first_idx >= len(prices):
            return ema_values
        
        ema_values[first_idx] = prices[first_idx]
        
        # EMA 계산
        for i in range(first_idx + 1, len(prices)):
            if prices[i] is not None:
                prev_ema = ema_values[i-1]
                if prev_ema is not None:
                    ema_values[i] = alpha * prices[i] + (1 - alpha) * prev_ema
                else:
                    ema_values[i] = prices[i]
        
        return ema_values
    
    def calculate_rsi_dict(self, prices: List[float], period: int = 14) -> List[float]:
        """Dict 기반 RSI 계산"""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        # 가격 변화 계산
        deltas = []
        for i in range(1, len(prices)):
            if prices[i] is not None and prices[i-1] is not None:
                deltas.append(prices[i] - prices[i-1])
            else:
                deltas.append(None)
        
        # 상승/하락 분리
        gains = [max(d, 0) if d is not None else None for d in deltas]
        losses = [-min(d, 0) if d is not None else None for d in deltas]
        
        # 평균 계산
        avg_gains = self.calculate_sma_dict(gains, period)
        avg_losses = self.calculate_sma_dict(losses, period)
        
        # RSI 계산
        rsi_values = [None] * (len(prices))
        for i in range(len(avg_gains)):
            if avg_gains[i] is not None and avg_losses[i] is not None:
                if avg_losses[i] == 0:
                    rsi_values[i + 1] = 100.0
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi_values
    
    # === 6. 주문 검증 강화 ===
    def validate_order(self, order: Order, data: MarketDataBatch) -> bool:
        """주문 유효성 검증"""
        # 기본 검증
        if not super().validate_order(order, data):
            return False
        
        # 포지션 제한 검증
        current_positions = self.get_current_positions()
        
        # 새로운 포지션 주문인 경우
        if self.is_new_position_order(order):
            if len(current_positions) >= self.max_positions:
                return False
        
        return True
    
    def is_new_position_order(self, order: Order) -> bool:
        """새로운 포지션을 여는 주문인지 확인"""
        current_positions = self.get_current_positions()
        symbol_position = current_positions.get(order.symbol, 0.0)
        
        # 매수 주문이고 현재 포지션이 없는 경우
        if order.side.name == "BUY" and symbol_position <= 0:
            return True
        
        # 매도 주문이고 현재 숏 포지션이 없는 경우 (공매도)
        if order.side.name == "SELL" and symbol_position >= 0:
            return True
        
        return False