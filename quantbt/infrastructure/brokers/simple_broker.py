"""
간단한 브로커 구현

백테스팅용 기본 브로커 구현체입니다.
"""

from typing import Optional
from datetime import datetime

from ...core.interfaces.broker import BrokerBase
from ...core.entities.order import Order, OrderType, OrderStatus
from ...core.entities.trade import Trade


class SimpleBroker(BrokerBase):
    """간단한 백테스팅 브로커"""
    
    def __init__(
        self, 
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% 수수료
        slippage_rate: float = 0.0001    # 0.01% 슬리피지
    ):
        super().__init__(initial_cash)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
    
    def _validate_order(self, order: Order) -> bool:
        """주문 유효성 검증"""
        # 기본 검증
        if order.quantity <= 0:
            return False
        
        # 시장 데이터 존재 여부 확인
        if self.current_data is None:
            return False
        
        # 심볼이 현재 데이터에 있는지 확인
        if order.symbol not in self.current_data.symbols:
            return False
        
        # 매수 주문의 경우 현금 충분성 확인
        if order.is_buy:
            current_price = self._get_current_price(order.symbol)
            if current_price is None:
                return False
            
            estimated_cost = order.quantity * current_price * (1 + self.commission_rate)
            if estimated_cost > self.portfolio.cash:
                return False
        
        # 매도 주문의 경우 포지션 충분성 확인
        if order.is_sell:
            position = self.portfolio.get_position(order.symbol)
            if position.quantity < order.quantity:
                return False
        
        return True
    
    def _can_execute_immediately(self, order: Order) -> bool:
        """즉시 실행 가능 여부 확인"""
        if self.current_data is None:
            return False
        
        current_price = self._get_current_price(order.symbol)
        if current_price is None:
            return False
        
        # 시장가 주문은 다음 바까지 대기 (미래 참조 방지)
        if order.order_type == OrderType.MARKET:
            # 시장가 주문은 즉시 실행하지 않고 다음 바에서 체결
            return False
        
        # 지정가 주문 실행 조건 확인
        if order.order_type == OrderType.LIMIT:
            if order.is_buy and current_price <= order.price:
                return True
            if order.is_sell and current_price >= order.price:
                return True
        
        # 스톱 주문 실행 조건 확인
        if order.order_type == OrderType.STOP:
            if order.is_buy and current_price >= order.stop_price:
                return True
            if order.is_sell and current_price <= order.stop_price:
                return True
        
        return False
    
    def _execute_order(self, order: Order) -> None:
        """주문 실행"""
        if not order.is_active:
            return
        
        execution_price = self._get_execution_price(order)
        if execution_price is None:
            order.reject("Cannot determine execution price")
            return
        
        # 현재 시장 데이터의 타임스탬프 사용
        current_timestamp = self._get_current_timestamp()
        
        # 주문 체결
        order.fill(
            quantity=order.remaining_quantity,
            price=execution_price,
            timestamp=current_timestamp
        )
        
        # 거래 생성 - 시장 데이터 타임스탬프 전달
        trade = self._create_trade(
            order=order,
            quantity=order.filled_quantity,
            price=execution_price,
            timestamp=current_timestamp
        )
        
        # 포트폴리오 업데이트
        self._update_portfolio_from_trade(trade)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재 가격 조회"""
        if self.current_data is None:
            return None
        
        latest_data = self.current_data.get_latest(symbol)
        return latest_data.close if latest_data else None
    
    def _get_current_timestamp(self) -> datetime:
        """현재 시장 데이터 타임스탬프 조회"""
        if self.current_data is None:
            return datetime.now()
        
        # 현재 시장 데이터의 최신 타임스탬프 사용
        try:
            # MarketDataBatch에서 최신 타임스탬프 조회
            latest_timestamp = self.current_data.data["timestamp"].max()
            if latest_timestamp:
                return latest_timestamp
        except (AttributeError, KeyError):
            pass
        
        # 시장 데이터가 없거나 타임스탬프를 가져올 수 없는 경우
        return datetime.now()
    
    def _get_execution_price(self, order: Order) -> Optional[float]:
        """실행 가격 결정"""
        # 시장가 주문은 현재 바의 시가로 실행 (다음 바가 시작된 상태)
        if order.order_type == OrderType.MARKET:
            return self._get_market_order_price(order.symbol)
        
        current_price = self._get_current_price(order.symbol)
        if current_price is None:
            return None
        
        # 지정가 주문은 지정가격으로 실행
        if order.order_type == OrderType.LIMIT:
            return order.price
        
        # 스톱 주문은 현재 가격으로 실행
        if order.order_type == OrderType.STOP:
            return current_price
        
        return current_price
    
    def _get_market_order_price(self, symbol: str) -> Optional[float]:
        """시장가 주문 체결 가격 조회 (현재 바의 시가)"""
        if self.current_data is None:
            return None
        
        latest_data = self.current_data.get_latest(symbol)
        # 시장가 주문은 시가로 체결 (실제 거래에서는 다음 바의 시가에 체결됨)
        return latest_data.open if latest_data else None
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """수수료 계산"""
        return quantity * price * self.commission_rate
    
    def _calculate_slippage(self, order: Order, price: float) -> float:
        """슬리피지 계산"""
        # 시장가 주문에만 슬리피지 적용
        if order.order_type == OrderType.MARKET:
            return self.slippage_rate
        return 0.0
    
    def _process_pending_orders(self) -> None:
        """대기 중인 주문들 처리 - 시장가 주문 특별 처리 포함"""
        for order in self.orders.values():
            if not order.is_active:
                continue
            
            # 시장가 주문은 새로운 바에서 체결
            if order.order_type == OrderType.MARKET:
                self._execute_market_order_on_new_bar(order)
            # 다른 주문 타입은 기존 로직대로 처리
            elif self._can_execute_immediately(order):
                self._execute_order(order)
    
    def _execute_market_order_on_new_bar(self, order: Order) -> None:
        """새로운 바에서 시장가 주문 체결"""
        if not order.is_active:
            return
        
        # 현재 바의 시가로 체결 가격 결정
        execution_price = self._get_market_order_price(order.symbol)
        if execution_price is None:
            order.reject("Cannot determine market order execution price")
            return
        
        # 추가 자금 검증 (매수 주문의 경우)
        if order.is_buy:
            estimated_cost = order.quantity * execution_price * (1 + self.commission_rate)
            if estimated_cost > self.portfolio.cash:
                order.reject("Insufficient cash for market order execution")
                return
        
        # 현재 시장 데이터의 타임스탬프 사용
        current_timestamp = self._get_current_timestamp()
        
        # 주문 체결
        order.fill(
            quantity=order.remaining_quantity,
            price=execution_price,
            timestamp=current_timestamp
        )
        
        # 거래 생성 - 시장 데이터 타임스탬프 전달
        trade = self._create_trade(
            order=order,
            quantity=order.filled_quantity,
            price=execution_price,
            timestamp=current_timestamp
        )
        
        # 포트폴리오 업데이트
        self._update_portfolio_from_trade(trade)

    def get_broker_info(self) -> dict:
        """브로커 정보 반환"""
        return {
            "broker_type": "SimpleBroker",
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "initial_cash": self.initial_cash,
            "current_cash": self.portfolio.cash,
            "total_orders": len(self.orders),
            "total_trades": len(self.trades)
        } 