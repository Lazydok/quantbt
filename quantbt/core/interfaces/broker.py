"""
브로커 인터페이스

주문 실행과 포트폴리오 관리를 위한 핵심 인터페이스들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional
from datetime import datetime

from ..entities.order import Order
from ..entities.trade import Trade
from ..entities.position import Portfolio
from ..entities.market_data import MarketDataBatch


class IBroker(Protocol):
    """브로커 인터페이스"""
    
    def submit_order(self, order: Order) -> str:
        """주문 제출
        
        Args:
            order: 제출할 주문
            
        Returns:
            주문 ID
        """
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소
        
        Args:
            order_id: 취소할 주문 ID
            
        Returns:
            취소 성공 여부
        """
        ...
    
    def get_portfolio(self) -> Portfolio:
        """포트폴리오 조회
        
        Returns:
            현재 포트폴리오 상태
        """
        ...
    
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """주문 목록 조회
        
        Args:
            status: 필터링할 주문 상태 (None이면 모든 주문)
            
        Returns:
            주문 목록
        """
        ...
    
    def get_trades(self) -> List[Trade]:
        """거래 내역 조회
        
        Returns:
            거래 내역 목록
        """
        ...
    
    def update_market_data(self, data: MarketDataBatch) -> None:
        """시장 데이터 업데이트
        
        Args:
            data: 새로운 시장 데이터
        """
        ...


class BrokerBase(ABC):
    """브로커 기본 클래스"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.portfolio = Portfolio(positions={}, cash=initial_cash)
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.current_data: Optional[MarketDataBatch] = None
        
    def submit_order(self, order: Order) -> str:
        """주문 제출"""
        # 주문 유효성 검증
        if not self._validate_order(order):
            order.reject("Order validation failed")
            return order.order_id
        
        # 주문 저장
        self.orders[order.order_id] = order
        
        # 즉시 실행 가능한 주문인지 확인
        if self._can_execute_immediately(order):
            self._execute_order(order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.is_active:
            order.cancel()
            return True
        
        return False
    
    def get_portfolio(self) -> Portfolio:
        """포트폴리오 조회"""
        return self.portfolio
    
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """주문 목록 조회"""
        if status is None:
            return list(self.orders.values())
        
        return [order for order in self.orders.values() if order.status.value == status]
    
    def get_trades(self) -> List[Trade]:
        """거래 내역 조회"""
        return self.trades.copy()
    
    def update_market_data(self, data: MarketDataBatch) -> None:
        """시장 데이터 업데이트"""
        self.current_data = data
        
        # 포트폴리오 시장 가격 업데이트
        price_dict = data.get_price_dict("close")
        self.portfolio.update_market_prices(price_dict)
        
        # 대기 중인 주문들 확인
        self._process_pending_orders()
    
    @abstractmethod
    def _validate_order(self, order: Order) -> bool:
        """주문 유효성 검증 - 서브클래스에서 구현"""
        pass
    
    @abstractmethod
    def _can_execute_immediately(self, order: Order) -> bool:
        """즉시 실행 가능 여부 확인 - 서브클래스에서 구현"""
        pass
    
    @abstractmethod
    def _execute_order(self, order: Order) -> None:
        """주문 실행 - 서브클래스에서 구현"""
        pass
    
    def _process_pending_orders(self) -> None:
        """대기 중인 주문들 처리"""
        for order in self.orders.values():
            if order.is_active and self._can_execute_immediately(order):
                self._execute_order(order)
    
    def _create_trade(
        self, 
        order: Order, 
        quantity: float, 
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Trade:
        """거래 생성"""
        trade = Trade(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp or datetime.now(),
            commission=self._calculate_commission(quantity, price),
            slippage=self._calculate_slippage(order, price)
        )
        
        self.trades.append(trade)
        return trade
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """수수료 계산 - 서브클래스에서 오버라이드 가능"""
        return 0.0  # 기본적으로 수수료 없음
    
    def _calculate_slippage(self, order: Order, price: float) -> float:
        """슬리피지 계산 - 서브클래스에서 오버라이드 가능"""
        return 0.0  # 기본적으로 슬리피지 없음
    
    def _update_portfolio_from_trade(self, trade: Trade) -> None:
        """거래로부터 포트폴리오 업데이트"""
        # 거래 전 포지션 정보 기록
        current_position = self.portfolio.get_position(trade.symbol)
        trade.position_size_before = current_position.quantity
        trade.avg_price_before = current_position.avg_price
        
        # 포지션 업데이트
        realized_pnl = self.portfolio.update_position(
            trade.symbol, 
            trade.signed_quantity, 
            trade.price
        )
        
        # 거래 후 포지션 정보 기록
        updated_position = self.portfolio.get_position(trade.symbol)
        trade.position_size_after = updated_position.quantity
        trade.avg_price_after = updated_position.avg_price
        trade.realized_pnl = realized_pnl
        
        # 현금 업데이트 (거래 비용 차감)
        if trade.side.value == 1:  # 매수
            self.portfolio.cash -= trade.total_cost
        else:  # 매도
            self.portfolio.cash += trade.notional_value - trade.commission
    
    def get_account_summary(self) -> Dict[str, Any]:
        """계좌 요약 정보"""
        return {
            "initial_cash": self.initial_cash,
            "current_cash": self.portfolio.cash,
            "total_equity": self.portfolio.equity,
            "total_pnl": self.portfolio.total_pnl,
            "total_realized_pnl": self.portfolio.total_realized_pnl,
            "total_unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "active_positions": len(self.portfolio.active_positions),
            "total_orders": len(self.orders),
            "total_trades": len(self.trades)
        } 