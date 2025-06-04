"""
주문 엔티티

주문과 관련된 핵심 엔티티들을 정의합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import uuid4


class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """주문 방향"""
    BUY = 1
    SELL = -1


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    """주문 유효기간"""
    GTC = "gtc"  # Good Till Cancelled
    DAY = "day"  # Day order
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class Order:
    """주문 엔티티"""
    
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    order_id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    reduce_only: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """주문 유효성 검증"""
        if self.quantity <= 0:
            raise ValueError("수량은 0보다 커야 합니다")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"{self.order_type.value} 주문은 가격이 필요합니다")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} 주문은 스톱 가격이 필요합니다")
    
    @property
    def is_buy(self) -> bool:
        """매수 주문 여부"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """매도 주문 여부"""
        return self.side == OrderSide.SELL
    
    @property
    def is_market_order(self) -> bool:
        """시장가 주문 여부"""
        return self.order_type == OrderType.MARKET
    
    @property
    def is_limit_order(self) -> bool:
        """지정가 주문 여부"""
        return self.order_type == OrderType.LIMIT
    
    @property
    def is_stop_order(self) -> bool:
        """스톱 주문 여부"""
        return self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
    
    @property
    def remaining_quantity(self) -> float:
        """미체결 수량"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """완전 체결 여부"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """활성 상태 여부"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    def fill(
        self, 
        quantity: float, 
        price: float, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """주문 체결 처리"""
        if not self.is_active:
            raise ValueError("비활성 주문은 체결할 수 없습니다")
        
        if quantity > self.remaining_quantity:
            raise ValueError("체결 수량이 미체결 수량을 초과할 수 없습니다")
        
        self.filled_quantity += quantity
        self.filled_price = price
        self.filled_at = timestamp or datetime.now()
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self) -> None:
        """주문 취소"""
        if not self.is_active:
            raise ValueError("비활성 주문은 취소할 수 없습니다")
        
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str = "") -> None:
        """주문 거부"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("대기 중인 주문만 거부할 수 있습니다")
        
        self.status = OrderStatus.REJECTED
        if not self.metadata:
            self.metadata = {}
        self.metadata["rejection_reason"] = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "reduce_only": self.reduce_only,
            "metadata": self.metadata
        } 