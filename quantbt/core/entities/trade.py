"""
거래 엔티티

체결된 거래 정보를 표현하는 엔티티들을 정의합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from .order import OrderSide


@dataclass
class Trade:
    """체결된 거래 엔티티"""
    
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def notional_value(self) -> float:
        """명목 가치"""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """총 비용 (수수료 포함)"""
        return self.notional_value + self.commission
    
    @property
    def effective_price(self) -> float:
        """실효 가격 (슬리피지 적용)"""
        if self.side == OrderSide.BUY:
            return self.price * (1 + self.slippage)
        else:
            return self.price * (1 - self.slippage)
    
    @property
    def signed_quantity(self) -> float:
        """부호가 있는 수량 (매수: +, 매도: -)"""
        return self.quantity if self.side == OrderSide.BUY else -self.quantity
    
    @property
    def pnl_impact(self) -> float:
        """손익에 미치는 영향 (수수료 제외)"""
        return self.signed_quantity * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "commission": self.commission,
            "slippage": self.slippage,
            "notional_value": self.notional_value,
            "total_cost": self.total_cost,
            "effective_price": self.effective_price,
            "metadata": self.metadata
        } 