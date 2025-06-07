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
    
    # 포지션 기반 손익 정보 추가
    realized_pnl: float = 0.0  # 이 거래로 인한 실현 손익
    position_size_before: float = 0.0  # 거래 전 포지션 크기
    position_size_after: float = 0.0   # 거래 후 포지션 크기
    avg_price_before: float = 0.0      # 거래 전 평균단가
    avg_price_after: float = 0.0       # 거래 후 평균단가
    
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
    
    @property
    def is_position_opening(self) -> bool:
        """포지션 진입 거래인지 확인"""
        return (self.position_size_before == 0 and self.position_size_after != 0) or \
               (self.position_size_before * self.position_size_after < 0)  # 방향 전환
    
    @property
    def is_position_closing(self) -> bool:
        """포지션 청산 거래인지 확인"""
        return (self.position_size_before != 0 and self.position_size_after == 0) or \
               (abs(self.position_size_after) < abs(self.position_size_before))  # 부분 청산 포함
    
    @property
    def is_profitable(self) -> bool:
        """수익 거래인지 확인 (실현 손익 기준)"""
        return self.realized_pnl > 0
    
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
            "realized_pnl": self.realized_pnl,
            "position_size_before": self.position_size_before,
            "position_size_after": self.position_size_after,
            "avg_price_before": self.avg_price_before,
            "avg_price_after": self.avg_price_after,
            "notional_value": self.notional_value,
            "total_cost": self.total_cost,
            "effective_price": self.effective_price,
            "is_position_opening": self.is_position_opening,
            "is_position_closing": self.is_position_closing,
            "is_profitable": self.is_profitable,
            "metadata": self.metadata
        } 