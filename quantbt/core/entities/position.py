"""
포지션 엔티티

포지션과 관련된 핵심 엔티티들을 정의합니다.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Position:
    """포지션 엔티티"""
    
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """시장 가치"""
        return abs(self.quantity) * self.market_price
    
    @property
    def notional_value(self) -> float:
        """명목 가치 (포지션 방향 고려)"""
        return self.quantity * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익"""
        if self.quantity == 0:
            return 0.0
        return (self.market_price - self.avg_price) * self.quantity
    
    @property
    def total_pnl(self) -> float:
        """총 손익 (실현 + 미실현)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_long(self) -> bool:
        """롱 포지션 여부"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """숏 포지션 여부"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """포지션 없음 여부"""
        return self.quantity == 0
    
    @property
    def pnl_percentage(self) -> float:
        """수익률 (%)"""
        if self.avg_price == 0:
            return 0.0
        return (self.market_price - self.avg_price) / self.avg_price * 100
    
    def update_position(self, trade_quantity: float, trade_price: float) -> float:
        """포지션 업데이트
        
        Args:
            trade_quantity: 거래 수량 (양수: 매수, 음수: 매도)
            trade_price: 거래 가격
            
        Returns:
            실현 손익
        """
        realized_pnl = 0.0
        
        if self.quantity == 0:
            # 새로운 포지션 생성
            self.quantity = trade_quantity
            self.avg_price = trade_price
        elif (self.quantity > 0 and trade_quantity > 0) or (self.quantity < 0 and trade_quantity < 0):
            # 포지션 증가 (같은 방향)
            total_cost = abs(self.quantity) * self.avg_price + abs(trade_quantity) * trade_price
            self.quantity += trade_quantity
            if self.quantity != 0:
                self.avg_price = total_cost / abs(self.quantity)
        else:
            # 포지션 감소 또는 방향 전환
            if abs(trade_quantity) <= abs(self.quantity):
                # 포지션 감소
                realized_pnl = (trade_price - self.avg_price) * abs(trade_quantity) * (1 if self.quantity > 0 else -1)
                self.quantity += trade_quantity
                if abs(self.quantity) < 1e-8:  # 포지션 종료
                    self.quantity = 0.0
                    self.avg_price = 0.0
            else:
                # 포지션 방향 전환
                # 기존 포지션 전체 청산
                realized_pnl = (trade_price - self.avg_price) * abs(self.quantity) * (1 if self.quantity > 0 else -1)
                remaining_quantity = trade_quantity + self.quantity
                
                # 새로운 방향 포지션 생성
                self.quantity = remaining_quantity
                self.avg_price = trade_price
        
        self.realized_pnl += realized_pnl
        return realized_pnl
    
    def update_market_price(self, price: float) -> None:
        """시장 가격 업데이트"""
        self.market_price = price
    
    def close_position(self, close_price: Optional[float] = None) -> float:
        """포지션 강제 청산
        
        Args:
            close_price: 청산 가격 (None인 경우 시장 가격 사용)
            
        Returns:
            실현 손익
        """
        if self.quantity == 0:
            return 0.0
        
        price = close_price if close_price is not None else self.market_price
        realized_pnl = self.update_position(-self.quantity, price)
        
        return realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "notional_value": self.notional_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "pnl_percentage": self.pnl_percentage,
            "is_long": self.is_long,
            "is_short": self.is_short,
            "is_flat": self.is_flat
        }


@dataclass
class Portfolio:
    """포트폴리오 - 여러 포지션들을 관리"""
    
    positions: Dict[str, Position]
    cash: float = 0.0
    
    def __post_init__(self) -> None:
        if not hasattr(self, 'positions'):
            self.positions = {}
    
    def get_position(self, symbol: str) -> Position:
        """포지션 조회 (없으면 새로 생성)"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_position(self, symbol: str, quantity: float, price: float) -> float:
        """포지션 업데이트"""
        position = self.get_position(symbol)
        return position.update_position(quantity, price)
    
    def update_market_prices(self, price_dict: Dict[str, float]) -> None:
        """모든 포지션의 시장 가격 업데이트"""
        for symbol, price in price_dict.items():
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
    
    @property
    def total_market_value(self) -> float:
        """총 포지션 시장 가치"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_notional_value(self) -> float:
        """총 명목 가치"""
        return sum(pos.notional_value for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        """총 미실현 손익"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """총 실현 손익"""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def equity(self) -> float:
        """총 자본 (현금 + 포지션 시장가치)"""
        return self.cash + self.total_market_value
    
    @property
    def active_positions(self) -> Dict[str, Position]:
        """활성 포지션들 (수량이 0이 아닌 포지션)"""
        return {symbol: pos for symbol, pos in self.positions.items() if not pos.is_flat}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "cash": self.cash,
            "total_market_value": self.total_market_value,
            "total_notional_value": self.total_notional_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "total_pnl": self.total_pnl,
            "equity": self.equity,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.active_positions.items()}
        } 