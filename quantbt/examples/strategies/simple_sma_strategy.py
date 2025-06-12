"""
간단한 SMA 전략

TradingStrategy를 상속한 단순 이동평균 전략
"""

from typing import List, Dict, Any, Optional
from ...core.interfaces.strategy import TradingStrategy
from ...core.entities.order import Order, OrderSide, OrderType

class SimpleSMAStrategy(TradingStrategy):
    """SMA 전략
    
    하이브리드 방식:
    - 지표 계산: Polars 벡터연산
    - 신호 생성: Dict Native 방식
    
    매수: 가격이 SMA15 상회
    매도: 가격이 SMA30 하회  
    """
    
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30):
        super().__init__(
            name="SimpleSMAStrategy",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma
            },
            position_size_pct=0.8,  # 80%씩 포지션
            max_positions=1
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        
    def _compute_indicators_for_symbol(self, symbol_data):
        """심볼별 이동평균 지표 계산 (Polars 벡터 연산)"""
        
        # 시간순 정렬 확인
        data = symbol_data.sort("timestamp")
        
        # 단순 이동평균 계산
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        
        # 지표 컬럼 추가
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}")
        ])
    
    def _generate_single_timeframe_signals(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성"""
        orders = []
        
        if not self._broker:
            return orders
        
        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')
        
        # 지표가 계산되지 않은 경우 건너뛰기
        if buy_sma is None or sell_sma is None:
            return orders
        
        current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 SMA15 상회 + 포지션 없음
        if current_price > buy_sma and symbol not in current_positions:
            portfolio_value = self.get_portfolio_value()
            quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
            
            if quantity > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                # print(f"📈 매수 신호: {symbol} @ {current_price:,.0f}원 (SMA{self.buy_sma}: {buy_sma:,.0f})")
        
        # 매도 신호: 가격이 SMA30 하회 + 포지션 있음
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_positions[symbol],
                order_type=OrderType.MARKET
            )
            orders.append(order)
            # print(f"📉 매도 신호: {symbol} @ {current_price:,.0f}원 (SMA{self.sell_sma}: {sell_sma:,.0f})")
        
        return orders