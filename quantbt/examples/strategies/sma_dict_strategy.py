"""
Dict 기반 단순 이동평균 전략

고성능 Dict Native 방식으로 구현된 SMA 브레이크아웃 전략
"""

from typing import List, Dict, Any
from ...core.strategies.dict_based import DictTradingStrategy
from ...core.entities.order import Order, OrderSide, OrderType

class SimpleSMAStrategyDict(DictTradingStrategy):
    """Dict 기반 간단한 SMA 전략
    
    매수 조건: 가격이 SMA(buy_sma) 상회
    매도 조건: 가격이 SMA(sell_sma) 하회
    
    특징:
    - Zero Conversion: Dict 데이터 직접 활용
    - 극고속 처리: 벡터화된 지표 계산
    - 메모리 효율: 최소한의 메모리 사용
    """
    
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30):
        """SMA 전략 초기화
        
        Args:
            buy_sma: 매수 신호용 이동평균 기간
            sell_sma: 매도 신호용 이동평균 기간
        """
        super().__init__(
            name="SimpleSMAStrategyDict",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma
            },
            position_size_pct=0.8,
            max_positions=1
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
    
    def _compute_indicators_for_symbol_dict(self, symbol_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """심볼별 이동평균 지표 계산
        
        Args:
            symbol_data: 시간순 정렬된 심볼 데이터
            
        Returns:
            이동평균이 추가된 데이터
        """
        # 가격 데이터 추출
        prices = [row['close'] for row in symbol_data]
        
        # 이동평균 계산
        buy_sma_values = self.calculate_sma_dict(prices, self.buy_sma)
        sell_sma_values = self.calculate_sma_dict(prices, self.sell_sma)
        
        # 기존 데이터에 지표 추가
        enriched_data = []
        for i, row in enumerate(symbol_data):
            enriched_row = row.copy()
            enriched_row[f'sma_{self.buy_sma}'] = buy_sma_values[i]
            enriched_row[f'sma_{self.sell_sma}'] = sell_sma_values[i]
            enriched_data.append(enriched_row)
        
        return enriched_data
    
    def generate_signals_dict(self, current_data: Dict[str, Any], 
                            historical_data: List[Dict[str, Any]] = None) -> List[Order]:
        """Dict 기반 신호 생성
        
        Args:
            current_data: 현재 캔들 데이터
            historical_data: 과거 데이터 (미사용)
            
        Returns:
            생성된 주문 리스트
        """
        orders = []
        
        if not self.broker:
            return orders
        
        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')
        
        # 지표가 계산되지 않은 경우 건너뛰기
        if buy_sma is None or sell_sma is None:
            return orders
        
        current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 SMA5 상회 + 포지션 없음
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
        
        # 매도 신호: 가격이 SMA10 하회 + 포지션 있음
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_positions[symbol],
                order_type=OrderType.MARKET
            )
            orders.append(order)
        
        return orders 