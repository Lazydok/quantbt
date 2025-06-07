"""
SMA 그리드 서치 전략

매수/매도 SMA가 분리된 그리드 서치용 전략입니다.
"""

from typing import List
import polars as pl
import pandas as pd

from ...core.interfaces.strategy import TradingStrategy
from ...core.entities.order import Order, OrderType, OrderSide
from ...core.entities.market_data import MarketDataBatch


class SMAGridStrategy(TradingStrategy):
    """매수/매도 SMA가 분리된 그리드 서치용 전략"""
    
    def __init__(self, buy_sma: int, sell_sma: int, position_size_pct: float = 0.95, max_positions: int = 1):
        """
        Args:
            buy_sma: 매수 기준 SMA 기간
            sell_sma: 매도 기준 SMA 기간  
            position_size_pct: 포지션 크기 비율
            max_positions: 최대 포지션 수
        """
        super().__init__(
            name=f"SMAGridStrategy_B{buy_sma}_S{sell_sma}",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma,
                "position_size_pct": position_size_pct
            },
            position_size_pct=position_size_pct,
            max_positions=max_positions
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.current_positions = {}  # {symbol: quantity}
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """SMA 지표 계산"""
        # 매수/매도 SMA 계산
        sma_data = symbol_data.with_columns([
            pl.col("close").rolling_mean(window_size=self.buy_sma).alias("buy_sma"),
            pl.col("close").rolling_mean(window_size=self.sell_sma).alias("sell_sma")
        ])
        
        # 매매 신호 생성
        sma_data = sma_data.with_columns([
            # 매수 신호: 현재가 > 매수 SMA
            pl.when(pl.col("close") > pl.col("buy_sma"))
            .then(1)
            .otherwise(0)
            .alias("buy_signal"),
            
            # 매도 신호: 현재가 < 매도 SMA  
            pl.when(pl.col("close") < pl.col("sell_sma"))
            .then(1)
            .otherwise(0)
            .alias("sell_signal")
        ])
        
        return sma_data
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """SMA 기반 매매 신호 생성"""
        orders = []
        
        if not self.context:
            return orders
        
        for symbol in data.symbols:
            symbol_data = data.get_symbol_data(symbol)
            if symbol_data is None or len(symbol_data) == 0:
                continue
                
            # 최신 데이터만 확인
            latest_data = symbol_data.tail(1)
            if len(latest_data) == 0:
                continue
                
            latest_row = latest_data.row(0, named=True)
            current_price = latest_row.get("close")
            buy_sma_value = latest_row.get("buy_sma")
            sell_sma_value = latest_row.get("sell_sma")
            buy_signal = latest_row.get("buy_signal", 0)
            sell_signal = latest_row.get("sell_signal", 0)
            
            # SMA가 계산되지 않은 경우 (초기 데이터) 건너뛰기
            if (buy_sma_value is None or pd.isna(buy_sma_value) or 
                sell_sma_value is None or pd.isna(sell_sma_value)):
                continue
                
            current_position = self.current_positions.get(symbol, 0)
            
            # 매수 신호: 현재가 > 매수 SMA이고 포지션이 없는 경우
            if buy_signal == 1 and current_position == 0:
                portfolio_value = self.get_portfolio_value()
                position_value = portfolio_value * self.position_size_pct
                quantity = position_value / current_price
                
                if quantity > 0.01:  # 최소 수량 확인
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
                    self.current_positions[symbol] = quantity
            
            # 매도 신호: 현재가 < 매도 SMA이고 포지션이 있는 경우
            elif sell_signal == 1 and current_position > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_position,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                self.current_positions[symbol] = 0
        
        return orders
    
    def on_order_filled(self, order: Order, fill_price: float, fill_quantity: float) -> None:
        """주문 체결 시 포지션 업데이트"""
        if order.side == OrderSide.BUY:
            self.current_positions[order.symbol] = fill_quantity
        elif order.side == OrderSide.SELL:
            self.current_positions[order.symbol] = 0 