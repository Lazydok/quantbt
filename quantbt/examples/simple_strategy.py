"""
간단한 테스트 전략

백테스팅 엔진 테스트를 위한 기본적인 전략 예제입니다.
지표는 사전에 계산되고, 백테스팅 시에는 단순 비교로 신호를 생성합니다.
"""

from typing import List, Dict, Any, Optional
import polars as pl

from ..core.interfaces.strategy import TradingStrategy, BacktestContext
from ..core.entities.market_data import MarketDataBatch
from ..core.entities.order import Order, OrderType, OrderSide
from ..core.entities.trade import Trade


class BuyAndHoldStrategy(TradingStrategy):
    """바이 앤 홀드 전략 - 한 번 매수 후 보유"""
    
    def __init__(self):
        super().__init__(
            name="BuyAndHoldStrategy",
            config={},
            position_size_pct=1.0,  # 전체 자본으로 매수
            max_positions=10
        )
        self.bought_symbols: set[str] = set()  # 이미 매수한 심볼들 추적
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """바이 앤 홀드는 지표 불필요 - 원본 데이터 그대로 반환"""
        return symbol_data
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 한 번만 매수"""
        orders = []
        
        if not self.context:
            return orders
        
        # 아직 매수하지 않은 심볼 중 하나만 선택하여 매수
        for symbol in data.symbols:
            if symbol not in self.bought_symbols:
                current_price = self.get_current_price(symbol, data)
                if current_price and current_price > 0:
                    # 현재 포트폴리오 가치의 일정 비율로 매수
                    current_portfolio_value = self.get_portfolio_value()
                    position_value = current_portfolio_value / len(self.context.symbols) * 0.8  # 80%만 사용
                    quantity = position_value / current_price
                    
                    # 최소 수량 확인
                    if quantity > 0.01:
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        orders.append(order)
                        self.bought_symbols.add(symbol)
                        break  # 한 번에 하나씩만 매수
        
        return orders


class SimpleMovingAverageCrossStrategy(TradingStrategy):
    """단순 이동평균 교차 전략 - 지표 사전 계산 버전"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__(
            name="SimpleMovingAverageCrossStrategy",
            config={
                "short_window": short_window,
                "long_window": long_window
            },
            position_size_pct=0.2,  # 20%씩 포지션
            max_positions=5
        )
        self.short_window = short_window
        self.long_window = long_window
        self.indicator_columns = [f"sma_{short_window}", f"sma_{long_window}"]
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """심볼별 이동평균 지표 계산"""
        # 시간순 정렬 확인
        data = symbol_data.sort("timestamp")
        
        # 단순 이동평균 계산
        short_sma = self.calculate_sma(data["close"], self.short_window)
        long_sma = self.calculate_sma(data["close"], self.long_window)
        
        # 지표 컬럼 추가
        return data.with_columns([
            short_sma.alias(f"sma_{self.short_window}"),
            long_sma.alias(f"sma_{self.long_window}")
        ])
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 단순 이동평균 교차 확인"""
        orders = []
        
        if not self.context:
            return orders
        
        for symbol in data.symbols:
            current_price = self.get_current_price(symbol, data)
            if not current_price:
                continue
            
            # 현재 지표 값 조회
            short_ma = self.get_indicator_value(symbol, f"sma_{self.short_window}", data)
            long_ma = self.get_indicator_value(symbol, f"sma_{self.long_window}", data)
            
            if short_ma is None or long_ma is None:
                continue
            
            # 이전 지표 값 조회 (골든/데드 크로스 확인을 위해)
            symbol_data = data.get_symbol_data(symbol)
            if symbol_data.height < 2:
                continue
            
            prev_row = symbol_data.row(-2, named=True)
            prev_short_ma = prev_row.get(f"sma_{self.short_window}")
            prev_long_ma = prev_row.get(f"sma_{self.long_window}")
            
            if prev_short_ma is None or prev_long_ma is None:
                continue
            
            # 골든 크로스 (매수 신호)
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
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
            
            # 데드 크로스 (매도 신호)
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                # 현재 포지션이 있다면 매도
                current_positions = self.get_current_positions()
                if symbol in current_positions and current_positions[symbol] > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=current_positions[symbol],
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
        
        return orders


class RSIStrategy(TradingStrategy):
    """RSI 전략 - 지표 사전 계산 버전"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(
            name="RSIStrategy",
            config={
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought
            },
            position_size_pct=0.15,
            max_positions=5
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.indicator_columns = ["rsi"]
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """심볼별 RSI 지표 계산"""
        # 시간순 정렬 확인
        data = symbol_data.sort("timestamp")
        
        # RSI 계산
        rsi = self.calculate_rsi(data["close"], self.rsi_period)
        
        # RSI 컬럼 추가
        return data.with_columns([
            rsi.alias("rsi")
        ])
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - RSI 기반 매수/매도"""
        orders = []
        
        if not self.context:
            return orders
        
        for symbol in data.symbols:
            current_price = self.get_current_price(symbol, data)
            if not current_price:
                continue
            
            # 현재 RSI 값 조회
            rsi = self.get_indicator_value(symbol, "rsi", data)
            if rsi is None:
                continue
            
            current_positions = self.get_current_positions()
            
            # 과매도 구간 - 매수
            if rsi < self.oversold and symbol not in current_positions:
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
            
            # 과매수 구간 - 매도
            elif rsi > self.overbought and symbol in current_positions and current_positions[symbol] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                )
                orders.append(order)
        
        return orders


class RandomStrategy(TradingStrategy):
    """랜덤 전략 - 테스트 목적"""
    
    def __init__(self, trade_probability: float = 0.1):
        super().__init__(
            name="RandomStrategy",
            config={"trade_probability": trade_probability},
            position_size_pct=0.1,
            max_positions=3
        )
        self.trade_probability = trade_probability
        self.trade_count = 0
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """랜덤 전략은 지표 불필요 - 원본 데이터 그대로 반환"""
        return symbol_data
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 랜덤"""
        import random
        
        orders = []
        
        if not self.context or random.random() > self.trade_probability:
            return orders
        
        # 랜덤하게 심볼 선택
        if data.symbols:
            symbol = random.choice(data.symbols)
            current_price = self.get_current_price(symbol, data)
            
            if current_price:
                # 랜덤하게 매수/매도 결정
                current_positions = self.get_current_positions()
                
                if symbol in current_positions and current_positions[symbol] > 0:
                    # 포지션이 있으면 50% 확률로 매도
                    if random.random() > 0.5:
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_positions[symbol],
                            order_type=OrderType.MARKET
                        )
                        orders.append(order)
                else:
                    # 포지션이 없으면 매수
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
        
        return orders 