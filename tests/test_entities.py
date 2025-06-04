"""
핵심 엔티티 테스트

기본 엔티티들의 동작을 검증합니다.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
from typing import Optional

from quantbt.core.entities.market_data import MarketData, MarketDataBatch
from quantbt.core.entities.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from quantbt.core.entities.position import Position, Portfolio
from quantbt.core.entities.trade import Trade


class TestMarketData:
    """MarketData 엔티티 테스트"""
    
    def test_market_data_creation(self):
        """MarketData 생성 테스트"""
        timestamp = datetime(2023, 1, 1, 9, 0)
        data = MarketData(
            timestamp=timestamp,
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        
        assert data.timestamp == timestamp
        assert data.symbol == "AAPL"
        assert data.ohlc == (150.0, 155.0, 149.0, 154.0)
        assert data.typical_price == (155.0 + 149.0 + 154.0) / 3
    
    def test_market_data_to_dict(self):
        """MarketData to_dict 테스트"""
        data = MarketData(
            timestamp=datetime(2023, 1, 1),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        
        result = data.to_dict()
        assert result["symbol"] == "AAPL"
        assert result["close"] == 154.0
        assert "metadata" in result


class TestMarketDataBatch:
    """MarketDataBatch 엔티티 테스트"""
    
    def create_sample_data(self) -> pl.DataFrame:
        """샘플 데이터 생성"""
        return pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "symbol": ["AAPL", "AAPL"],
            "open": [150.0, 155.0],
            "high": [155.0, 160.0],
            "low": [149.0, 154.0],
            "close": [154.0, 158.0],
            "volume": [1000000, 1100000]
        })
    
    def test_market_data_batch_creation(self):
        """MarketDataBatch 생성 테스트"""
        df = self.create_sample_data()
        batch = MarketDataBatch(
            data=df,
            symbols=["AAPL"],
            timeframe="1D"
        )
        
        assert len(batch.symbols) == 1
        assert "AAPL" in batch.symbols
        assert batch.timeframe == "1D"
        assert batch.data.height == 2
    
    def test_get_latest(self):
        """최신 데이터 조회 테스트"""
        df = self.create_sample_data()
        batch = MarketDataBatch(data=df, symbols=["AAPL"], timeframe="1D")
        
        latest = batch.get_latest("AAPL")
        assert latest is not None
        assert latest.close == 158.0
        assert latest.symbol == "AAPL"
    
    def test_get_price_dict(self):
        """가격 딕셔너리 조회 테스트"""
        df = self.create_sample_data()
        batch = MarketDataBatch(data=df, symbols=["AAPL"], timeframe="1D")
        
        prices = batch.get_price_dict("close")
        assert "AAPL" in prices
        assert prices["AAPL"] == 158.0
    
    def test_invalid_data_raises_error(self):
        """잘못된 데이터로 생성 시 에러 테스트"""
        invalid_df = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1)],
            "symbol": ["AAPL"],
            "open": [150.0]
            # close, high, low, volume 누락
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            MarketDataBatch(data=invalid_df, symbols=["AAPL"], timeframe="1D")


class TestOrder:
    """Order 엔티티 테스트"""
    
    def test_market_order_creation(self):
        """시장가 주문 생성 테스트"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "AAPL"
        assert order.is_buy
        assert order.is_market_order
        assert order.status == OrderStatus.PENDING
        assert order.remaining_quantity == 100.0
    
    def test_limit_order_creation(self):
        """지정가 주문 생성 테스트"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50.0,
            order_type=OrderType.LIMIT,
            price=155.0
        )
        
        assert order.is_sell
        assert order.is_limit_order
        assert order.price == 155.0
    
    def test_order_validation_invalid_quantity(self):
        """잘못된 수량으로 주문 생성 시 에러 테스트"""
        with pytest.raises(ValueError, match="수량은 0보다 커야 합니다"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=-10.0,
                order_type=OrderType.MARKET
            )
    
    def test_limit_order_without_price(self):
        """가격 없는 지정가 주문 생성 시 에러 테스트"""
        with pytest.raises(ValueError, match="주문은 가격이 필요합니다"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100.0,
                order_type=OrderType.LIMIT
            )
    
    def test_order_fill(self):
        """주문 체결 테스트"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        # 부분 체결
        order.fill(50.0, 150.0)
        assert order.filled_quantity == 50.0
        assert order.remaining_quantity == 50.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # 완전 체결
        order.fill(50.0, 151.0)
        assert order.filled_quantity == 100.0
        assert order.remaining_quantity == 0.0
        assert order.status == OrderStatus.FILLED
    
    def test_order_cancel(self):
        """주문 취소 테스트"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        
        # 취소된 주문은 다시 취소할 수 없음
        with pytest.raises(ValueError, match="비활성 주문은 취소할 수 없습니다"):
            order.cancel()


class TestPosition:
    """Position 엔티티 테스트"""
    
    def test_position_creation(self):
        """포지션 생성 테스트"""
        position = Position(symbol="AAPL")
        
        assert position.symbol == "AAPL"
        assert position.is_flat
        assert position.quantity == 0.0
        assert position.unrealized_pnl == 0.0
    
    def test_position_long_trade(self):
        """롱 포지션 거래 테스트"""
        position = Position(symbol="AAPL")
        
        # 매수 (롱 포지션 생성)
        realized_pnl = position.update_position(100.0, 150.0)
        assert realized_pnl == 0.0  # 신규 포지션은 실현손익 없음
        assert position.quantity == 100.0
        assert position.avg_price == 150.0
        assert position.is_long
        
        # 시장가격 업데이트
        position.update_market_price(155.0)
        assert position.unrealized_pnl == 500.0  # (155-150) * 100
        assert position.pnl_percentage == pytest.approx(3.33, rel=1e-2)
    
    def test_position_short_trade(self):
        """숏 포지션 거래 테스트"""
        position = Position(symbol="AAPL")
        
        # 매도 (숏 포지션 생성)
        position.update_position(-100.0, 150.0)
        assert position.quantity == -100.0
        assert position.is_short
        
        # 시장가격 하락 시 이익
        position.update_market_price(145.0)
        assert position.unrealized_pnl == 500.0  # (150-145) * 100
    
    def test_position_increase(self):
        """포지션 증가 테스트"""
        position = Position(symbol="AAPL")
        
        # 첫 번째 매수
        position.update_position(100.0, 150.0)
        
        # 두 번째 매수 (포지션 증가)
        position.update_position(50.0, 160.0)
        
        assert position.quantity == 150.0
        # 평균가격 = (100*150 + 50*160) / 150 = 153.33
        assert position.avg_price == pytest.approx(153.33, rel=1e-2)
    
    def test_position_partial_close(self):
        """포지션 부분 청산 테스트"""
        position = Position(symbol="AAPL")
        
        # 포지션 생성
        position.update_position(100.0, 150.0)
        
        # 부분 청산
        realized_pnl = position.update_position(-30.0, 160.0)
        
        assert position.quantity == 70.0
        assert realized_pnl == 300.0  # (160-150) * 30
        assert position.avg_price == 150.0  # 평균가격 유지
    
    def test_position_full_close(self):
        """포지션 완전 청산 테스트"""
        position = Position(symbol="AAPL")
        
        # 포지션 생성
        position.update_position(100.0, 150.0)
        
        # 완전 청산
        realized_pnl = position.update_position(-100.0, 160.0)
        
        assert position.is_flat
        assert realized_pnl == 1000.0  # (160-150) * 100
        assert position.avg_price == 0.0


class TestPortfolio:
    """Portfolio 엔티티 테스트"""
    
    def test_portfolio_creation(self):
        """포트폴리오 생성 테스트"""
        portfolio = Portfolio(positions={}, cash=100000.0)
        
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0
        assert portfolio.equity == 100000.0
    
    def test_portfolio_update_position(self):
        """포트폴리오 포지션 업데이트 테스트"""
        portfolio = Portfolio(positions={}, cash=100000.0)
        
        # 새 포지션 생성
        realized_pnl = portfolio.update_position("AAPL", 100.0, 150.0)
        
        assert realized_pnl == 0.0
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100.0
    
    def test_portfolio_market_price_update(self):
        """포트폴리오 시장가격 업데이트 테스트"""
        portfolio = Portfolio(positions={}, cash=100000.0)
        
        # 포지션 생성
        portfolio.update_position("AAPL", 100.0, 150.0)
        portfolio.update_position("MSFT", -50.0, 300.0)
        
        # 시장가격 업데이트
        price_dict = {"AAPL": 155.0, "MSFT": 295.0}
        portfolio.update_market_prices(price_dict)
        
        assert portfolio.positions["AAPL"].market_price == 155.0
        assert portfolio.positions["MSFT"].market_price == 295.0
        
        # 총 미실현 손익 = AAPL: (155-150)*100 + MSFT: (300-295)*50 = 500 + 250 = 750
        assert portfolio.total_unrealized_pnl == 750.0


class TestTrade:
    """Trade 엔티티 테스트"""
    
    def test_trade_creation(self):
        """거래 생성 테스트"""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=150.0,
            commission=1.0,
            slippage=0.001
        )
        
        assert trade.symbol == "AAPL"
        assert trade.notional_value == 15000.0
        assert trade.total_cost == 15001.0  # 수수료 포함
        assert trade.signed_quantity == 100.0
        assert trade.effective_price == pytest.approx(150.15, rel=1e-6)  # 슬리피지 적용
    
    def test_trade_sell_side(self):
        """매도 거래 테스트"""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            price=150.0,
            slippage=0.001
        )
        
        assert trade.signed_quantity == -100.0
        assert trade.effective_price == pytest.approx(149.85, rel=1e-6)  # 매도 시 슬리피지


if __name__ == "__main__":
    pytest.main([__file__]) 