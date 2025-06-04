"""
브로커 테스트

브로커의 주문 실행과 포트폴리오 관리 기능을 검증합니다.
"""

import pytest
import polars as pl
from datetime import datetime

from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.core.entities.order import Order, OrderType, OrderSide, OrderStatus
from quantbt.core.entities.market_data import MarketDataBatch


class TestSimpleBroker:
    """SimpleBroker 테스트"""
    
    @pytest.fixture
    def broker(self):
        """브로커 인스턴스 생성"""
        return SimpleBroker(initial_cash=100000.0, commission_rate=0.001, slippage_rate=0.0001)
    
    @pytest.fixture
    def sample_market_data(self):
        """샘플 시장 데이터 생성"""
        data = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "symbol": ["AAPL", "MSFT"],
            "open": [150.0, 300.0],
            "high": [155.0, 305.0],
            "low": [149.0, 299.0],
            "close": [154.0, 304.0],
            "volume": [1000000, 800000]
        })
        
        return MarketDataBatch(
            data=data,
            symbols=["AAPL", "MSFT"],
            timeframe="1D"
        )
    
    def test_broker_creation(self, broker):
        """브로커 생성 테스트"""
        assert broker.initial_cash == 100000.0
        assert broker.portfolio.cash == 100000.0
        assert broker.commission_rate == 0.001
        assert broker.slippage_rate == 0.0001
        assert len(broker.orders) == 0
        assert len(broker.trades) == 0
    
    def test_market_data_update(self, broker, sample_market_data):
        """시장 데이터 업데이트 테스트"""
        broker.update_market_data(sample_market_data)
        
        assert broker.current_data is not None
        assert broker.current_data.symbols == ["AAPL", "MSFT"]
    
    def test_submit_market_buy_order(self, broker, sample_market_data):
        """시장가 매수 주문 제출 테스트"""
        broker.update_market_data(sample_market_data)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        order_id = broker.submit_order(order)
        
        assert order_id == order.order_id
        assert order.status == OrderStatus.FILLED
        assert len(broker.trades) == 1
        
        # 포트폴리오 확인
        portfolio = broker.get_portfolio()
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100.0
        
        # 현금 감소 확인
        trade = broker.trades[0]
        expected_cash = 100000.0 - trade.total_cost
        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
    
    def test_submit_market_sell_order(self, broker, sample_market_data):
        """시장가 매도 주문 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 먼저 매수
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        broker.submit_order(buy_order)
        
        # 매도
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50.0,
            order_type=OrderType.MARKET
        )
        broker.submit_order(sell_order)
        
        assert len(broker.trades) == 2
        
        # 포지션 확인
        portfolio = broker.get_portfolio()
        assert portfolio.positions["AAPL"].quantity == 50.0
    
    def test_submit_limit_order(self, broker, sample_market_data):
        """지정가 주문 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 현재 가격(154.0)보다 낮은 지정가 매수 주문 (실행되지 않음)
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.LIMIT,
            price=150.0
        )
        
        broker.submit_order(order)
        
        assert order.status == OrderStatus.PENDING
        assert len(broker.trades) == 0
        
        # 가격이 하락하면 실행되어야 함
        updated_data = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 2)],
            "symbol": ["AAPL"],
            "open": [149.0],
            "high": [151.0],
            "low": [148.0],
            "close": [149.0],  # 지정가보다 낮음
            "volume": [1100000]
        })
        
        updated_batch = MarketDataBatch(
            data=updated_data,
            symbols=["AAPL"],
            timeframe="1D"
        )
        
        broker.update_market_data(updated_batch)
        
        assert order.status == OrderStatus.FILLED
        assert len(broker.trades) == 1
    
    def test_insufficient_cash_order_rejection(self, broker, sample_market_data):
        """현금 부족 시 주문 거부 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 현금보다 큰 주문
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000.0,  # 154,000 + 수수료 > 100,000
            order_type=OrderType.MARKET
        )
        
        broker.submit_order(order)
        
        assert order.status == OrderStatus.REJECTED
        assert len(broker.trades) == 0
    
    def test_insufficient_position_order_rejection(self, broker, sample_market_data):
        """포지션 부족 시 주문 거부 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 포지션 없이 매도 주문
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        broker.submit_order(order)
        
        assert order.status == OrderStatus.REJECTED
        assert len(broker.trades) == 0
    
    def test_cancel_order(self, broker, sample_market_data):
        """주문 취소 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 실행되지 않을 지정가 주문
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.LIMIT,
            price=100.0  # 현재 가격보다 훨씬 낮음
        )
        
        order_id = broker.submit_order(order)
        assert order.status == OrderStatus.PENDING
        
        # 주문 취소
        success = broker.cancel_order(order_id)
        assert success
        assert order.status == OrderStatus.CANCELLED
    
    def test_get_orders_by_status(self, broker, sample_market_data):
        """상태별 주문 조회 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 여러 주문 제출
        market_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        broker.submit_order(market_order)
        
        limit_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.LIMIT,
            price=100.0
        )
        broker.submit_order(limit_order)
        
        # 상태별 조회
        filled_orders = broker.get_orders("filled")
        pending_orders = broker.get_orders("pending")
        all_orders = broker.get_orders()
        
        assert len(filled_orders) == 1
        assert len(pending_orders) == 1
        assert len(all_orders) == 2
    
    def test_commission_calculation(self, broker, sample_market_data):
        """수수료 계산 테스트"""
        broker.update_market_data(sample_market_data)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        broker.submit_order(order)
        
        trade = broker.trades[0]
        expected_commission = 100.0 * 154.0 * 0.001  # quantity * price * commission_rate
        assert trade.commission == pytest.approx(expected_commission, rel=1e-6)
    
    def test_slippage_calculation(self, broker, sample_market_data):
        """슬리피지 계산 테스트"""
        broker.update_market_data(sample_market_data)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        broker.submit_order(order)
        
        trade = broker.trades[0]
        assert trade.slippage == 0.0001  # 시장가 주문에 슬리피지 적용
    
    def test_portfolio_pnl_calculation(self, broker, sample_market_data):
        """포트폴리오 손익 계산 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 매수
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        broker.submit_order(order)
        
        # 가격 상승 시뮬레이션
        updated_data = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 2)],
            "symbol": ["AAPL"],
            "open": [160.0],
            "high": [165.0],
            "low": [159.0],
            "close": [164.0],  # 10달러 상승
            "volume": [1100000]
        })
        
        updated_batch = MarketDataBatch(
            data=updated_data,
            symbols=["AAPL"],
            timeframe="1D"
        )
        
        broker.update_market_data(updated_batch)
        
        portfolio = broker.get_portfolio()
        position = portfolio.positions["AAPL"]
        
        # 미실현 손익 확인 (164 - 154) * 100 = 1000
        assert position.unrealized_pnl == pytest.approx(1000.0, rel=1e-6)
    
    def test_get_account_summary(self, broker, sample_market_data):
        """계좌 요약 정보 테스트"""
        broker.update_market_data(sample_market_data)
        
        # 거래 실행
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        broker.submit_order(order)
        
        summary = broker.get_account_summary()
        
        assert summary["initial_cash"] == 100000.0
        assert summary["current_cash"] < 100000.0  # 매수로 인한 현금 감소
        assert summary["total_orders"] == 1
        assert summary["total_trades"] == 1
        assert summary["active_positions"] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 