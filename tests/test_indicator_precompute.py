"""
지표 사전 계산 기능 테스트

전략별 지표 사전 계산이 올바르게 동작하는지 테스트합니다.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
from quantbt import (
    SimpleMovingAverageCrossStrategy,
    RSIStrategy,
    BuyAndHoldStrategy,
    MarketDataBatch
)


class TestIndicatorPrecompute:
    """지표 사전 계산 테스트"""
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터 생성"""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        data = []
        
        for i, date in enumerate(dates):
            price = 100 + i * 0.5 + (i % 5 - 2) * 2  # 상승 추세 + 변동성
            data.append({
                "timestamp": date,
                "symbol": "TEST",
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000
            })
        
        return pl.DataFrame(data)
    
    def test_moving_average_precompute(self, sample_data):
        """이동평균 지표 사전 계산 테스트"""
        strategy = SimpleMovingAverageCrossStrategy(short_window=5, long_window=10)
        
        # 지표 계산
        enriched_data = strategy.precompute_indicators(sample_data)
        
        # 지표 컬럼이 추가되었는지 확인
        assert "sma_5" in enriched_data.columns
        assert "sma_10" in enriched_data.columns
        
        # 원본 컬럼들이 그대로 있는지 확인
        original_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        for col in original_columns:
            assert col in enriched_data.columns
        
        # 데이터 수가 동일한지 확인
        assert enriched_data.height == sample_data.height
        
        # 이동평균 값이 올바른지 확인 (5번째 행부터)
        test_data = enriched_data.filter(pl.col("symbol") == "TEST").sort("timestamp")
        for i in range(5, test_data.height):
            row = test_data.row(i, named=True)
            expected_sma5 = test_data.slice(i-4, 5)["close"].mean()
            assert abs(row["sma_5"] - expected_sma5) < 0.01
    
    def test_rsi_precompute(self, sample_data):
        """RSI 지표 사전 계산 테스트"""
        strategy = RSIStrategy(rsi_period=14)
        
        # 지표 계산
        enriched_data = strategy.precompute_indicators(sample_data)
        
        # RSI 컬럼이 추가되었는지 확인
        assert "rsi" in enriched_data.columns
        
        # RSI 값이 0-100 범위에 있는지 확인
        rsi_values = enriched_data.filter(
            pl.col("rsi").is_not_null()
        )["rsi"].to_list()
        
        for rsi in rsi_values:
            assert 0 <= rsi <= 100
    
    def test_buy_and_hold_no_indicators(self, sample_data):
        """바이 앤 홀드 전략 - 지표 없음 테스트"""
        strategy = BuyAndHoldStrategy()
        
        # 지표 계산 (없어야 함)
        enriched_data = strategy.precompute_indicators(sample_data)
        
        # 원본 데이터와 동일해야 함
        assert enriched_data.equals(sample_data)
    
    def test_multi_symbol_precompute(self):
        """다중 심볼 지표 계산 테스트"""
        # 2개 심볼 데이터 생성
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        data = []
        
        for symbol in ["AAPL", "MSFT"]:
            for i, date in enumerate(dates):
                price = 100 + i * 0.3 + (i % 3 - 1) * 1.5
                data.append({
                    "timestamp": date,
                    "symbol": symbol,
                    "open": price - 0.3,
                    "high": price + 0.8,
                    "low": price - 0.8,
                    "close": price,
                    "volume": 1000000
                })
        
        df = pl.DataFrame(data)
        strategy = SimpleMovingAverageCrossStrategy(short_window=3, long_window=7)
        
        # 지표 계산
        enriched_data = strategy.precompute_indicators(df)
        
        # 각 심볼별로 지표가 계산되었는지 확인
        for symbol in ["AAPL", "MSFT"]:
            symbol_data = enriched_data.filter(pl.col("symbol") == symbol)
            assert symbol_data.height == 30
            assert "sma_3" in symbol_data.columns
            assert "sma_7" in symbol_data.columns
    
    def test_market_data_batch_with_indicators(self, sample_data):
        """지표가 포함된 MarketDataBatch 테스트"""
        strategy = SimpleMovingAverageCrossStrategy(short_window=5, long_window=10)
        enriched_data = strategy.precompute_indicators(sample_data)
        
        # MarketDataBatch 생성
        batch = MarketDataBatch(
            data=enriched_data,
            symbols=["TEST"],
            timeframe="1D"
        )
        
        # 지표 컬럼 확인
        assert "sma_5" in batch.indicator_columns
        assert "sma_10" in batch.indicator_columns
        
        # 지표 값 조회 테스트
        latest_data = batch.get_latest_with_indicators("TEST")
        assert latest_data is not None
        assert "sma_5" in latest_data
        assert "sma_10" in latest_data
        
        # 지표 딕셔너리 조회 테스트
        sma5_dict = batch.get_indicator_dict("sma_5")
        assert "TEST" in sma5_dict
        assert isinstance(sma5_dict["TEST"], float)


class TestStrategySignalGeneration:
    """전략 신호 생성 테스트"""
    
    @pytest.fixture
    def enriched_data_batch(self):
        """지표가 포함된 테스트 데이터"""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(20)]
        data = []
        
        for i, date in enumerate(dates):
            price = 100 + i * 0.5
            sma_5 = price - 2 if i >= 5 else None
            sma_10 = price - 4 if i >= 10 else None
            rsi = 50 + (i % 20 - 10) * 2  # 30-70 범위
            
            data.append({
                "timestamp": date,
                "symbol": "TEST",
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000,
                "sma_5": sma_5,
                "sma_10": sma_10,
                "rsi": rsi
            })
        
        df = pl.DataFrame(data)
        return MarketDataBatch(data=df, symbols=["TEST"], timeframe="1D")
    
    def test_moving_average_signal_generation(self, enriched_data_batch):
        """이동평균 교차 신호 생성 테스트"""
        from quantbt.core.interfaces.strategy import BacktestContext
        
        strategy = SimpleMovingAverageCrossStrategy(short_window=5, long_window=10)
        context = BacktestContext(initial_cash=100000, symbols=["TEST"])
        strategy.initialize(context)
        
        # 신호 생성
        orders = strategy.generate_signals(enriched_data_batch)
        
        # 주문이 생성되는지 확인 (골든/데드 크로스 상황에 따라)
        assert isinstance(orders, list)
        
        # 각 주문이 올바른 형식인지 확인
        for order in orders:
            assert hasattr(order, 'symbol')
            assert hasattr(order, 'side')
            assert hasattr(order, 'quantity')
            assert order.quantity > 0
    
    def test_rsi_signal_generation(self, enriched_data_batch):
        """RSI 신호 생성 테스트"""
        from quantbt.core.interfaces.strategy import BacktestContext
        
        strategy = RSIStrategy(oversold=30, overbought=70)
        context = BacktestContext(initial_cash=100000, symbols=["TEST"])
        strategy.initialize(context)
        
        # 신호 생성
        orders = strategy.generate_signals(enriched_data_batch)
        
        # 결과 확인
        assert isinstance(orders, list)
        
        # RSI 값에 따른 신호가 올바른지 확인
        rsi_value = strategy.get_indicator_value("TEST", "rsi", enriched_data_batch)
        if rsi_value is not None:
            if rsi_value < 30:
                # 과매도 구간에서는 매수 신호가 있어야 함
                buy_orders = [o for o in orders if o.side.name == "BUY"]
                assert len(buy_orders) > 0 or len(orders) == 0  # 포지션 제한으로 주문이 없을 수도 있음


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 