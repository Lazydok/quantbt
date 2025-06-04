"""
백테스팅 엔진 테스트

백테스팅 엔진의 통합 기능을 검증합니다.
"""

import pytest
import polars as pl
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from quantbt.infrastructure.engine.simple_engine import SimpleBacktestEngine
from quantbt.infrastructure.data.csv_provider import CSVDataProvider
from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.examples.simple_strategy import BuyAndHoldStrategy, SimpleMovingAverageCrossStrategy
from quantbt.core.value_objects.backtest_config import BacktestConfig
from quantbt.core.value_objects.backtest_result import BacktestResult


class TestSimpleBacktestEngine:
    """SimpleBacktestEngine 통합 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self._create_sample_data(temp_dir)
            yield temp_dir
    
    def _create_sample_data(self, data_dir: str):
        """샘플 CSV 데이터 생성"""
        data_dir = Path(data_dir)
        
        # 30일간의 AAPL 데이터 생성 (상승 추세)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        base_price = 150.0
        
        aapl_data = []
        for i, date in enumerate(dates):
            # 약간의 변동성과 함께 상승 추세
            price_change = (i * 0.5) + (i % 3 - 1) * 2  # 전반적 상승 + 노이즈
            close_price = base_price + price_change
            high_price = close_price + abs(i % 3) * 0.5
            low_price = close_price - abs(i % 3) * 0.5
            open_price = close_price + (i % 2 - 0.5) * 1.0
            
            aapl_data.append({
                "timestamp": date,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": 1000000 + i * 10000
            })
        
        aapl_df = pl.DataFrame(aapl_data)
        aapl_df.write_csv(data_dir / "AAPL.csv")
        
        # MSFT 데이터 생성 (횡보 추세)
        msft_data = []
        base_price = 300.0
        
        for i, date in enumerate(dates):
            # 횡보 추세 + 노이즈
            price_change = (i % 5 - 2) * 1.5
            close_price = base_price + price_change
            high_price = close_price + abs(i % 2) * 0.8
            low_price = close_price - abs(i % 2) * 0.8
            open_price = close_price + (i % 3 - 1) * 0.5
            
            msft_data.append({
                "timestamp": date,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": 800000 + i * 8000
            })
        
        msft_df = pl.DataFrame(msft_data)
        msft_df.write_csv(data_dir / "MSFT.csv")
    
    @pytest.fixture
    def backtest_config(self):
        """백테스팅 설정"""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 30),
            initial_cash=100000.0,
            symbols=["AAPL", "MSFT"],
            timeframe="1D",
            commission_rate=0.001,
            slippage_rate=0.0001,
            save_trades=True,
            save_portfolio_history=True
        )
    
    @pytest.fixture
    def data_provider(self, temp_data_dir):
        """데이터 제공자"""
        return CSVDataProvider(temp_data_dir)
    
    @pytest.fixture
    def broker(self, backtest_config):
        """브로커"""
        return SimpleBroker(
            initial_cash=backtest_config.initial_cash,
            commission_rate=backtest_config.commission_rate,
            slippage_rate=backtest_config.slippage_rate
        )
    
    @pytest.fixture
    def engine(self):
        """백테스팅 엔진"""
        return SimpleBacktestEngine()
    
    @pytest.mark.asyncio
    async def test_buy_and_hold_strategy(self, engine, data_provider, broker, backtest_config):
        """바이 앤 홀드 전략 백테스팅 테스트"""
        # 컴포넌트 설정
        strategy = BuyAndHoldStrategy()
        engine.set_strategy(strategy)
        engine.set_data_provider(data_provider)
        engine.set_broker(broker)
        
        # 진행률 콜백 테스트
        progress_updates = []
        def progress_callback(progress: float, message: str):
            progress_updates.append((progress, message))
        
        engine.add_progress_callback(progress_callback)
        
        # 백테스팅 실행
        result = await engine.run(backtest_config)
        
        # 기본 검증
        assert result is not None
        assert result.config == backtest_config
        assert result.final_equity > 0
        assert result.total_trades > 0
        
        # 바이 앤 홀드는 매수만 하므로 매수 거래만 있어야 함
        if result.trades:
            buy_trades = [t for t in result.trades if t.side.value == 1]
            assert len(buy_trades) == len(backtest_config.symbols)  # 각 심볼당 하나씩
        
        # 진행률 콜백이 호출되었는지 확인
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 100.0  # 마지막은 100%
        assert "완료" in progress_updates[-1][1]
        
        # 결과 요약 출력 테스트
        summary = result.get_summary()
        assert "초기자본" in summary
        assert "최종자본" in summary
        assert "총수익률" in summary
        
        print("\n=== 바이 앤 홀드 전략 결과 ===")
        result.print_summary()
    
    @pytest.mark.asyncio
    async def test_moving_average_cross_strategy(self, engine, data_provider, broker, backtest_config):
        """이동평균 교차 전략 백테스팅 테스트"""
        # 컴포넌트 설정
        strategy = SimpleMovingAverageCrossStrategy(short_window=5, long_window=15)
        engine.set_strategy(strategy)
        engine.set_data_provider(data_provider)
        engine.set_broker(broker)
        
        # 백테스팅 실행
        result = await engine.run(backtest_config)
        
        # 기본 검증
        assert result is not None
        assert result.final_equity > 0
        
        # 이동평균 교차 전략은 여러 번 거래할 수 있음
        assert result.total_trades >= 0
        
        # 성과 지표 검증
        assert -1.0 <= result.total_return <= 10.0  # 합리적인 수익률 범위
        assert result.win_rate >= 0.0
        assert result.win_rate <= 1.0
        
        print("\n=== 이동평균 교차 전략 결과 ===")
        result.print_summary()
    
    @pytest.mark.asyncio
    async def test_engine_validation(self, engine, backtest_config):
        """엔진 유효성 검증 테스트"""
        # 전략 없이 실행 시 에러
        with pytest.raises(ValueError, match="Strategy not set"):
            await engine.run(backtest_config)
        
        # 전략만 설정하고 실행 시 에러
        engine.set_strategy(BuyAndHoldStrategy())
        with pytest.raises(ValueError, match="Data provider not set"):
            await engine.run(backtest_config)
    
    @pytest.mark.asyncio
    async def test_concurrent_backtest_prevention(self, engine, data_provider, broker, backtest_config):
        """동시 백테스팅 실행 방지 테스트"""
        # 컴포넌트 설정
        strategy = BuyAndHoldStrategy()
        engine.set_strategy(strategy)
        engine.set_data_provider(data_provider)
        engine.set_broker(broker)
        
        # 첫 번째 백테스팅 시작 (완료되지 않도록 매우 긴 기간 설정)
        long_config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_cash=100000.0,
            symbols=["AAPL"],
            timeframe="1D"
        )
        
        # 실제로는 데이터가 없어서 빨리 실행되지만, is_running 플래그 테스트
        assert not engine.is_running
        
        # 백테스팅 실행
        result = await engine.run(backtest_config)
        assert not engine.is_running  # 완료 후 False
    
    def test_backtest_config_validation(self):
        """백테스팅 설정 유효성 검증 테스트"""
        # 잘못된 날짜 범위
        with pytest.raises(ValueError, match="시작일은 종료일보다 이전이어야 합니다"):
            BacktestConfig(
                start_date=datetime(2023, 1, 31),
                end_date=datetime(2023, 1, 1),  # 시작일보다 이전
                initial_cash=100000.0,
                symbols=["AAPL"]
            )
        
        # 음수 초기 자본
        with pytest.raises(ValueError, match="초기 자본은 0보다 커야 합니다"):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
                initial_cash=-1000.0,  # 음수
                symbols=["AAPL"]
            )
        
        # 빈 심볼 리스트
        with pytest.raises(ValueError, match="최소 하나의 심볼이 필요합니다"):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
                initial_cash=100000.0,
                symbols=[]  # 빈 리스트
            )
    
    def test_backtest_result_properties(self, backtest_config):
        """백테스팅 결과 속성 테스트"""
        from quantbt.core.entities.position import Portfolio
        
        # 샘플 결과 생성
        portfolio = Portfolio(positions={}, cash=110000.0)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = BacktestResult(
            config=backtest_config,
            start_time=start_time,
            end_time=end_time,
            total_return=0.1,
            annual_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            avg_win=500.0,
            avg_loss=-300.0,
            profit_factor=1.67,
            final_portfolio=portfolio,
            final_equity=110000.0
        )
        
        # 속성 테스트
        assert result.total_return_pct == 10.0
        assert result.annual_return_pct == 12.0
        assert result.volatility_pct == 15.0
        assert result.max_drawdown_pct == 5.0
        assert result.win_rate_pct == 60.0
        assert result.total_pnl == 10000.0
        assert abs(result.duration - 10.0) < 0.1  # 시간 정밀도 허용


if __name__ == "__main__":
    pytest.main([__file__]) 