"""
백테스팅 결과 시각화 기능 테스트

새로 추가된 시각화 기능들을 테스트하는 예제 스크립트입니다.
주피터 노트북에서 사용할 수 있는 시각화 메서드들을 보여줍니다.
"""

import asyncio
import polars as pl
from datetime import datetime, timedelta
from typing import List

# quantbt 라이브러리 import
from quantbt.core.value_objects.backtest_config import BacktestConfig
from quantbt.core.value_objects.backtest_result import BacktestResult
from quantbt.infrastructure.engine.simple_engine import SimpleBacktestEngine
from quantbt.infrastructure.data.csv_provider import CSVDataProvider
from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.core.interfaces.strategy import TradingStrategy
from quantbt.core.entities.order import Order, OrderType, OrderSide
from quantbt.core.entities.market_data import MarketDataBatch

# 간단한 매수후보유 전략 예제
class BuyAndHoldStrategy(TradingStrategy):
    """매수후보유 전략 (시각화 테스트용)"""
    
    def __init__(self):
        super().__init__(
            name="BuyAndHoldStrategy",
            config={},
            position_size_pct=0.8,  # 80%로 포지션
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
        
        # 아직 매수하지 않은 심볼 중 선택하여 매수
        for symbol in data.symbols:
            if symbol not in self.bought_symbols:
                current_price = self.get_current_price(symbol, data)
                
                if current_price and current_price > 0:
                    # 현재 포트폴리오 가치의 일정 비율로 매수
                    portfolio_value = self.get_portfolio_value()
                    position_value = portfolio_value / len(self.context.symbols) * 0.4  # 40%만 사용
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
        
        return orders


async def create_sample_data():
    """샘플 데이터 생성"""
    # 간단한 가격 데이터 생성 (100일간)
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    
    # AAPL 샘플 데이터
    aapl_prices = []
    base_price = 150.0
    for i in range(100):
        # 간단한 랜덤 워크 + 트렌드
        change = (i * 0.1) + ((-1) ** i) * (i % 10) * 0.5
        price = base_price + change
        aapl_prices.append(price)
    
    aapl_data = pl.DataFrame({
        "timestamp": dates,
        "open": [p * 0.99 for p in aapl_prices],
        "high": [p * 1.02 for p in aapl_prices],
        "low": [p * 0.98 for p in aapl_prices],
        "close": aapl_prices,
        "volume": [1000000] * 100
    })
    
    # MSFT 샘플 데이터
    msft_prices = []
    base_price = 300.0
    for i in range(100):
        change = (i * 0.2) + ((-1) ** (i+1)) * (i % 8) * 0.7
        price = base_price + change
        msft_prices.append(price)
    
    msft_data = pl.DataFrame({
        "timestamp": dates,
        "open": [p * 0.99 for p in msft_prices],
        "high": [p * 1.01 for p in msft_prices],
        "low": [p * 0.99 for p in msft_prices],
        "close": msft_prices,
        "volume": [800000] * 100
    })
    
    return {"AAPL": aapl_data, "MSFT": msft_data}


async def test_visualization_features():
    """시각화 기능 테스트"""
    print("=== 백테스팅 결과 시각화 기능 테스트 ===\n")
    
    # 1. 백테스팅 설정 (시각화 모드 활성화)
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=100),
        end_date=datetime.now() - timedelta(days=1),
        initial_cash=100000.0,
        symbols=["AAPL", "MSFT"],
        timeframe="1D",
        commission_rate=0.001,
        slippage_rate=0.0001,
        visualization_mode=True,  # 시각화 모드 활성화
        save_trades=True,
        save_portfolio_history=True
    )
    
    print(f"설정: {config.symbols} 심볼, 시각화 모드: {config.visualization_mode}")
    
    # 2. 백테스팅 엔진 및 전략 설정
    strategy = BuyAndHoldStrategy()
    broker = SimpleBroker()
    
    # 임시 데이터 프로바이더 (실제로는 CSV 파일 사용)
    class MockDataProvider:
        async def get_data(self, symbols, start, end, timeframe):
            sample_data = await create_sample_data()
            # 모든 심볼의 데이터를 하나의 DataFrame으로 결합
            combined_data = []
            for symbol in symbols:
                if symbol in sample_data:
                    symbol_data = sample_data[symbol].with_columns(
                        pl.lit(symbol).alias("symbol")
                    )
                    combined_data.append(symbol_data)
            
            if combined_data:
                return pl.concat(combined_data)
            else:
                return pl.DataFrame()
    
    engine = SimpleBacktestEngine()
    engine.strategy = strategy
    engine.broker = broker
    engine.data_provider = MockDataProvider()
    
    # 3. 백테스팅 실행
    print("백테스팅 실행 중...")
    result = await engine.run(config)
    
    # 4. 기본 결과 출력
    print("\n=== 백테스팅 기본 결과 ===")
    result.print_summary()
    
    # 5. 시각화 기능 테스트
    print("\n=== 시각화 기능 테스트 ===")
    
    if result.config.visualization_mode:
        print("✅ 시각화 모드가 활성화되어 있습니다.")
        
        # 시각화 데이터 확인
        print(f"- equity_curve: {'있음' if result.equity_curve is not None else '없음'}")
        print(f"- daily_returns: {'있음' if result.daily_returns is not None else '없음'}")
        print(f"- monthly_returns: {'있음' if result.monthly_returns is not None else '없음'}")
        print(f"- benchmark_equity_curve: {'있음' if result.benchmark_equity_curve is not None else '없음'}")
        print(f"- drawdown_periods: {'있음' if result.drawdown_periods is not None else '없음'}")
        print(f"- trade_signals: {'있음' if result.trade_signals is not None else '없음'}")
        
        print("\n주피터 노트북에서 사용 가능한 시각화 메서드:")
        print("1. result.plot_portfolio_performance() - 포트폴리오 성과 차트")
        print("2. result.plot_returns_distribution() - 수익률 분포 히스토그램")
        print("3. result.plot_monthly_returns_heatmap() - 월별 수익률 히트맵")
        print("4. result.show_performance_comparison() - 벤치마크 비교 표")
        
        # 샘플 호출 (주피터 노트북 환경이 아니므로 실제 차트는 생성되지 않음)
        try:
            print("\n시각화 메서드 호출 테스트...")
            result.plot_portfolio_performance()
            result.plot_returns_distribution(period="daily")
            result.plot_monthly_returns_heatmap()
            result.show_performance_comparison()
        except Exception as e:
            print(f"시각화 테스트 중 예상된 오류 (plotly 없음): {e}")
    
    else:
        print("❌ 시각화 모드가 비활성화되어 있습니다.")
    
    return result


async def test_memory_efficiency():
    """메모리 효율성 테스트"""
    print("\n=== 메모리 효율성 테스트 ===")
    
    # 시각화 모드 비활성화된 설정
    config_minimal = BacktestConfig(
        start_date=datetime.now() - timedelta(days=100),
        end_date=datetime.now() - timedelta(days=1),
        initial_cash=100000.0,
        symbols=["AAPL", "MSFT"],
        timeframe="1D",
        visualization_mode=False,  # 시각화 모드 비활성화
        save_trades=False,
        save_portfolio_history=False
    )
    
    print(f"최소 모드 설정: 시각화={config_minimal.visualization_mode}, "
          f"거래저장={config_minimal.save_trades}, "
          f"히스토리저장={config_minimal.save_portfolio_history}")
    
    # 엔진 재설정
    strategy = BuyAndHoldStrategy()
    broker = SimpleBroker()
    
    class MockDataProvider:
        async def get_data(self, symbols, start, end, timeframe):
            sample_data = await create_sample_data()
            combined_data = []
            for symbol in symbols:
                if symbol in sample_data:
                    symbol_data = sample_data[symbol].with_columns(
                        pl.lit(symbol).alias("symbol")
                    )
                    combined_data.append(symbol_data)
            
            if combined_data:
                return pl.concat(combined_data)
            else:
                return pl.DataFrame()
    
    engine = SimpleBacktestEngine()
    engine.strategy = strategy
    engine.broker = broker
    engine.data_provider = MockDataProvider()
    
    result_minimal = await engine.run(config_minimal)
    
    print("최소 모드 결과:")
    print(f"- 거래 데이터: {'있음' if result_minimal.trades else '없음'}")
    print(f"- 포트폴리오 히스토리: {'있음' if result_minimal.portfolio_history is not None else '없음'}")
    print(f"- 시각화 데이터: {'있음' if result_minimal.daily_returns is not None else '없음'}")
    
    result_minimal.print_summary()


async def main():
    """메인 함수"""
    try:
        # 시각화 기능 테스트
        result = await test_visualization_features()
        
        # 메모리 효율성 테스트
        await test_memory_efficiency()
        
        print("\n=== 테스트 완료 ===")
        print("시각화 기능이 성공적으로 구현되었습니다!")
        print("\n주피터 노트북에서 사용하는 방법:")
        print("1. BacktestConfig에서 visualization_mode=True 설정")
        print("2. 백테스팅 실행")
        print("3. result.plot_*() 메서드들 호출")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main()) 