"""
멀티타임프레임 백테스팅 실행 예제

실제 멀티타임프레임 전략을 테스트할 수 있는 완전한 예제를 제공합니다.
"""

import asyncio
import polars as pl
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from ..core.entities.market_data import MultiTimeframeDataBatch, MarketDataBatch
from ..core.utils.timeframe import TimeframeUtils
from ..core.value_objects.backtest_config import BacktestConfig
from ..examples.multi_timeframe_strategy import MultiTimeframeSMAStrategy
from ..infrastructure.brokers.simple_broker import SimpleBroker


class MultiTimeframeBacktestExample:
    """멀티타임프레임 백테스팅 예제 클래스"""
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.timeframes = ["5m", "1h"]
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 3, 31)
        
    def generate_sample_1m_data(self) -> pl.DataFrame:
        """1분봉 샘플 데이터 생성"""
        print("📊 1분봉 샘플 데이터 생성 중...")
        
        # 3개월간 1분봉 데이터 생성 (약 129,600개 레코드)
        start_ts = self.start_date
        end_ts = self.end_date
        
        # 1분 간격으로 타임스탬프 생성
        timestamps = []
        current_ts = start_ts
        while current_ts < end_ts:
            # 주말 제외 (월-금만)
            if current_ts.weekday() < 5:
                # 거래시간만 (9:30 - 16:00)
                if 9.5 <= current_ts.hour + current_ts.minute/60 <= 16:
                    timestamps.append(current_ts)
            current_ts += timedelta(minutes=1)
        
        print(f"✅ 생성된 타임스탬프 수: {len(timestamps):,}개")
        
        # 각 심볼별 OHLCV 데이터 생성
        all_data = []
        base_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}
        
        for symbol in self.symbols:
            print(f"   {symbol} 데이터 생성 중...")
            
            base_price = base_prices[symbol]
            current_price = base_price
            
            symbol_data = []
            for ts in timestamps:
                # 랜덤 워크 + 약간의 트렌드
                change_pct = np.random.normal(0, 0.002)  # 0.2% 표준편차
                trend = 0.00001 if ts.hour >= 12 else -0.00001  # 약간의 일중 트렌드
                
                current_price *= (1 + change_pct + trend)
                
                # OHLC 생성 (현실적인 범위)
                volatility = current_price * 0.005  # 0.5% 변동성
                high = current_price + np.random.uniform(0, volatility)
                low = current_price - np.random.uniform(0, volatility)
                open_price = current_price + np.random.uniform(-volatility/2, volatility/2)
                close_price = current_price
                
                # 거래량 (랜덤하지만 현실적인 범위)
                base_volume = 1000000
                volume_multiplier = np.random.lognormal(0, 0.5)
                volume = int(base_volume * volume_multiplier)
                
                symbol_data.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": round(max(open_price, 0.01), 2),
                    "high": round(max(high, open_price, close_price), 2),
                    "low": round(min(low, open_price, close_price), 2),
                    "close": round(max(close_price, 0.01), 2),
                    "volume": volume
                })
            
            all_data.extend(symbol_data)
        
        df = pl.DataFrame(all_data).sort(["timestamp", "symbol"])
        print(f"✅ 전체 1분봉 데이터: {len(df):,}개 레코드")
        return df
    
    def create_multi_timeframe_data(self, base_data: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """멀티타임프레임 데이터셋 생성"""
        print("\n🔄 멀티타임프레임 데이터 생성 중...")
        
        multi_data = TimeframeUtils.create_multi_timeframe_data(
            base_data, self.timeframes
        )
        
        for tf, df in multi_data.items():
            print(f"   {tf}: {len(df):,}개 레코드")
            
        return multi_data
    
    async def run_strategy_backtest(self) -> Dict[str, Any]:
        """멀티타임프레임 전략 백테스팅 실행"""
        print("\n🚀 멀티타임프레임 백테스팅 시작")
        
        # 1. 샘플 데이터 생성
        base_data = self.generate_sample_1m_data()
        multi_data = self.create_multi_timeframe_data(base_data)
        
        # 2. 전략 초기화
        strategy = MultiTimeframeSMAStrategy(
            timeframes=self.timeframes,
            config={
                "hourly_short_period": 10,
                "hourly_long_period": 20, 
                "signal_short_period": 3,
                "signal_long_period": 10
            },
            position_size_pct=0.3,
            max_positions=2
        )
        
        print(f"📈 전략 설정: {strategy.name}")
        print(f"   타임프레임: {strategy.timeframes}")
        print(f"   포지션 크기: {strategy.position_size_pct*100}%")
        print(f"   최대 포지션: {strategy.max_positions}개")
        
        # 3. 지표 사전 계산
        print("\n📊 멀티타임프레임 지표 계산 중...")
        enriched_data = strategy.precompute_indicators_multi_timeframe(multi_data)
        
        for tf, df in enriched_data.items():
            print(f"   {tf}: {df.columns} 컬럼")
        
        # 4. 브로커 초기화
        broker = SimpleBroker(
            initial_cash=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0001
        )
        
        # 5. 시뮬레이션 실행
        print("\n⚡ 시뮬레이션 실행 중...")
        
        # 5분봉 타임스탬프를 기준으로 시뮬레이션
        primary_data = enriched_data["5m"].sort("timestamp")
        unique_timestamps = primary_data["timestamp"].unique().sort()
        
        total_timestamps = len(unique_timestamps)
        print(f"   총 시뮬레이션 시점: {total_timestamps:,}개")
        
        orders_executed = 0
        portfolio_values = []
        
        for i, current_timestamp in enumerate(unique_timestamps):
            # 현재 시점까지의 누적 데이터 생성
            cumulative_data = {}
            for tf, df in enriched_data.items():
                mask = df["timestamp"] <= current_timestamp
                cumulative_data[tf] = df.filter(mask)
            
            # MultiTimeframeDataBatch 생성
            multi_batch = MultiTimeframeDataBatch(cumulative_data, self.symbols)
            
            # 브로커 시장 데이터 업데이트
            if "5m" in cumulative_data and len(cumulative_data["5m"]) > 0:
                latest_5m = cumulative_data["5m"].filter(
                    pl.col("timestamp") == current_timestamp
                )
                if len(latest_5m) > 0:
                    # MarketDataBatch 생성하여 브로커에 전달
                    market_batch = MarketDataBatch(latest_5m, self.symbols, "5m")
                    broker.update_market_data(market_batch)
            
            # 전략 신호 생성
            orders = strategy.generate_signals_multi_timeframe(multi_batch)
            
            # 주문 실행
            for order in orders:
                order_id = broker.submit_order(order)
                executed_order = broker.orders.get(order_id)
                if executed_order and executed_order.status.name == "FILLED":
                    orders_executed += 1
                    strategy.on_order_fill(executed_order)
            
            # 포트폴리오 값 기록
            portfolio_value = self._calculate_portfolio_value(broker, multi_batch)
            portfolio_values.append({
                "timestamp": current_timestamp,
                "portfolio_value": portfolio_value,
                "cash": broker.portfolio.cash,
                "positions": len(broker.portfolio.positions)
            })
            
            # 진행상황 출력
            if i % 1000 == 0 or i == total_timestamps - 1:
                progress = (i + 1) / total_timestamps * 100
                print(f"   진행률: {progress:.1f}% ({i+1:,}/{total_timestamps:,})")
        
        # 6. 결과 분석
        print(f"\n📋 백테스팅 완료!")
        print(f"   실행된 주문 수: {orders_executed}개")
        
        initial_value = portfolio_values[0]["portfolio_value"]
        final_value = portfolio_values[-1]["portfolio_value"]
        total_return = (final_value - initial_value) / initial_value * 100
        
        print(f"   초기 자본: ${initial_value:,.2f}")
        print(f"   최종 자본: ${final_value:,.2f}")
        print(f"   총 수익률: {total_return:.2f}%")
        
        # 최대 포지션 수 확인
        max_positions = max(pv["positions"] for pv in portfolio_values)
        print(f"   최대 동시 포지션: {max_positions}개")
        
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return_pct": total_return,
            "orders_executed": orders_executed,
            "max_positions": max_positions,
            "portfolio_history": portfolio_values,
            "enriched_data": enriched_data
        }
    
    def _calculate_portfolio_value(self, broker, multi_batch: MultiTimeframeDataBatch) -> float:
        """포트폴리오 총 가치 계산"""
        total_value = broker.portfolio.cash
        
        # 포지션 가치 추가
        for symbol, position in broker.portfolio.positions.items():
            if position.quantity > 0:
                current_price = multi_batch.get_timeframe_price("5m", symbol, "close")
                if current_price:
                    total_value += position.quantity * current_price
        
        return total_value
    
    def analyze_results(self, results: Dict[str, Any]):
        """결과 상세 분석"""
        print("\n📈 상세 분석 결과")
        print("=" * 50)
        
        portfolio_history = results["portfolio_history"]
        
        # 수익률 분석
        values = [pv["portfolio_value"] for pv in portfolio_history]
        returns = [
            (values[i] - values[i-1]) / values[i-1] * 100 
            for i in range(1, len(values))
        ]
        
        if returns:
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            max_return = max(returns)
            min_return = min(returns)
            
            print(f"평균 수익률: {avg_return:.4f}%")
            print(f"변동성: {volatility:.4f}%")
            print(f"최대 수익률: {max_return:.2f}%")
            print(f"최대 손실률: {min_return:.2f}%")
            
            if volatility > 0:
                sharpe_ratio = avg_return / volatility
                print(f"샤프 비율: {sharpe_ratio:.2f}")
        
        # 최대 낙폭 계산
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"최대 낙폭: {max_drawdown:.2f}%")
        
        # 포지션 분석
        position_counts = [pv["positions"] for pv in portfolio_history]
        avg_positions = np.mean(position_counts)
        print(f"평균 포지션 수: {avg_positions:.2f}개")
        
        # 멀티타임프레임 데이터 통계
        print("\n📊 멀티타임프레임 데이터 통계")
        for tf, df in results["enriched_data"].items():
            print(f"{tf}: {len(df):,}개 레코드, {len(df.columns)}개 컬럼")


async def main():
    """멀티타임프레임 백테스팅 예제 실행"""
    print("🎯 멀티타임프레임 백테스팅 예제")
    print("=" * 60)
    
    example = MultiTimeframeBacktestExample()
    
    try:
        # 백테스팅 실행
        results = await example.run_strategy_backtest()
        
        # 결과 분석
        example.analyze_results(results)
        
        print("\n✅ 멀티타임프레임 백테스팅 예제 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 