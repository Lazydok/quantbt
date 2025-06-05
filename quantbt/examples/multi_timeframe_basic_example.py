#!/usr/bin/env python3
"""
QuantBT 멀티타임프레임 백테스팅 기본 예제

이 예제는 다음을 보여줍니다:
1. 1시간봉 트렌드 분석 (장기)
2. 5분봉 진입 신호 생성 (단기)
3. 두 타임프레임 조합으로 매매 결정

전략 로직:
- 1시간봉 SMA 기반 트렌드 확인
- 5분봉 RSI + SMA 교차 기반 진입/청산
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import polars as pl

# QuantBT 모듈 추가 (quantbt/examples에서 실행되는 경우)
sys.path.append(str(Path(__file__).parent.parent.parent))

from quantbt.core.interfaces.strategy import MultiTimeframeTradingStrategy
from quantbt.core.entities.order import Order, OrderSide, OrderType
from quantbt.infrastructure.engine.simple_engine import SimpleBacktestEngine
from quantbt.infrastructure.data.csv_provider import CSVDataProvider
from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.core.value_objects.backtest_config import BacktestConfig


class BasicMultiTimeframeStrategy(MultiTimeframeTradingStrategy):
    """기본 멀티타임프레임 전략
    
    1시간봉: 장기 트렌드 분석 (SMA 20/50 교차)
    5분봉: 단기 진입 신호 (RSI + SMA 10/20 교차)
    """
    
    def __init__(self):
        super().__init__(
            name="BasicMultiTimeframe",
            timeframes=["1m", "5m", "1h"],  # 1분, 5분, 1시간봉 사용
            position_size_pct=0.8,  # 80% 포지션 
            max_positions=2         # 최대 2개 동시 포지션
        )
        
        # 전략 파라미터
        self.hourly_short_sma = 20
        self.hourly_long_sma = 50
        self.signal_short_sma = 10
        self.signal_long_sma = 20
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        print(f"🎯 {self.name} 전략 초기화 완료")
        print(f"   • 타임프레임: {self.timeframes}")
        print(f"   • 1시간봉 SMA: {self.hourly_short_sma}/{self.hourly_long_sma}")
        print(f"   • 5분봉 SMA: {self.signal_short_sma}/{self.signal_long_sma}")
        print(f"   • RSI 기준: {self.rsi_oversold}/{self.rsi_overbought}")
    
    def precompute_indicators_multi_timeframe(self, data_dict):
        """각 타임프레임별 지표 사전 계산"""
        result = {}
        
        for timeframe, df in data_dict.items():
            # 심볼별로 그룹화하여 지표 계산
            enriched_data = df.sort(["symbol", "timestamp"]).group_by("symbol").map_groups(
                lambda group: self._compute_indicators_for_timeframe(group, timeframe)
            )
            
            result[timeframe] = enriched_data
            
        return result
    
    def _compute_indicators_for_timeframe(self, symbol_data, timeframe):
        """특정 타임프레임의 심볼 데이터에 대해 지표 계산"""
        
        if timeframe == "1h":
            # 1시간봉: 장기 트렌드 분석용 SMA
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(self.hourly_short_sma).alias("sma_20"),
                pl.col("close").rolling_mean(self.hourly_long_sma).alias("sma_50"),
                self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi")
            ])
            
        elif timeframe == "5m":
            # 5분봉: 단기 신호 생성용 지표
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(self.signal_short_sma).alias("sma_10"),
                pl.col("close").rolling_mean(self.signal_long_sma).alias("sma_20"),
                self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi")
            ])
            
        else:
            # 1분봉: 기본 지표만
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(10).alias("sma_10"),
                self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi")
            ])
    
    def generate_signals_multi_timeframe(self, multi_data):
        """멀티타임프레임 신호 생성"""
        orders = []
        
        for symbol in multi_data.symbols:
            # 1시간봉 트렌드 분석
            hourly_trend = self._analyze_hourly_trend(multi_data, symbol)
            
            # 5분봉 진입 신호 확인
            entry_signal = self._check_entry_signal(multi_data, symbol)
            
            # 현재 포지션 확인
            current_positions = self.get_current_positions()
            position_count = len(current_positions)
            
            # 매수 조건
            if (hourly_trend == "bullish" and 
                entry_signal == "buy" and 
                symbol not in current_positions and
                position_count < self.max_positions):
                
                current_price = multi_data.get_timeframe_price("5m", symbol, "close")
                if current_price:
                    quantity = self.calculate_position_size(
                        symbol, current_price, self.get_portfolio_value()
                    )
                    
                    if quantity > 0:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET,
                            metadata={
                                "hourly_trend": hourly_trend,
                                "entry_signal": entry_signal,
                                "strategy": "basic_multi_timeframe"
                            }
                        ))
                        
                        print(f"📈 {symbol} 매수 신호: 1H 트렌드={hourly_trend}, 5M 신호={entry_signal}")
            
            # 매도 조건
            elif symbol in current_positions:
                exit_signal = self._check_exit_signal(multi_data, symbol, hourly_trend)
                
                if exit_signal == "sell":
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=current_positions[symbol],
                        order_type=OrderType.MARKET,
                        metadata={
                            "exit_reason": exit_signal,
                            "hourly_trend": hourly_trend
                        }
                    ))
                    
                    print(f"📉 {symbol} 매도 신호: 사유={exit_signal}, 1H 트렌드={hourly_trend}")
        
        return orders
    
    def _analyze_hourly_trend(self, multi_data, symbol):
        """1시간봉 기반 트렌드 분석"""
        current_price = multi_data.get_timeframe_price("1h", symbol, "close")
        sma_20 = multi_data.get_timeframe_indicator("1h", "sma_20", symbol)
        sma_50 = multi_data.get_timeframe_indicator("1h", "sma_50", symbol)
        
        if not all([current_price, sma_20, sma_50]):
            return "neutral"
        
        # 트렌드 판단
        if current_price > sma_20 > sma_50:
            return "bullish"     # 강한 상승 트렌드
        elif current_price > sma_20 and sma_20 < sma_50:
            return "weak_bull"   # 약한 상승
        elif current_price < sma_20 < sma_50:
            return "bearish"     # 강한 하락 트렌드
        elif current_price < sma_20 and sma_20 > sma_50:
            return "weak_bear"   # 약한 하락
        else:
            return "neutral"     # 횡보
    
    def _check_entry_signal(self, multi_data, symbol):
        """5분봉 기반 진입 신호 확인"""
        current_price = multi_data.get_timeframe_price("5m", symbol, "close")
        sma_10 = multi_data.get_timeframe_indicator("5m", "sma_10", symbol)
        sma_20 = multi_data.get_timeframe_indicator("5m", "sma_20", symbol)
        rsi = multi_data.get_timeframe_indicator("5m", "rsi", symbol)
        
        if not all([current_price, sma_10, sma_20, rsi]):
            return "hold"
        
        # 매수 신호: SMA 골든크로스 + RSI 과매도에서 회복
        if (sma_10 > sma_20 and 
            current_price > sma_10 and 
            self.rsi_oversold < rsi < 50):
            return "buy"
        
        # 매도 신호: SMA 데드크로스 + RSI 과매수
        elif (sma_10 < sma_20 and 
              current_price < sma_10 and 
              rsi > self.rsi_overbought):
            return "sell"
        
        return "hold"
    
    def _check_exit_signal(self, multi_data, symbol, hourly_trend):
        """청산 신호 확인"""
        # 1시간봉 트렌드 전환시 무조건 청산
        if hourly_trend in ["bearish", "weak_bear"]:
            return "trend_change"
        
        # 5분봉 기술적 청산 신호
        entry_signal = self._check_entry_signal(multi_data, symbol)
        if entry_signal == "sell":
            return "technical"
        
        # RSI 과매수 구간에서 청산
        rsi = multi_data.get_timeframe_indicator("5m", "rsi", symbol)
        if rsi and rsi > self.rsi_overbought:
            return "overbought"
        
        return "hold"
    
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 호환성을 위한 구현 (주 타임프레임 사용)"""
        # 5분봉 지표만 계산하여 반환
        return symbol_data.with_columns([
            pl.col("close").rolling_mean(self.signal_short_sma).alias("sma_10"),
            pl.col("close").rolling_mean(self.signal_long_sma).alias("sma_20"),
            self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi")
        ])
    
    def generate_signals(self, data) -> List[Order]:
        """단일 타임프레임 호환성을 위한 구현"""
        # MultiTimeframeDataBatch가 아닌 경우 단순 전략으로 대체
        if not hasattr(data, 'timeframes'):
            # 기본 SMA 교차 전략으로 작동
            orders = []
            
            for symbol in data.symbols:
                current_price_data = data.get_latest(symbol)
                if not current_price_data:
                    continue
                
                latest_data = data.get_latest_with_indicators(symbol)
                if not latest_data:
                    continue
                
                sma_10 = latest_data.get("sma_10")
                sma_20 = latest_data.get("sma_20")
                
                if sma_10 and sma_20:
                    current_positions = self.get_current_positions()
                    
                    # 매수 신호
                    if (sma_10 > sma_20 and 
                        symbol not in current_positions and 
                        len(current_positions) < self.max_positions):
                        
                        current_price = current_price_data.close
                        quantity = self.calculate_position_size(
                            symbol, current_price, self.get_portfolio_value()
                        )
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                    
                    # 매도 신호
                    elif sma_10 < sma_20 and symbol in current_positions:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_positions[symbol],
                            order_type=OrderType.MARKET
                        ))
            
            return orders
        else:
            # 멀티타임프레임 신호 생성
            return self.generate_signals_multi_timeframe(data)


async def generate_sample_data():
    """샘플 데이터 생성"""
    # 2개 종목, 3개월 치 1분봉 데이터
    symbols = ["BTC", "ETH"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 4, 1)
    
    data_rows = []
    
    for symbol in symbols:
        
        # 각 종목별로 시작 가격 설정
        base_price = 50000 if symbol == "BTC" else 3000
        current_price = base_price
        
        current_time = start_date
        
        while current_time < end_date:
            # 랜덤 가격 변동 (리얼한 패턴)
            import random
            change_pct = random.uniform(-0.005, 0.005)  # ±0.5% 변동
            current_price *= (1 + change_pct)
            
            # OHLCV 생성
            open_price = current_price
            close_price = current_price * (1 + random.uniform(-0.002, 0.002))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
            volume = random.uniform(100, 1000)
            
            data_rows.append({
                "timestamp": current_time,
                "symbol": symbol,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
            
            current_price = close_price
            current_time += timedelta(minutes=1)
    
    return pl.DataFrame(data_rows)


async def main():
    """메인 백테스팅 실행"""
    try:
        # 1. 샘플 데이터 생성
        sample_data = await generate_sample_data()
        
        # 2. 임시 CSV 파일로 저장 (프로젝트 루트에)
        data_dir = Path(__file__).parent.parent.parent / "temp_data"
        data_dir.mkdir(exist_ok=True)
        
        for symbol in sample_data["symbol"].unique():
            symbol_data = sample_data.filter(pl.col("symbol") == symbol)
            csv_path = data_dir / f"{symbol}.csv"
            symbol_data.write_csv(csv_path)
        
        # 3. 백테스팅 컴포넌트 설정
        data_provider = CSVDataProvider(str(data_dir))
        broker = SimpleBroker(
            initial_cash=100000,
            commission_rate=0.001,
            slippage_rate=0.0001
        )
        strategy = BasicMultiTimeframeStrategy()
        
        # 4. 백테스팅 설정
        config = BacktestConfig(
            symbols=["BTC", "ETH"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 1),
            initial_cash=100000,
            timeframe="1m",  # 기준 타임프레임
            commission_rate=0.001
        )
        
        # 5. 백테스팅 실행
        engine = SimpleBacktestEngine()
        engine.set_strategy(strategy)
        engine.set_data_provider(data_provider)
        engine.set_broker(broker)
        
        result = await engine.run(config)
        
        # 6. 결과 출력
        print(f"\n" + "="*60)
        print(f"📊 멀티타임프레임 백테스팅 결과")
        print(f"="*60)
        print(f"총 수익률:      {result.total_return_pct:>8.2f}%")
        print(f"연간 수익률:    {result.annual_return_pct:>8.2f}%")
        print(f"변동성:        {result.volatility_pct:>8.2f}%")
        print(f"샤프 비율:     {result.sharpe_ratio:>8.2f}")
        print(f"최대 낙폭:     {result.max_drawdown_pct:>8.2f}%")
        print(f"총 거래 수:     {result.total_trades:>8}")
        print(f"승률:          {result.win_rate_pct:>8.2f}%")
        print(f"평균 수익:     ${getattr(result, 'avg_win', 0):>8.2f}")
        print(f"평균 손실:     ${getattr(result, 'avg_loss', 0):>8.2f}")
        print(f"수익/손실비:    {getattr(result, 'profit_loss_ratio', 0):>8.2f}")
        print(f"="*60)
        
        # 7. 포트폴리오 가치 변화
        if hasattr(result, 'portfolio_values') and result.portfolio_values:
            print(f"\n📈 포트폴리오 가치 변화:")
            values = result.portfolio_values
            print(f"   시작: ${values[0]:,.2f}")
            print(f"   최고: ${max(values):,.2f}")
            print(f"   최저: ${min(values):,.2f}")
            print(f"   최종: ${values[-1]:,.2f}")
        
        # 8. 청소
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"\n🧹 임시 데이터 정리 완료")
        
        print(f"\n✅ 백테스팅 완료!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 비동기 실행
    result = asyncio.run(main())
    
    if result:
        print(f"\n💡 이 예제는 다음을 보여줍니다:")
        print(f"   • 1시간봉과 5분봉을 조합한 멀티타임프레임 분석")
        print(f"   • 장기 트렌드 확인 + 단기 진입 타이밍 최적화")
        print(f"   • 리스크 관리 (최대 포지션 수 제한)")
        print(f"   • 다양한 청산 조건 (트렌드 전환, 기술적 신호)")
    else:
        print(f"\n🔧 문제가 발생했습니다. 로그를 확인해주세요.") 