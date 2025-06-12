# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path

# 현재 노트북의 위치에서 프로젝트 루트 찾기
current_dir = Path.cwd()
if 'examples' in str(current_dir):
    # examples 폴더에서 실행하는 경우
    project_root = current_dir.parent.parent
else:
    # 프로젝트 루트에서 실행하는 경우
    project_root = current_dir

# 프로젝트 루트를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 필요한 모듈 가져오기
from typing import List, Dict, Any, Optional
from datetime import datetime
import polars as pl

from quantbt import (
    # 멀티 타임프레임 전략 시스템
    MultiTimeframeTradingStrategy,
    BacktestEngine,
    
    # 기본 모듈들
    SimpleBroker, 
    BacktestConfig,
    UpbitDataProvider,
    
    # 주문 관련
    Order, OrderSide, OrderType,
)


class MultiTimeframeSMAStrategy(MultiTimeframeTradingStrategy):
    """멀티 타임프레임 SMA 전략
    
    1분봉: 단기 SMA 크로스오버 신호 생성
    5분봉: 중기 추세 확인 및 RSI 필터링
    
    매수 조건:
    - 1분봉: 가격이 SMA10 상회 + SMA10 > SMA20
    - 5분봉: SMA5 > SMA15 (상승 추세) + RSI 30-70 구간
    
    매도 조건:
    - 1분봉: 가격이 SMA20 하회
    - 5분봉: RSI > 75 (과매수) 또는 SMA5 < SMA15 (하락 추세)
    """
    
    def __init__(self):
        timeframe_configs = {
            "1m": {
                "sma_windows": [10, 20],
                "volume_threshold": 1.2
            },
            "5m": {
                "sma_windows": [5, 15], 
                "rsi_period": 14,
                "volatility_window": 10
            }
        }
        
        super().__init__(
            name="MultiTimeframeSMA",
            timeframe_configs=timeframe_configs,
            primary_timeframe="1m",
            position_size_pct=0.8,  # 80% 포지션 크기
            max_positions=1
        )
        
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str, 
        config: Dict[str, Any]
    ) -> pl.DataFrame:
        """심볼별 + 타임프레임별 지표 계산 (벡터 연산)"""
        
        data = symbol_data.sort("timestamp")
        indicators = []
        
        if timeframe == "1m":
            # 1분봉: SMA + 볼륨 지표
            indicators.extend([
                pl.col("close").rolling_mean(10).alias("sma_10"),
                pl.col("close").rolling_mean(20).alias("sma_20"),
                pl.col("volume").rolling_mean(20).alias("volume_sma"),
                (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio")
            ])
            
        elif timeframe == "5m":
            # 5분봉: SMA + RSI + 변동성
            indicators.extend([
                pl.col("close").rolling_mean(5).alias("sma_5"),
                pl.col("close").rolling_mean(15).alias("sma_15"),
                self.calculate_rsi(pl.col("close"), 14).alias("rsi_14"),
                pl.col("close").rolling_std(10).alias("volatility_10")
            ])
        
        return data.with_columns(indicators)
    
    def calculate_rsi(self, prices: pl.Expr, period: int = 14) -> pl.Expr:
        """RSI 계산 (최적화된 Polars 벡터 연산)
        
        순수 벡터 연산으로 RSI를 계산하여 성능을 최적화했습니다.
        
        Args:
            prices: 가격 데이터 (Polars Expression)
            period: RSI 계산 기간 (기본값: 14)
            
        Returns:
            RSI 값 (0-100 범위)
        """
        price_change = prices.diff(1)
        
        # 🚀 최적화: pl.when 사용으로 벡터화 (map_elements 대신)
        gains = pl.when(price_change > 0).then(price_change).otherwise(0)
        losses = pl.when(price_change < 0).then(-price_change).otherwise(0)
        
        # 지수이동평균 계산 (RSI 표준 방식)
        avg_gains = gains.ewm_mean(span=period)
        avg_losses = losses.ewm_mean(span=period)
        
        # 0으로 나누기 방지
        rs = avg_gains / pl.when(avg_losses > 0).then(avg_losses).otherwise(1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals_multi_timeframe(
        self, 
        multi_current_data: Dict[str, Dict[str, Any]]
    ) -> List[Order]:
        """멀티 타임프레임 신호 생성"""
        
        orders = []
        
        # 1분봉과 5분봉 데이터 확인
        data_1m = multi_current_data.get("1m")
        data_5m = multi_current_data.get("5m")
        
        if not data_1m or not data_5m:
            return orders
        
        symbol = data_1m.get('symbol')
        if not symbol:
            return orders
        
        # 멀티 타임프레임 분석
        signal = self._analyze_multi_timeframe_signal(data_1m, data_5m, symbol)
        
        if signal == "BUY":
            orders.append(self._create_buy_order(symbol, data_1m))
        elif signal == "SELL":
            sell_order = self._create_sell_order(symbol)
            if sell_order:
                orders.append(sell_order)
        
        return orders
    
    def _analyze_multi_timeframe_signal(
        self, 
        data_1m: Dict[str, Any], 
        data_5m: Dict[str, Any], 
        symbol: str
    ) -> str:
        """크로스 타임프레임 신호 분석"""
        
        # 1분봉 조건 (None 값 처리)
        price_1m = data_1m.get('close', 0) or 0
        sma_10_1m = data_1m.get('sma_10') or 0
        sma_20_1m = data_1m.get('sma_20') or 0
        volume_ratio_1m = data_1m.get('volume_ratio') or 1
        
        # 5분봉 조건 (None 값 처리)
        sma_5_5m = data_5m.get('sma_5') or 0
        sma_15_5m = data_5m.get('sma_15') or 0
        rsi_5m = data_5m.get('rsi_14') or 50
        
        # 지표가 계산되지 않은 경우 HOLD 반환
        if not all([sma_10_1m, sma_20_1m, sma_5_5m, sma_15_5m]):
            return "HOLD"
        
        current_positions = self.get_current_positions()
        
        # 매수 조건: 1분봉 + 5분봉 모든 조건 만족
        buy_conditions = [
            price_1m > sma_10_1m,           # 1분봉 단기 상승
            sma_10_1m > sma_20_1m,          # 1분봉 골든크로스
            volume_ratio_1m > 1.2,          # 거래량 증가
            sma_5_5m > sma_15_5m,           # 5분봉 상승 추세
            30 < rsi_5m < 70,               # RSI 적정 구간
            symbol not in current_positions
        ]
        
        # 매도 조건
        sell_conditions = [
            price_1m < sma_20_1m,           # 1분봉 지지선 이탈
            sma_5_5m < sma_15_5m,           # 5분봉 하락 추세
            rsi_5m > 75 or rsi_5m < 25,     # RSI 극값
            symbol in current_positions and current_positions[symbol] > 0
        ]
        
        if all(buy_conditions):
            return "BUY"
        elif any(sell_conditions):
            return "SELL"
        else:
            return "HOLD"
    
    def _create_buy_order(self, symbol: str, data_1m: Dict[str, Any]) -> Order:
        """매수 주문 생성"""
        current_price = data_1m.get('close', 0)
        portfolio_value = self.get_portfolio_value()
        quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
        
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
    
    def _create_sell_order(self, symbol: str) -> Optional[Order]:
        """매도 주문 생성"""
        current_positions = self.get_current_positions()
        quantity = current_positions.get(symbol, 0)
        
        if quantity > 0:
            return Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
        return None



# 1. 업비트 데이터 프로바이더
print("🔄 데이터 프로바이더 초기화 중...")
upbit_provider = UpbitDataProvider()

# 2. 멀티 타임프레임 백테스팅 설정
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),  # 1주일 테스트
    timeframes=["1m", "5m"],        # 멀티 타임프레임 설정
    primary_timeframe="1m",         # 주요 타임프레임
    initial_cash=10_000_000,        # 1천만원
    commission_rate=0.001,          # 0.1% 수수료
    slippage_rate=0.0005,           # 0.05% 슬리피지
    save_portfolio_history=True
)

# 3. 멀티 타임프레임 SMA 전략
print("⚡ 멀티 타임프레임 SMA 전략 초기화 중...")
strategy = MultiTimeframeSMAStrategy()

print(f"📊 전략명: {strategy.name}")
print(f"🕐 사용 타임프레임: {strategy.available_timeframes}")
print(f"📈 멀티 타임프레임: {strategy.is_multi_timeframe_strategy}")
print(f"🎯 주요 타임프레임: {strategy.primary_timeframe}")

# 4. 브로커 설정
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
    slippage_rate=config.slippage_rate
)

# 5. 멀티 타임프레임 백테스트 엔진
print("🚀 멀티 타임프레임 백테스트 엔진 초기화 중...")
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(upbit_provider)
engine.set_broker(broker)

# 6. 백테스팅 실행
print("\n" + "=" * 60)
print("🚀 멀티 타임프레임 백테스팅 실행 중...")
print("=" * 60)
print(f"📊 심볼: {config.symbols}")
print(f"📅 기간: {config.start_date} ~ {config.end_date}")
print(f"⏰ 타임프레임: {config.timeframes}")
print(f"💰 초기 자본: {config.initial_cash:,}원")
print(f"💸 수수료: {config.commission_rate*100:.1f}%")
print(f"📉 슬리피지: {config.slippage_rate*100:.2f}%")

result = engine.run(config)
    
# 결과 요약 출력
print("\n" + "=" * 60)
print("📊 멀티 타임프레임 백테스팅 결과")
print("=" * 60)
result.print_summary()

# 멀티 타임프레임 특화 정보 출력
if hasattr(result, 'metadata') and result.metadata:
    metadata = result.metadata
    if metadata.get('multi_timeframe_engine'):
        print(f"\n🔧 멀티 타임프레임 엔진 정보:")
        print(f"   📊 사용된 타임프레임: {metadata.get('timeframes', [])}")
        print(f"   🎯 주요 타임프레임: {metadata.get('primary_timeframe', 'unknown')}")
        print(f"   ⏱️  데이터 로딩 시간: {metadata.get('data_load_time', 0):.3f}초")
        print(f"   📈 지표 계산 시간: {metadata.get('indicator_time', 0):.3f}초")
        print(f"   🔄 백테스트 시간: {metadata.get('backtest_time', 0):.3f}초")
        print(f"   📊 총 처리 시간: {metadata.get('total_time', 0):.3f}초")
        
        processing_speed = metadata.get('processing_speed', 0)
        if processing_speed > 0:
            print(f"   ⚡ 처리 속도: {processing_speed:,.0f} trades/second")