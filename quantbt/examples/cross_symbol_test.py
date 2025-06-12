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
    # Phase 7 하이브리드 전략 시스템
    TradingStrategy,
    BacktestEngine,  # Dict Native 엔진 사용!
    
    # 기본 모듈들
    SimpleBroker, 
    BacktestConfig,
    UpbitDataProvider,
    
    # 주문 관련
    Order, OrderSide, OrderType,
)


class MultiSymbolSMAStrategy(TradingStrategy):
    """변동성 기반 심볼 선택 SMA 전략
    
    하이브리드 방식:
    - 지표 계산: Polars 벡터연산 (SMA + 변동성)
    - 심볼간 비교: precompute_indicators에서 타임스탬프별 변동성 순위 계산
    - 신호 생성: Dict Native 방식 (변동성 1등 심볼만 거래)
    
    매수: 가격이 SMA15 상회 + 변동성 순위 1등
    매도: 가격이 SMA30 하회 + 포지션 보유중
    """
    
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30, volatility_window: int = 14, symbols: List[str] = ["KRW-BTC", "KRW-ETH"]):
        super().__init__(
            name="VolatilityBasedMultiSymbolSMAStrategy",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma,
                "volatility_window": volatility_window
            },
            position_size_pct=0.8,  # 80%씩 포지션
            max_positions=1
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.volatility_window = volatility_window
        self.symbols = symbols
        
    def calculate_volatility(self, prices: pl.Series, window: int = 14) -> pl.Series:
        """롤링 표준편차 기반 변동성 계산"""
        returns = prices.pct_change()
        return returns.rolling_std(window_size=window)
        
    def _compute_indicators_for_symbol(self, symbol_data):
        """심볼별 기본 지표 계산 (Polars 벡터 연산)"""
        
        # 시간순 정렬 확인
        data = symbol_data.sort("timestamp")
        
        # 단순 이동평균 계산
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        
        # 변동성 계산 (표준편차 기반)
        volatility = self.calculate_volatility(data["close"], self.volatility_window)
        
        # 지표 컬럼 추가
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}"),
            volatility.alias("volatility")
        ])
    
    # precompute_indicators는 BaseStrategy에서 표준 2단계 처리로 자동 실행됨
    # 1단계: _compute_indicators_for_symbol (심볼별 지표)
    # 2단계: _compute_cross_symbol_indicators (심볼간 비교)
    
    def _compute_cross_symbol_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """심볼간 비교 지표 계산 - 시간 동기화 보장 (완전 벡터 연산)"""
        
        # 🚀 완전 벡터 연산 방식: window function 활용
        ranked_data = data.with_columns([
            # 타임스탬프별로 변동성 순위 계산 (None/NaN을 inf로 처리)
            pl.col("volatility")
            .fill_null(float('inf'))
            .fill_nan(float('inf'))
            .rank("ordinal")
            .over("timestamp")  # 타임스탬프별 윈도우 함수
            .alias("vol_rank"),
            
            # 타임스탬프별로 최소 변동성인지 판단
            (pl.col("volatility") == pl.col("volatility").min().over("timestamp"))
            .alias("is_lowest_volatility")
        ])
        
        return ranked_data
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """변동성 순위 기반 필터링이 적용된 신호 생성"""
        orders = []
        
        if not self.broker:
            return orders
        
        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')
        vol_rank = current_data.get('vol_rank', 999)
        
        # 변동성 순위 기반 필터링 (1등만 거래)
        if vol_rank != 1:
            return orders  # 변동성 1등이 아니면 거래 중단
        
        # 지표가 계산되지 않은 경우 건너뛰기
        if buy_sma is None or sell_sma is None:
            return orders
        
        current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 SMA15 상회 + 포지션 없음 + 변동성 1등
        if current_price > buy_sma and symbol not in current_positions:
            portfolio_value = self.get_portfolio_value()
            quantity = self.calculate_position_size(symbol, current_price, portfolio_value) / len(self.symbols)
            
            if quantity > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                # print(f"🎯 변동성 1등 매수: {symbol} @ {current_price:,.0f}원 (SMA{self.buy_sma}: {buy_sma:,.0f}, Vol순위: {vol_rank})")
        
        # 매도 신호: 가격이 SMA30 하회 + 포지션 있음 (변동성 순위 무관)
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_positions[symbol],
                order_type=OrderType.MARKET
            )
            orders.append(order)
            # print(f"📉 매도 신호: {symbol} @ {current_price:,.0f}원 (SMA{self.sell_sma}: {sell_sma:,.0f})")
        
        return orders



# 1. 업비트 데이터 프로바이더
print("🔄 데이터 프로바이더 초기화 중...")
upbit_provider = UpbitDataProvider()

# 2. 백테스팅 설정 (Phase 7 최적화)
config = BacktestConfig(
    symbols=["KRW-BTC", "KRW-ETH"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31), 
    timeframe="1m", 
    initial_cash=10_000_000,  # 1천만원
    commission_rate=0.0,      # 수수료 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
    slippage_rate=0.0,         # 슬리피지 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
    save_portfolio_history=True
)

# 3. Phase 7 하이브리드 + 변동성 기반 SMA 전략
print("⚡ Phase 7 하이브리드 + 변동성 필터 전략 초기화 중...")
strategy = MultiSymbolSMAStrategy(
    buy_sma=15,   # 매수: 가격이 15일 이평선 상회
    sell_sma=30,   # 매도: 가격이 30일 이평선 하회
    volatility_window=14,  # 변동성 계산 기간 (14일)
    symbols=["KRW-BTC", "KRW-ETH"],
)

# 4. 브로커 설정
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
    slippage_rate=config.slippage_rate
)

# 5. Dict Native 백테스트 엔진 (Phase 7)
print("🚀 Dict Native 백테스트 엔진 초기화 중...")
engine = BacktestEngine()  # Dict Native 엔진 사용!
engine.set_strategy(strategy)
engine.set_data_provider(upbit_provider)
engine.set_broker(broker)

# 6. 백테스팅 실행
print("🎯 변동성 기반 멀티심볼 백테스팅 시작...")
print("📊 전략: 변동성이 가장 낮은 심볼에만 투자")
print(f"📈 매수 조건: 가격 > SMA{strategy.buy_sma} + 변동성 순위 1등")
print(f"📉 매도 조건: 가격 < SMA{strategy.sell_sma}")
print("=" * 60)

result = engine.run(config)
    
# 7. 결과 요약 출력
print("\n" + "=" * 60)
print("🎉 변동성 기반 멀티심볼 백테스팅 완료!")
print("=" * 60)
result.print_summary()