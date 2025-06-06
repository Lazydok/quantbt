# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path
import asyncio

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
import polars as pl
from datetime import datetime

from quantbt import (
    SimpleBacktestEngine, 
    SimpleBroker, 
    TradingStrategy, 
    MarketDataBatch, 
    BacktestConfig,
    UpbitDataProvider,
    Order,
    OrderSide,
    OrderType,
)
print("✅ 모든 QuantBT 모듈이 성공적으로 가져와졌습니다!")

class SimpleSMAStrategy(TradingStrategy):
    """간단한 SMA 브레이크아웃 전략
    
    매수: 가격이 SMA20 상회
    매도: 가격이 SMA5 하회  
    """
    
    def __init__(self, buy_sma: int = 20, sell_sma: int = 5):
        super().__init__(
            name="SimpleSMAStrategy",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma
            },
            position_size_pct=0.8,  # 80%씩 포지션
            max_positions=1
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.indicator_columns = [f"sma_{buy_sma}", f"sma_{sell_sma}"]
        
    def _compute_indicators_for_symbol(self, symbol_data):
        """심볼별 이동평균 지표 계산"""
        
        # 시간순 정렬 확인
        data = symbol_data.sort("timestamp")
        
        # 단순 이동평균 계산
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        
        # 지표 컬럼 추가
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}")
        ])
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성 - 가격과 이동평균 비교"""
        orders = []
        
        if not self.context:
            return orders
        
        for symbol in data.symbols:
            current_price = self.get_current_price(symbol, data)
            if not current_price:
                continue
            
            # 현재 지표 값 조회
            buy_sma = self.get_indicator_value(symbol, f"sma_{self.buy_sma}", data)
            sell_sma = self.get_indicator_value(symbol, f"sma_{self.sell_sma}", data)
            
            if buy_sma is None or sell_sma is None:
                continue
            
            current_positions = self.get_current_positions()
            
            # 매수 신호: 가격이 SMA20 상회 + 포지션 없음
            if current_price > buy_sma and symbol not in current_positions:
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
            
            # 매도 신호: 가격이 SMA5 하회 + 포지션 있음
            elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                )
                orders.append(order)
        
        return orders
    
# 1. 업비트 데이터 프로바이더
upbit_provider = UpbitDataProvider(
    cache_dir="./data/upbit_cache",
    rate_limit_delay=0.1
)

# 2. 백테스팅 설정 (2024년 1년)
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe="1d",  # 일봉
    initial_cash=10_000_000,  # 1천만원
    commission_rate=0.0,      # 수수료 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
    slippage_rate=0.0         # 슬리피지 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
)

# 3. 간단한 SMA 전략
strategy = SimpleSMAStrategy(
    buy_sma=20,   # 매수: 가격이 20일 이평선 상회
    sell_sma=5    # 매도: 가격이 5일 이평선 하회
)

# 4. 브로커 설정
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
    slippage_rate=config.slippage_rate
)

# 5. 백테스트 엔진
engine = SimpleBacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(upbit_provider)
engine.set_broker(broker)

async def run():
    # 6. 백테스팅 실행
    print(f"📅 기간: {config.start_date.date()} ~ {config.end_date.date()}")
    print(f"📈 전략: 가격 > SMA{strategy.buy_sma} 매수, 가격 < SMA{strategy.sell_sma} 매도")
    print(f"💰 초기 자본: {config.initial_cash:,.0f}원")
    print(f"📊 수수료: {config.commission_rate:.1%} | 슬리피지: {config.slippage_rate:.1%}")

    result = await engine.run(config)

    return result



result = asyncio.run(run())

 # 7. 결과 출력
print(f"\n📊 백테스팅 결과")
print("=" * 40)
print(f"💰 초기 자본: {result.config.initial_cash:,.0f}원")
print(f"💵 최종 자산: {result.final_equity:,.0f}원")
print(f"📈 총 수익률: {result.total_return:.2%}")
print(f"🔄 총 거래 수: {result.total_trades}")

if result.total_trades > 0:
    print(f"🎯 승률: {result.win_rate:.1%}")
    
    # 주요 거래 내역 (상위 10개)
    if hasattr(result, 'trades') and result.trades:
        print(f"\n📋 주요 거래 내역 (상위 10개)")
        print("-" * 70)
        
        for i, trade in enumerate(result.trades[:10], 1):
            date = trade.timestamp.strftime("%Y-%m-%d")
            side = "매수" if trade.side.value == "BUY" else "매도"
            
            print(f"{i:2d}. {date} | {side} | "
                    f"{trade.quantity:.6f} BTC @ {trade.price:,.0f}원")
else:
    print("❌ 거래가 발생하지 않았습니다.")
