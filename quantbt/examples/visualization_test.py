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

# 필요한 라이브러리 import
import asyncio
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# quantbt 라이브러리
from typing import List

# quantbt 라이브러리 import
from quantbt import (
    UpbitDataProvider,
    SimpleBacktestEngine,
    SimpleBroker,
    TradingStrategy,
    MarketDataBatch,
    BacktestConfig,
    Order,
    OrderType,
    OrderSide,
)

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print("✅ 라이브러리 import 완료")


# SMA 10 기반 매매 전략
class SMAStrategy(TradingStrategy):
    """SMA 10 기반 매매 전략 (시각화 테스트용)"""
    
    def __init__(self, sma_period: int = 10):
        super().__init__(
            name=f"SMAStrategy_{sma_period}",
            config={"sma_period": sma_period},
            position_size_pct=0.95,  # 95% 자금 활용
            max_positions=1
        )
        self.sma_period = sma_period
        self.current_positions = {}  # {symbol: quantity}
        
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """SMA 10 지표 계산"""
        # SMA 계산
        sma_data = symbol_data.with_columns([
            pl.col("close").rolling_mean(window_size=self.sma_period).alias("sma_10")
        ])
        
        # 매매 신호 생성 (SMA와 현재가 비교)
        sma_data = sma_data.with_columns([
            pl.when(pl.col("close") > pl.col("sma_10"))
            .then(1)  # 매수 신호
            .when(pl.col("close") < pl.col("sma_10"))
            .then(-1)  # 매도 신호
            .otherwise(0)  # 보유
            .alias("signal")
        ])
        
        return sma_data
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """SMA 기반 매매 신호 생성"""
        orders = []
        
        if not self.context:
            return orders
        
        for symbol in data.symbols:
            symbol_data = data.get_symbol_data(symbol)
            if symbol_data is None or len(symbol_data) == 0:
                continue
                
            # 최신 데이터만 확인
            latest_data = symbol_data.tail(1)
            if len(latest_data) == 0:
                continue
                
            latest_row = latest_data.row(0, named=True)
            current_price = latest_row.get("close")
            sma_value = latest_row.get("sma_10")
            signal = latest_row.get("signal", 0)
            
            # SMA가 계산되지 않은 경우 (초기 데이터) 건너뛰기
            if sma_value is None or pd.isna(sma_value):
                continue
                
            current_position = self.current_positions.get(symbol, 0)
            
            # 매수 신호: 현재가 > SMA이고 포지션이 없는 경우
            if signal == 1 and current_position == 0:
                portfolio_value = self.get_portfolio_value()
                position_value = portfolio_value * self.position_size_pct
                quantity = position_value / current_price
                
                if quantity > 0.01:  # 최소 수량 확인
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
                    self.current_positions[symbol] = quantity
            
            # 매도 신호: 현재가 < SMA이고 포지션이 있는 경우
            elif signal == -1 and current_position > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_position,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
                self.current_positions[symbol] = 0
        
        return orders
    
# 1. 업비트 데이터 프로바이더
upbit_provider = UpbitDataProvider()

# 2. 백테스팅 설정 (2024년 6개월 - 빠른 테스트용)
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 7, 1),  # 6개월
    timeframe="1d",  # 일봉
    initial_cash=10_000_000,  # 1천만원
    commission_rate=0.001,    # 0.1% 수수료
    slippage_rate=0.001,      # 0.1% 슬리피지
    save_portfolio_history=True
)


# 3. SMA 10 전략
strategy = SMAStrategy(sma_period=10)

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


# 6. 백테스팅 실행
result = asyncio.run(engine.run(config))


print("✅ SMA 10 전략 백테스트 완료!")


result.print_summary()