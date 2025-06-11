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
    """SMA 전략
    
    하이브리드 방식:
    - 지표 계산: Polars 벡터연산
    - 신호 생성: Dict Native 방식
    
    매수: 가격이 SMA15 상회
    매도: 가격이 SMA30 하회  
    """
    
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30, symbols: List[str] = ["KRW-BTC", "KRW-ETH"]):
        super().__init__(
            name="MultiSymbolSMAStrategy",
            config={
                "buy_sma": buy_sma,
                "sell_sma": sell_sma
            },
            position_size_pct=0.8,  # 80%씩 포지션
            max_positions=1
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.symbols = symbols
        
    def _compute_indicators_for_symbol(self, symbol_data):
        """심볼별 이동평균 지표 계산 (Polars 벡터 연산)"""
        
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
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성"""
        orders = []
        
        if not self._broker:
            return orders
        
        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')
        
        # 지표가 계산되지 않은 경우 건너뛰기
        if buy_sma is None or sell_sma is None:
            return orders
        
        current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 SMA15 상회 + 포지션 없음
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
                # print(f"📈 매수 신호: {symbol} @ {current_price:,.0f}원 (SMA{self.buy_sma}: {buy_sma:,.0f})")
        
        # 매도 신호: 가격이 SMA30 하회 + 포지션 있음
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
    end_date=datetime(2024, 12, 31), 
    timeframe="1d", 
    initial_cash=10_000_000,  # 1천만원
    commission_rate=0.0,      # 수수료 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
    slippage_rate=0.0,         # 슬리피지 0% (테스트용) - 실제 백테스팅에는 적절한 값 사용
    save_portfolio_history=True
)

# 3. Phase 7 하이브리드 SMA 전략
print("⚡ Phase 7 하이브리드 전략 초기화 중...")
strategy = MultiSymbolSMAStrategy(
    buy_sma=15,   # 매수: 가격이 15시간 이평선 상회
    sell_sma=30,   # 매도: 가격이 30시간 이평선 하회
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

# 7. 결과 출력
result = engine.run(config)
    
# 결과 요약 출력
result.print_summary()