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


class SimpleSMAStrategy(TradingStrategy):
    """SMA 전략 - 포트폴리오 추적용"""
    
    def __init__(self, buy_sma: int = 15, sell_sma: int = 30):
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
            quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
            
            if quantity > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                orders.append(order)
        
        # 매도 신호: 가격이 SMA30 하회 + 포지션 있음
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_positions[symbol],
                order_type=OrderType.MARKET
            )
            orders.append(order)
        
        return orders


class PortfolioTracker:
    """포트폴리오 변화 추적기"""
    
    def __init__(self):
        self.snapshots = []
        self.trade_count = 0
        
    def take_snapshot(self, timestamp, portfolio, context="", trade_info=None):
        """포트폴리오 스냅샷 저장"""
        snapshot = {
            'timestamp': timestamp,
            'context': context,
            'cash': portfolio.cash,
            'positions': {},
            'total_equity': portfolio.equity,
            'trade_info': trade_info
        }
        
        for symbol, position in portfolio.positions.items():
            if position.quantity != 0:
                snapshot['positions'][symbol] = {
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'market_price': position.market_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl
                }
        
        self.snapshots.append(snapshot)
        
        # 급격한 변화 감지
        if len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            current = snapshot
            
            if prev['total_equity'] > 0:
                equity_change = (current['total_equity'] - prev['total_equity']) / prev['total_equity']
                if abs(equity_change) > 0.1:  # 10% 이상 변화
                    print(f"⚠️ 급격한 포트폴리오 변화 감지: {equity_change:.2%}")
                    print(f"  시점: {timestamp}")
                    print(f"  컨텍스트: {context}")
                    print(f"  이전 equity: {prev['total_equity']:,.0f}")
                    print(f"  현재 equity: {current['total_equity']:,.0f}")
                    if trade_info:
                        print(f"  거래: {trade_info}")
    
    def validate_equity_calculation(self, portfolio, context=""):
        """Equity 계산 검증"""
        # 1. 브로커가 제공하는 equity
        broker_equity = portfolio.equity
        
        # 2. 수동 계산
        manual_cash = portfolio.cash
        manual_position_value = 0.0
        
        print(f"\n=== Equity 검증: {context} ===")
        print(f"현금: {manual_cash:,.0f}")
        
        position_details = []
        for symbol, position in portfolio.positions.items():
            if position.quantity > 0:
                position_value = position.quantity * position.market_price
                manual_position_value += position_value
                
                detail = {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'market_price': position.market_price,
                    'market_value': position_value,
                    'avg_price': position.avg_price,
                    'unrealized_pnl': position.unrealized_pnl
                }
                position_details.append(detail)
                
                print(f"{symbol} 포지션:")
                print(f"  수량: {position.quantity:.6f}")
                print(f"  Market Price: {position.market_price:,.0f}")
                print(f"  Market Value: {position_value:,.0f}")
                print(f"  Avg Price: {position.avg_price:,.0f}")
                print(f"  Unrealized PnL: {position.unrealized_pnl:,.0f}")
        
        manual_total_equity = manual_cash + manual_position_value
        
        print(f"총 포지션 가치: {manual_position_value:,.0f}")
        print(f"브로커 Equity: {broker_equity:,.0f}")
        print(f"수동 Equity: {manual_total_equity:,.0f}")
        print(f"차이: {broker_equity - manual_total_equity:,.0f}")
        
        # 1% 이상 차이나면 경고
        is_valid = True
        if manual_total_equity > 0:
            diff_pct = abs(broker_equity - manual_total_equity) / manual_total_equity
            if diff_pct > 0.01:
                print(f"⚠️ Equity 계산 불일치! 차이: {diff_pct:.2%}")
                is_valid = False
        
        return is_valid, {
            'broker_equity': broker_equity,
            'manual_equity': manual_total_equity,
            'cash': manual_cash,
            'position_value': manual_position_value,
            'positions': position_details
        }
    
    def print_summary(self):
        """추적 요약 출력"""
        if not self.snapshots:
            print("추적된 스냅샷이 없습니다.")
            return
        
        initial = self.snapshots[0]
        final = self.snapshots[-1]
        
        print(f"\n=== 포트폴리오 추적 요약 ===")
        print(f"초기 Equity: {initial['total_equity']:,.0f}")
        print(f"최종 Equity: {final['total_equity']:,.0f}")
        print(f"총 수익률: {(final['total_equity'] - initial['total_equity']) / initial['total_equity']:.2%}")
        print(f"총 스냅샷 수: {len(self.snapshots)}")
        
        # 최대/최소 equity
        max_equity = max(s['total_equity'] for s in self.snapshots)
        min_equity = min(s['total_equity'] for s in self.snapshots)
        print(f"최대 Equity: {max_equity:,.0f}")
        print(f"최소 Equity: {min_equity:,.0f}")


def test_portfolio_equity_calculation():
    """포트폴리오 평가금 계산 검증 테스트"""
    
    print("🔄 포트폴리오 평가금 계산 검증 테스트 시작...")
    
    # 1. 업비트 데이터 프로바이더
    upbit_provider = UpbitDataProvider()
    
    # 2. 백테스팅 설정
    config = BacktestConfig(
        symbols=["KRW-BTC", "KRW-ETH"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),  # 짧은 기간으로 테스트
        timeframe="1m",
        initial_cash=10_000_000,
        commission_rate=0.0,
        slippage_rate=0.0,
        save_portfolio_history=True
    )
    
    # 3. 전략
    strategy = SimpleSMAStrategy(buy_sma=15, sell_sma=30)
    
    # 4. 브로커
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )
    
    # 5. 백테스트 엔진
    engine = BacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(upbit_provider)
    engine.set_broker(broker)
    
    # 6. 포트폴리오 추적기 설정
    tracker = PortfolioTracker()
    
    # 원래 메서드 백업
    original_execute_pending_order = engine._execute_pending_order
    original_calculate_equity = engine._calculate_and_store_portfolio_equity
    
    def enhanced_execute_pending_order(pending_order, current_candle):
        # 거래 전 스냅샷
        tracker.take_snapshot(
            current_candle['timestamp'], 
            engine.broker.get_portfolio(), 
            "거래 전"
        )
        
        # 원래 로직 실행
        result = original_execute_pending_order(pending_order, current_candle)
        
        # 거래 후 스냅샷
        if result:
            tracker.take_snapshot(
                current_candle['timestamp'], 
                engine.broker.get_portfolio(), 
                "거래 후",
                result
            )
            
            # Equity 계산 검증
            is_valid, details = tracker.validate_equity_calculation(
                engine.broker.get_portfolio(), 
                f"거래 후 - {result['symbol']} {result['side']}"
            )
            
            if not is_valid:
                print(f"⚠️ 거래 후 Equity 계산 오류 발견!")
                return None  # 테스트 중단
        
        return result
    
    def enhanced_calculate_equity(current_candle, config):
        # 원래 로직 실행
        original_calculate_equity(current_candle, config)
        
        # 매 캔들마다 Equity 검증 (일부만)
        if tracker.trade_count % 100 == 0:  # 100번마다 검증
            is_valid, details = tracker.validate_equity_calculation(
                engine.broker.get_portfolio(), 
                f"캔들 처리 - {current_candle['symbol']} @ {current_candle['timestamp']}"
            )
    
    # 메서드 패치
    engine._execute_pending_order = enhanced_execute_pending_order
    engine._calculate_and_store_portfolio_equity = enhanced_calculate_equity
    
    try:
        # 7. 백테스트 실행
        print("백테스트 실행 중...")
        result = engine.run(config)
        
        # 8. 최종 결과 검증
        print("\n=== 최종 결과 검증 ===")
        final_portfolio = engine.broker.get_portfolio()
        is_valid, final_details = tracker.validate_equity_calculation(
            final_portfolio, 
            "최종 결과"
        )
        
        print(f"\n최종 Equity: {result.final_equity:,.0f}")
        print(f"검증된 Equity: {final_details['manual_equity']:,.0f}")
        print(f"차이: {result.final_equity - final_details['manual_equity']:,.0f}")
        
        # 9. 추적 요약
        tracker.print_summary()
        
        # 10. 오버플로우 테스트
        try:
            duration_years = (config.end_date - config.start_date).days / 365.25
            annual_return = ((result.final_equity / config.initial_cash) ** (1 / duration_years) - 1)
            print(f"\n연간 수익률 계산 성공: {annual_return:.2%}")
            print(f"기간: {duration_years:.3f}년")
        except OverflowError as e:
            print(f"\n❌ 연간 수익률 오버플로우: {e}")
            print(f"Final Equity: {result.final_equity:,.0f}")
            print(f"Initial Cash: {config.initial_cash:,.0f}")
            print(f"비율: {result.final_equity / config.initial_cash:.1f}배")
            
        return result, tracker
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, tracker


if __name__ == "__main__":
    result, tracker = test_portfolio_equity_calculation() 