# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path
import asyncio
import time
from contextlib import contextmanager
from collections import defaultdict

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
from datetime import datetime, timedelta

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

# 🔍 성능 측정 클래스 (루프 내부 측정용)
class LoopPerformanceTimer:
    def __init__(self):
        self.timings = defaultdict(float)  # 누적 시간 저장
        self.counts = defaultdict(int)     # 호출 횟수 저장
    
    @contextmanager
    def time_block(self, name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[name] += duration
            self.counts[name] += 1
    
    def get_summary(self, test_name: str):
        total_time = sum(self.timings.values())
        
        print(f"\n📊 {test_name} 루프 내부 성능 측정 결과")
        print("=" * 80)
        
        sorted_timings = sorted(
            [(name, total_time_task, self.counts[name]) for name, total_time_task in self.timings.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"{'순위':3s} | {'로직':25s} | {'누적시간':12s} | {'비율':8s} | {'호출수':10s} | {'평균':12s}")
        print("-" * 80)
        
        for i, (name, total_time_task, count) in enumerate(sorted_timings, 1):
            percentage = (total_time_task / total_time) * 100 if total_time > 0 else 0
            avg_time = total_time_task / count if count > 0 else 0
            
            print(f"{i:2d}. | {name:25s} | {total_time_task:10.6f}초 | {percentage:6.2f}% | {count:8d}회 | {avg_time*1000:8.4f}ms")
        
        print("-" * 80)
        print(f"{'':5s} {'총합':25s} | {total_time:10.6f}초 | 100.00%")
        print()
        
        return {
            'total_time': total_time,
            'timings': dict(self.timings),
            'counts': dict(self.counts),
            'sorted_timings': sorted_timings
        }
    
    def reset(self):
        self.timings.clear()
        self.counts.clear()

# 전역 성능 측정기
loop_perf_timer = LoopPerformanceTimer()

class SimpleSMAStrategy(TradingStrategy):
    """Phase 7 하이브리드 SMA 전략 (루프 내부 측정용)
    
    하이브리드 방식:
    - 지표 계산: Polars 벡터연산 (고성능)
    - 신호 생성: Dict Native 방식 (최고 성능)
    
    매수: 가격이 SMA15 상회
    매도: 가격이 SMA30 하회  
    """
    
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
    
    def generate_signals_dict(self, current_data: Dict[str, Any], 
                            historical_data: Optional[List[Dict[str, Any]]] = None) -> List[Order]:
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
        
        with loop_perf_timer.time_block("신호_포지션조회"):
            current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 SMA15 상회 + 포지션 없음
        if current_price > buy_sma and symbol not in current_positions:
            with loop_perf_timer.time_block("신호_포트폴리오조회"):
                portfolio_value = self.get_portfolio_value()
            
            with loop_perf_timer.time_block("신호_포지션사이즈계산"):
                quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
            
            if quantity > 0:
                with loop_perf_timer.time_block("신호_주문생성"):
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
        
        # 매도 신호: 가격이 SMA30 하회 + 포지션 있음
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            with loop_perf_timer.time_block("신호_주문생성"):
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                )
                orders.append(order)
        
        return orders


# 🔧 백테스팅 엔진에 성능 측정 추가
class PerformanceMeasuredBacktestEngine(BacktestEngine):
    """성능 측정이 추가된 백테스팅 엔진"""
    
    async def _run_dict_native_backtest_loop(self, config: BacktestConfig, 
                                           enriched_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dict Native 백테스팅 루프 - 루프 내부 성능 측정"""
        trades = []
        self.pending_orders = []  # 주문 대기열 초기화
        
        # 실시간 포트폴리오 평가금 추적 초기화
        self._portfolio_equity_history = {}  # {timestamp: equity}
        
        # pending_orders 사이즈 추적용 변수
        total_candles = len(enriched_data)
        check_points = [
            int(total_candles * 0.1),   # 10% 지점 (초반부)
            int(total_candles * 0.5),   # 50% 지점 (중반부)
            int(total_candles * 0.9),   # 90% 지점 (후반부)
        ]
        
        # tqdm 프로그레스바 생성 (save_portfolio_history=True일 때만)
        pbar = None
        if config.save_portfolio_history:
            pbar = self.create_progress_bar(len(enriched_data), "백테스팅 진행")
        
        try:
            # Dict Native 루프: 각 단계별 성능 측정
            for i, current_candle in enumerate(enriched_data):
                
                # 🔍 pending_orders 사이즈 체크 포인트
                if i in check_points:
                    progress_pct = (i / total_candles) * 100
                    pending_count = len(self.pending_orders)
                    active_pending_count = len([p for p in self.pending_orders if p['status'] == 'PENDING'])
                    filled_count = len([p for p in self.pending_orders if p['status'] == 'FILLED'])
                    failed_count = len([p for p in self.pending_orders if p['status'] == 'FAILED'])
                    
                    print(f"\n📊 진행률 {progress_pct:4.1f}% (캔들 {i+1:,}/{total_candles:,})")
                    print(f"   📋 pending_orders 총 크기: {pending_count:,}개")
                    print(f"   🟡 PENDING 주문: {active_pending_count:,}개")
                    print(f"   🟢 FILLED 주문: {filled_count:,}개")
                    print(f"   🔴 FAILED 주문: {failed_count:,}개")
                
                # 0단계: 브로커 시장 데이터 업데이트
                with loop_perf_timer.time_block("루프_브로커업데이트"):
                    try:
                        self._update_broker_market_data(current_candle)
                    except Exception as e:
                        pass
                
                # 1단계: 대기 중인 주문 조회
                with loop_perf_timer.time_block("루프_주문조회"):
                    ready_orders = self._get_ready_orders(i)
                
                # 2단계: 주문 체결 처리
                with loop_perf_timer.time_block("루프_주문체결"):
                    for pending_order in ready_orders:
                        trade_info = self._execute_pending_order(pending_order, current_candle)
                        if trade_info:
                            trades.append(trade_info)
                
                # 3단계: 신호 생성 (전략 호출)
                with loop_perf_timer.time_block("루프_신호생성"):
                    try:
                        # Phase 7 하이브리드 전략 지원
                        if isinstance(self.strategy, TradingStrategy):
                            signals = self.strategy.generate_signals_dict(current_candle)
                        else:
                            signals = self.strategy.generate_signals_dict(current_candle)
                    except Exception as e:
                        signals = []
                        import traceback
                        traceback.print_exc()
                
                # 4단계: 신호를 주문 대기열에 추가
                with loop_perf_timer.time_block("루프_주문대기열추가"):
                    for order in signals:
                        signal_price = current_candle['close']
                        self._add_order_to_queue(order, i, signal_price)
                
                # 5단계: 포트폴리오 평가금 계산
                with loop_perf_timer.time_block("루프_포트폴리오평가"):
                    self._calculate_and_store_portfolio_equity(current_candle, config)
                
                # 6단계: 프로그레스바 업데이트
                with loop_perf_timer.time_block("루프_진행률업데이트"):
                    if pbar is not None:
                        timestamp = current_candle.get('timestamp', 'N/A')
                        self.update_progress_bar(pbar, f"처리중... {i+1}/{len(enriched_data)} ({timestamp})")
        
        finally:
            # 프로그레스바 정리 (생성된 경우에만)
            if pbar is not None:
                pbar.close()
        
        # 마지막 캔들에서 남은 주문들 처리
        with loop_perf_timer.time_block("루프_마지막주문처리"):
            if enriched_data:
                last_candle = enriched_data[-1]
                final_ready_orders = self._get_ready_orders(len(enriched_data))
                for pending_order in final_ready_orders:
                    trade_info = self._execute_pending_order(pending_order, last_candle)
                    if trade_info:
                        trades.append(trade_info)

        # 최종 pending_orders 상태 출력
        final_pending_count = len(self.pending_orders)
        final_filled_count = len([p for p in self.pending_orders if p['status'] == 'FILLED'])
        final_failed_count = len([p for p in self.pending_orders if p['status'] == 'FAILED'])
        final_active_count = len([p for p in self.pending_orders if p['status'] == 'PENDING'])
        
        print(f"\n📊 최종 pending_orders 상태:")
        print(f"   📋 총 크기: {final_pending_count:,}개")
        print(f"   🟢 FILLED: {final_filled_count:,}개")  
        print(f"   🔴 FAILED: {final_failed_count:,}개")
        print(f"   🟡 PENDING: {final_active_count:,}개")

        return trades


def run_performance_test(days: int, test_name: str):
    """성능 테스트 실행 함수 (루프 내부 측정)"""
    print(f"\n🚀 {test_name} 백테스팅 시작...")
    
    # 성능 측정기 리셋
    loop_perf_timer.reset()
    
    # 1. 업비트 데이터 프로바이더
    upbit_provider = UpbitDataProvider()

    # 2. 백테스팅 설정
    end_date = datetime(2024, 3, 31)
    start_date = end_date - timedelta(days=days-1)
    
    config = BacktestConfig(
        symbols=["KRW-BTC"],
        start_date=start_date,
        end_date=end_date, 
        timeframe="1m", 
        initial_cash=10_000_000,  # 1천만원
        commission_rate=0.0,      # 수수료 0% (테스트용)
        slippage_rate=0.0,         # 슬리피지 0% (테스트용)
        save_portfolio_history=True
    )

    # 3. Phase 7 하이브리드 SMA 전략
    strategy = SimpleSMAStrategy(
        buy_sma=15,   # 매수: 가격이 15시간 이평선 상회
        sell_sma=30   # 매도: 가격이 30시간 이평선 하회
    )

    # 4. 브로커 설정
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )

    # 5. 성능 측정이 추가된 백테스트 엔진
    engine = PerformanceMeasuredBacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(upbit_provider)
    engine.set_broker(broker)

    try:
        result = asyncio.run(engine.run(config))
        
        # 7. 결과 출력   
        print(f"✅ {test_name} 백테스팅 완료!")
        result.print_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ 백테스팅 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # HTTP 세션 정리 (aiohttp 경고 해결)
        if hasattr(upbit_provider, '_session') and upbit_provider._session and not upbit_provider._session.closed:
            asyncio.run(upbit_provider._session.close())


def main():
    """메인 실행 함수 - 60일 테스트만 (pending_orders 검증용)"""
    
    test_configs = [
        (60, "60일 테스트 (pending_orders 검증)")
    ]
    
    results = {}
    
    print("=" * 80)
    print("🔍 pending_orders 리스트 크기 검증 테스트")
    print("=" * 80)
    
    for days, test_name in test_configs:
        start_time = time.perf_counter()
        
        success = run_performance_test(days, test_name)
        
        if success:
            end_time = time.perf_counter()
            total_test_time = end_time - start_time
            
            # 성능 결과 수집
            summary = loop_perf_timer.get_summary(test_name)
            summary['total_test_time'] = total_test_time
            summary['days'] = days
            results[test_name] = summary
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()