# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path
import time
from datetime import datetime

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

from quantbt import (
    # Dict Native 전략 시스템
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
    """SMA 전략 - 기존 방식 (to_dicts 사용)
    
    하이브리드 방식:
    - 지표 계산: Polars 벡터연산
    - 신호 생성: Dict Native 방식
    
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
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성"""
        orders = []
        
        if not self.broker:
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


class OptimizedBacktestEngine(BacktestEngine):
    """iter_rows() 방식을 지원하는 최적화된 백테스트 엔진"""
    
    def _run_dict_native_backtest_loop_optimized(
        self, 
        config: BacktestConfig, 
        enriched_df: pl.DataFrame
    ) -> List[Dict[str, Any]]:
        """iter_rows() 방식을 사용한 최적화된 백테스트 루프"""
        
        trades = []
        
        # 미체결 주문 관리
        self.pending_orders = []
        
        # 포트폴리오 기록용 (부모 클래스 속성 사용)
        self._portfolio_equity_history = []
        
        print(f"📊 백테스팅 시작 (총 {len(enriched_df):,}개 캔들) - iter_rows() 방식")
        
        try:
            # iter_rows() 방식 사용
            for i, row_dict in enumerate(enriched_df.iter_rows(named=True)):
                
                # 0단계: 브로커에게 현재 시장 데이터 업데이트 
                try:
                    self._update_broker_market_data(row_dict)
                except Exception as e:
                    pass
                
                # 1단계: 이전 신호로 생성된 주문들 체결
                ready_orders = self._get_ready_orders(i)
                
                for pending_order in ready_orders:
                    trade_info = self._execute_pending_order(pending_order, row_dict)
                    if trade_info:
                        trades.append(trade_info)
                    
                
                # 2단계: 현재 캔들에서 신호 생성 (Dict 방식)
                try:
                    signals = self.strategy.generate_signals_dict(row_dict)
                    
                    # 3단계: 신호를 주문 대기열에 추가 (다음 캔들에서 체결)
                    for order in signals:
                        signal_price = row_dict['close']  # 신호 생성 시점 가격
                        self._add_order_to_queue(order, i, signal_price)
                        
                except Exception as e:
                    # 신호 생성 오류 시 해당 캔들 건너뛰고 계속 진행
                    continue
                
                # 4단계: 현재 시점 포트폴리오 평가금 계산 및 저장
                self._calculate_and_store_portfolio_equity(row_dict, config)
                
                # 진행 상황 출력 (10% 단위)
                if (i + 1) % (len(enriched_df) // 10) == 0:
                    progress = ((i + 1) / len(enriched_df)) * 100
                    print(f"   진행률: {progress:.0f}%")
        
        except Exception as e:
            print(f"❌ 백테스팅 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print("✅ iter_rows() 백테스팅 완료")
        return trades
    
    def run_optimized(self, config: BacktestConfig) -> 'BacktestResult':
        """iter_rows() 방식을 사용한 최적화된 백테스팅 실행"""
        if not self.strategy:
            raise ValueError("전략이 설정되지 않았습니다")
        if not self.broker:
            raise ValueError("브로커가 설정되지 않았습니다")
        if not self.data_provider:
            raise ValueError("데이터 제공자가 설정되지 않았습니다")
        
        total_start_time = datetime.now()

        # 0단계: 브로커 초기화 (중요!)
        self.broker.portfolio.cash = config.initial_cash
        self.broker.portfolio.positions = {}
        self.broker.orders = {}
        self.broker.trades = []

        # 1단계: 원본 데이터 로딩 (시간 측정 별도)
        data_load_start = time.time()
        raw_data_df = self._load_raw_data_as_polars(config)
        data_load_time = time.time() - data_load_start
        
        # 2단계: 지표 계산 (시간 측정)
        indicator_start = time.time()
        enriched_df = self.strategy.precompute_indicators(raw_data_df)
        indicator_time = time.time() - indicator_start
        
        # 3단계: 최적화된 백테스팅 루프 실행 (iter_rows 사용)
        backtest_start = time.time()
        trades = self._run_dict_native_backtest_loop_optimized(config, enriched_df)
        backtest_time = time.time() - backtest_start
        
        # 4단계: 결과 생성
        result_start = time.time()
        end_time = datetime.now()
        result = self._create_result_from_dict(config, total_start_time, end_time, trades)
        result_time = time.time() - result_start
        
        # 성능 정보 출력
        print("\n🕒 성능 분석 (iter_rows 방식):")
        print(f"   - 데이터 로딩: {data_load_time:.4f}초")
        print(f"   - 지표 계산: {indicator_time:.4f}초")
        print(f"   - 백테스팅 루프: {backtest_time:.4f}초")
        print(f"   - 결과 생성: {result_time:.4f}초")
        print(f"   - 총 소요 시간: {(datetime.now() - total_start_time).total_seconds():.4f}초")
        
        return result


def run_performance_comparison(period_name: str, config: BacktestConfig):
    """성능 비교 실행 함수"""
    print(f"\n{'='*60}")
    print(f"🔥 성능 비교 테스트: {period_name}")
    print(f"{'='*60}")
    
    # 데이터 프로바이더 초기화
    upbit_provider = UpbitDataProvider()
    
    # 전략 초기화
    strategy = SimpleSMAStrategy(buy_sma=15, sell_sma=30)
    
    # 브로커 설정
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )
    
    # ============== 기존 방식 (to_dicts) ==============
    print("\n📊 [방식 1] 기존 to_dicts() 방식 백테스팅...")
    
    # 기존 엔진
    engine_original = BacktestEngine()
    engine_original.set_strategy(strategy)
    engine_original.set_data_provider(upbit_provider)
    engine_original.set_broker(broker)
    
    start_time_original = time.time()
    result_original = engine_original.run(config)
    time_original = time.time() - start_time_original
    
    # ============== 최적화 방식 (iter_rows) ==============
    print("\n⚡ [방식 2] 최적화된 iter_rows() 방식 백테스팅...")
    
    # 브로커 재초기화 (상태 리셋)
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )
    
    # 최적화 엔진
    engine_optimized = OptimizedBacktestEngine()
    engine_optimized.set_strategy(strategy)
    engine_optimized.set_data_provider(upbit_provider)
    engine_optimized.set_broker(broker)
    
    start_time_optimized = time.time()
    result_optimized = engine_optimized.run_optimized(config)
    time_optimized = time.time() - start_time_optimized
    
    # ============== 성능 비교 결과 ==============
    print(f"\n📈 성능 비교 결과 ({period_name}):")
    print(f"   - 기존 방식 (to_dicts): {time_original:.4f}초")
    print(f"   - 최적화 방식 (iter_rows): {time_optimized:.4f}초")
    print(f"   - 속도 개선: {((time_original - time_optimized) / time_original * 100):+.1f}%")
    
    if time_optimized < time_original:
        speedup = time_original / time_optimized
        print(f"   - 최적화 방식이 {speedup:.1f}배 빠름")
    else:
        slowdown = time_optimized / time_original
        print(f"   - 기존 방식이 {slowdown:.1f}배 빠름")
    
    # 백테스팅 결과 요약 (원본만 출력)
    print(f"\n💰 백테스팅 결과 요약 ({period_name}):")
    result_original.print_summary()
    
    return time_original, time_optimized


def main():
    """메인 실행 함수"""
    print("🚀 SMA 전략 성능 비교 테스트 시작")
    print("   - 기존 방식: precompute → to_dicts() → for loop")
    print("   - 최적화 방식: precompute → iter_rows() → for loop")
    
    # 테스트 기간 설정 (2024년 내 1분봉)
    test_periods = [
        ("3개월", datetime(2024, 1, 1), datetime(2024, 4, 1)),
        ("6개월", datetime(2024, 1, 1), datetime(2024, 7, 1)),
        ("1년", datetime(2024, 1, 1), datetime(2024, 12, 31))
    ]
    
    results = []
    
    for period_name, start_date, end_date in test_periods:
        # 백테스팅 설정 
        config = BacktestConfig(
            symbols=["KRW-BTC"],
            start_date=start_date,
            end_date=end_date, 
            timeframe="1m",  # 1분봉으로 변경
            initial_cash=10_000_000,  # 1천만원
            commission_rate=0.0,      # 수수료 0% (테스트용)
            slippage_rate=0.0,        # 슬리피지 0% (테스트용)
            save_portfolio_history=True
        )
        
        # 성능 비교 실행
        time_original, time_optimized = run_performance_comparison(period_name, config)
        results.append((period_name, time_original, time_optimized))
    
    # ============== 최종 성능 요약 ==============
    print(f"\n{'='*80}")
    print("🏆 최종 성능 비교 요약")
    print(f"{'='*80}")
    print(f"{'기간':<10} {'기존방식(초)':<15} {'최적화방식(초)':<15} {'개선율(%)':<12} {'배수':<8}")
    print("-" * 80)
    
    for period_name, time_original, time_optimized in results:
        improvement = ((time_original - time_optimized) / time_original * 100)
        multiplier = time_original / time_optimized if time_optimized < time_original else -(time_optimized / time_original)
        
        print(f"{period_name:<10} {time_original:<15.4f} {time_optimized:<15.4f} {improvement:<12.1f} {multiplier:<8.1f}")
    
    print("-" * 80)
    
    # 총합 계산
    total_original = sum(result[1] for result in results)
    total_optimized = sum(result[2] for result in results)
    total_improvement = ((total_original - total_optimized) / total_original * 100)
    
    print(f"{'총합':<10} {total_original:<15.4f} {total_optimized:<15.4f} {total_improvement:<12.1f}")
    print(f"\n⚡ 결론: ")
    if total_optimized < total_original:
        print(f"   - iter_rows() 방식이 평균 {total_improvement:.1f}% 빠름")
        print(f"   - 전체적으로 {total_original / total_optimized:.1f}배 성능 향상")
    else:
        print(f"   - to_dicts() 방식이 평균 {abs(total_improvement):.1f}% 빠름")
        print(f"   - iter_rows() 방식이 {total_optimized / total_original:.1f}배 느림")


if __name__ == "__main__":
    main() 