"""
ThreadPoolExecutor를 사용한 병렬 그리드 서치 테스트
"""

import asyncio
import sys
import os
from datetime import datetime
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 시스템 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from quantbt.infrastructure.data.upbit_provider import UpbitDataProvider
from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.infrastructure.engine.simple_engine import SimpleBacktestEngine
from quantbt.examples.strategies.sma_grid_strategy import SMAGridStrategy
from quantbt.core.value_objects.backtest_config import BacktestConfig


def run_single_backtest_sync(params):
    """단일 백테스트 동기 실행 (스레드용)"""
    try:
        print(f"🔄 스레드 {threading.current_thread().name}: {params}")
        
        # 새로운 이벤트 루프 생성 (스레드용)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 전략 생성
            strategy = SMAGridStrategy(**params)
            
            # 브로커 생성
            broker = SimpleBroker(
                initial_cash=100000.0,
                commission_rate=0.001,
                slippage_rate=0.001
            )
            
            # 데이터 프로바이더 생성
            data_provider = UpbitDataProvider(cache_dir="./data/upbit_cache")
            
            # 엔진 생성 및 설정
            engine = SimpleBacktestEngine()
            engine.set_data_provider(data_provider)
            engine.set_strategy(strategy)
            engine.set_broker(broker)
            
            # 백테스트 설정
            config = BacktestConfig(
                symbols=["KRW-BTC"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                timeframe="1d",
                initial_cash=100000.0,
                save_portfolio_history=False
            )
            
            # 백테스트 실행
            result = loop.run_until_complete(engine.run(config))
            
            # 결과 요약
            summary = {
                'params': params,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'calmar_ratio': result._calculate_calmar_ratio(),
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'final_equity': result.final_equity
            }
            
            print(f"  ✅ 완료 - Calmar: {summary['calmar_ratio']:.3f}, "
                  f"수익률: {summary['total_return']*100:.2f}%, "
                  f"거래횟수: {summary['total_trades']}")
            
            return summary
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"  ❌ 실패 - 파라미터: {params}, 오류: {str(e)}")
        return None


def main():
    print("🚀 ThreadPoolExecutor 병렬 그리드 서치 테스트 시작")
    
    # 파라미터 그리드 설정
    buy_sma_values = [5, 10, 15]
    sell_sma_values = [10, 20, 30]
    
    # 유효한 조합만 생성 (buy_sma < sell_sma)
    parameter_combinations = []
    for buy_sma, sell_sma in product(buy_sma_values, sell_sma_values):
        if buy_sma < sell_sma:
            parameter_combinations.append({
                'buy_sma': buy_sma,
                'sell_sma': sell_sma
            })
    
    print(f"📋 총 {len(parameter_combinations)}개 조합 병렬 테스트")
    print("조합 목록:", parameter_combinations)
    
    # ThreadPoolExecutor로 병렬 실행
    results = []
    max_workers = min(4, len(parameter_combinations))
    
    print(f"🔧 {max_workers}개 스레드로 병렬 실행")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_params = {
            executor.submit(run_single_backtest_sync, params): params 
            for params in parameter_combinations
        }
        
        # 결과 수집
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 작업 실패 - {params}: {e}")
    
    # 결과 분석
    if results:
        print(f"\n🏆 병렬 그리드 서치 결과 분석 ({len(results)}개 성공)")
        
        # Calmar 비율 기준 정렬
        results.sort(key=lambda x: x['calmar_ratio'], reverse=True)
        
        print("\n📈 상위 5개 결과:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['params']} "
                  f"- Calmar: {result['calmar_ratio']:.3f}, "
                  f"수익률: {result['total_return']*100:.2f}%, "
                  f"MDD: {result['max_drawdown']*100:.2f}%")
        
        # 최적 파라미터
        best_result = results[0]
        print(f"\n🥇 최적 파라미터: {best_result['params']}")
        print(f"   📊 Calmar 비율: {best_result['calmar_ratio']:.3f}")
        print(f"   💰 총 수익률: {best_result['total_return']*100:.2f}%")
        print(f"   📉 최대낙폭: {best_result['max_drawdown']*100:.2f}%")
        print(f"   🎯 승률: {best_result['win_rate']*100:.1f}%")
        print(f"   🔄 거래횟수: {best_result['total_trades']}")
        
    else:
        print("❌ 성공한 백테스트가 없습니다.")
    
    print("\n🏁 병렬 그리드 서치 완료")


if __name__ == "__main__":
    main() 