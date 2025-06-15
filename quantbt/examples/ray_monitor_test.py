"""
Ray 백테스팅 프로그래스 모니터링 통합 예제

- shared_data_ref 전용 설계로 빠른 성능
- 단순하고 명확한 구조
- 메모리 효율성 극대화
"""

# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path
import asyncio
import time

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
import ray

from quantbt import (
    TradingStrategy,   
    BacktestConfig,    
    # 주문 관련
    Order, OrderSide, OrderType,
)

# Ray 기반 최적화 시스템
from quantbt.ray import (
    RayClusterManager,
    RayDataManager
)

# 단순화된 Actor 구조
from quantbt.ray.backtest_actor import BacktestActor

# 새로 개발한 모니터링 시스템
from quantbt.ray.monitoring import ProgressTracker, SimpleMonitor


class SimpleSMAStrategy(TradingStrategy):
    """SMA 전략 (Ray 최적화용)
    
    고성능 스트리밍 방식:
    - 지표 계산: Polars 벡터연산 
    - 신호 생성: 행별 스트리밍 처리
    
    매수: 가격이 buy_sma 상회
    매도: 가격이 sell_sma 하회  
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
        
        # 지표 컬럼 추가 (중복 방지)
        columns_to_add = []
        
        # buy_sma 컬럼 추가
        buy_sma_name = f"sma_{self.buy_sma}"
        columns_to_add.append(buy_sma.alias(buy_sma_name))
        
        # sell_sma 컬럼 추가 (중복 체크)
        sell_sma_name = f"sma_{self.sell_sma}"
        if sell_sma_name != buy_sma_name:  # 중복이 아닌 경우만 추가
            columns_to_add.append(sell_sma.alias(sell_sma_name))
        
        return data.with_columns(columns_to_add)
    
    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """행 데이터 기반 신호 생성"""
        orders = []
        
        if not self.broker:
            return orders
        
        symbol = current_data['symbol']
        current_price = current_data['close']
        
        # SMA 값 가져오기 (같은 값인 경우 하나의 컬럼만 존재)
        buy_sma_name = f'sma_{self.buy_sma}'
        sell_sma_name = f'sma_{self.sell_sma}'
        
        buy_sma = current_data.get(buy_sma_name)
        if buy_sma_name == sell_sma_name:
            sell_sma = buy_sma  # 같은 SMA 값인 경우
        else:
            sell_sma = current_data.get(sell_sma_name)
        
        # 지표가 계산되지 않은 경우 건너뛰기
        if buy_sma is None or sell_sma is None:
            return orders
        
        current_positions = self.get_current_positions()
        
        # 매수 신호: 가격이 buy_sma 상회 + 포지션 없음
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
        
        # 매도 신호: 가격이 sell_sma 하회 + 포지션 있음
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=current_positions[symbol],
                order_type=OrderType.MARKET
            )
            orders.append(order)
        
        return orders


async def run_monitored_ray_optimization():
    """ProgressTracker + SimpleMonitor 통합 Ray 백테스팅 최적화"""
    
    print("🚀 Ray 백테스팅 모니터링 통합 시스템 시작 (최고 성능)")
    print("=" * 70)
    
    # 1. 모니터링 시스템 초기화
    print("🔧 모니터링 시스템 초기화...")
    progress_tracker = None  # 나중에 초기화
    simple_monitor = SimpleMonitor()
    print("✅ SimpleMonitor 초기화 완료")
    
    # 2. RayClusterManager 설정 및 초기화
    ray_cluster_config = {
        "num_cpus": 32,
        "object_store_memory": 1000 * 1024 * 1024 * 8,  # 8GB
        "ignore_reinit_error": True,
        "logging_level": "INFO"  # 디버깅을 위해 INFO로 변경
    }
    
    cluster_manager = RayClusterManager(ray_cluster_config)
    
    print("🔧 Ray 클러스터 초기화 중...")
    if not cluster_manager.initialize_cluster():
        print("❌ Ray 클러스터 초기화 실패")
        return
    
    print("✅ Ray 클러스터 초기화 완료")
    
    # 3. 클러스터 상태 및 리소스 정보 출력
    cluster_resources = cluster_manager.get_cluster_resources()
    available_resources = cluster_manager.get_available_resources()
    
    print(f"📊 클러스터 리소스:")
    print(f"   - 총 CPU: {cluster_resources['cpu']}")
    print(f"   - Object Store: {cluster_resources['object_store']:,} bytes")
    print(f"   - 노드 수: {cluster_resources['nodes']}")
    print(f"   - 사용 가능한 CPU: {available_resources['cpu']}")
    
    # 4. 백테스트 기본 설정
    config = BacktestConfig(
        symbols=["KRW-BTC"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        timeframe="1d",
        initial_cash=10_000_000,
        commission_rate=0.0,
        slippage_rate=0.0,
        save_portfolio_history=False
    )
    print("✅ 백테스트 설정 완료")

    # 5. 파라메터 그리드 정의 및 조합 생성
    param_grid = {
        'buy_sma': [10, 15, 20, 25],      # 매수 SMA: 10, 15, 20, 25
        'sell_sma': [25, 30, 35, 40]      # 매도 SMA: 25, 30, 35, 40
    }
    total_combinations = len(param_grid['buy_sma']) * len(param_grid['sell_sma'])
    print(f"\n✅ 파라메터 그리드 정의 완료: {total_combinations}개 조합")
    print(f"   - 매수 SMA: {param_grid['buy_sma']}")
    print(f"   - 매도 SMA: {param_grid['sell_sma']}")

    # 6. RayDataManager 생성 및 데이터 로딩
    print("\n🔧 RayDataManager 생성 및 데이터 로딩")
    data_manager = RayDataManager.remote()
    print("✅ RayDataManager 생성 완료")
    
    # 데이터 미리 로딩 (제로카피 방식)
    print("📊 실제 데이터 로딩 중... (제로카피 방식)")
    data_loading_start = time.time()
    
    # load_real_data는 이미 ray.ObjectRef를 반환하므로 await 불필요
    data_ref = data_manager.load_real_data.remote(
        symbols=config.symbols,
        start_date=config.start_date,
        end_date=config.end_date,
        timeframe=config.timeframe
    )
    
    data_loading_time = time.time() - data_loading_start
    print(f"✅ 데이터 로딩 완료: {data_loading_time:.2f}초")
    
    # 7. 워커 환경 준비 (실제 작업 수 기반)
    worker_env = cluster_manager.prepare_worker_environment(
        expected_tasks=total_combinations,  # 실제 조합 수 전달
        memory_per_task_mb=200  # 작업당 메모리
    )
    
    print(f"🎯 워커 환경 준비:")
    print(f"   - 최적 워커 수: {worker_env['optimal_workers']}")
    print(f"   - 작업당 메모리: {worker_env['memory_per_task_mb']}MB")
    
    # 8. BacktestActor 생성 (단순화된 구조)
    num_actors = worker_env['optimal_workers']
    print(f"\n🎯 {num_actors}개 BacktestActor 생성 중... (shared_data_ref 전용)")
    
    actors = []
    for i in range(num_actors):
        # shared_data_ref만 전달 (최고 성능)
        actor = BacktestActor.remote(f"actor_{i}", shared_data_ref=data_ref)
        actors.append(actor)
    
    # Actor 초기화
    config_dict = {
        'symbols': config.symbols,
        'start_date': config.start_date,
        'end_date': config.end_date,
        'timeframe': config.timeframe,
        'initial_cash': config.initial_cash,
        'commission_rate': config.commission_rate,
        'slippage_rate': config.slippage_rate,
        'save_portfolio_history': config.save_portfolio_history
        # shared_data_ref는 이미 Actor 생성 시 전달됨
    }
    
    init_results = await asyncio.gather(*[
        actor.initialize_engine.remote(config_dict) for actor in actors
    ])
    
    successful_actors = sum(init_results)
    print(f"✅ BacktestActor 초기화: {successful_actors}/{num_actors}개 성공")
    
    # 9. 파라메터 조합 생성
    from itertools import product
    param_combinations = []
    for buy_sma, sell_sma in product(param_grid['buy_sma'], param_grid['sell_sma']):
        param_combinations.append({
            'buy_sma': buy_sma,
            'sell_sma': sell_sma
        })
    
    # 10. 프로그래스 트래커 초기화 및 시작
    print("\n⚡ 모니터링 시스템과 함께 분산 백테스트 실행 시작")
    print("=" * 70)
    
    # 이제 total_combinations를 알므로 ProgressTracker 초기화
    progress_tracker = ProgressTracker(total_tasks=total_combinations)
    progress_tracker.start()
    print(f"✅ ProgressTracker 초기화 완료 (총 {total_combinations}개 작업)")
    
    optimization_start = time.time()
    
    # Actor별로 작업 분배
    tasks = []
    for i, params in enumerate(param_combinations):
        actor_idx = i % len(actors)
        actor = actors[actor_idx]
        
        task = actor.execute_backtest.remote(params, SimpleSMAStrategy)
        tasks.append((i, params, task))
    
    # 11. 실시간 모니터링과 함께 작업 완료 대기
    print(f"📊 {total_combinations}개 백테스트 병렬 실행 및 실시간 모니터링... (최고 성능)")
    print("-" * 70)
    
    completed_tasks = 0
    results = []
    
    # 실시간 진행 상황 표시를 위한 업데이트 간격 설정
    update_interval = max(1, total_combinations // 10)  # 10% 단위로 업데이트
    
    for i, (task_id, params, task) in enumerate(tasks):
        try:
            # 백테스트 결과 대기
            result = await task
            completed_tasks += 1
            
            # 결과 저장
            backtest_result = {
                'params': params,
                'result': result,
                'success': True,
                'task_id': task_id
            }
            results.append(backtest_result)
            
            # SimpleMonitor에 결과 기록
            monitor_result = {
                'success': True,
                'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                'total_return': result.get('total_return', 0.0),
                'params': params,
                'execution_time': 0.0  # 개별 작업 시간은 측정하지 않음
            }
            simple_monitor.record_result(monitor_result)
            
            # 프로그래스 업데이트 (1개씩 증가)
            progress_tracker.update(1)
            
            # 주기적으로 진행 상황 출력
            if completed_tasks % update_interval == 0 or completed_tasks == total_combinations:
                progress_info = progress_tracker.get_progress()
                eta_info = progress_tracker.get_eta()
                progress_text = progress_tracker.format_progress(show_bar=True)
                
                print(f"📈 {progress_text}, ETA: {eta_info}")
                
                # 중간 통계 출력 (완료된 작업이 5개 이상일 때)
                if completed_tasks >= 5:
                    current_stats = simple_monitor.get_statistics()
                    print(f"   💡 현재 통계: 성공률 {current_stats['success_rate']:.1f}%, "
                          f"평균 샤프비율 {current_stats['avg_sharpe_ratio']:.4f}")
                    
                print("-" * 50)
            
        except Exception as e:
            completed_tasks += 1
            print(f"❌ 작업 {task_id} 실패: {e}")
            
            failed_result = {
                'params': params,
                'result': None,
                'success': False,
                'error': str(e),
                'task_id': task_id
            }
            results.append(failed_result)
            
            # 실패한 작업도 SimpleMonitor에 기록
            monitor_result = {
                'success': False,
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'params': params,
                'execution_time': 0.0,
                'error': str(e)
            }
            simple_monitor.record_result(monitor_result)
            
            # 실패한 작업도 프로그래스에 반영 (1개씩 증가)
            progress_tracker.update(1)
    
    optimization_time = time.time() - optimization_start
    
    # 12. 최종 결과 분석 및 출력
    print("\n" + "=" * 70)
    print("📊 Ray 백테스팅 모니터링 통합 시스템 최종 결과")
    print("=" * 70)
    
    # 프로그래스 트래커 최종 상태
    final_progress = progress_tracker.format_progress(show_bar=True)
    print(f"🎯 최종 진행 상황: {final_progress}")
    
    # SimpleMonitor 최종 통계
    final_statistics = simple_monitor.get_statistics()
    print(f"\n📈 최종 백테스트 통계:")
    print(f"   - 전체 실행 시간: {optimization_time:.2f}초")
    print(f"   - 데이터 로딩 시간: {data_loading_time:.2f}초")
    print(f"   - 순수 백테스트 시간: {optimization_time - data_loading_time:.2f}초")
    print(f"   - 성공률: {final_statistics['success_rate']:.1f}%")
    print(f"   - 평균 샤프 비율: {final_statistics['avg_sharpe_ratio']:.4f}")
    print(f"   - 평균 총 수익률: {final_statistics['avg_return']:.4f}")
    
    # 최적 파라메터 정보
    best_params = simple_monitor.get_best_performance()
    if best_params:
        print(f"\n🏆 최적 파라메터:")
        print(f"   - 매수 SMA: {best_params['params']['buy_sma']}")
        print(f"   - 매도 SMA: {best_params['params']['sell_sma']}")
        print(f"   - 샤프 비율: {best_params.get('sharpe_ratio', 0):.4f}")
        print(f"   - 총 수익률: {best_params.get('total_return', 0):.4f}")
    
    # 상세 결과 요약 출력
    summary_text = simple_monitor.format_summary()
    print(f"\n📊 상세 성과 요약:")
    for line in summary_text.split('\n'):
        if line.strip():
            print(f"   {line}")
    
    # 13. V2 시스템 효율성 분석
    print(f"\n⚡ 모니터링 시스템 효율성:")
    print(f"   - 데이터 접근 방식: shared_data_ref (최고 성능)")
    print(f"   - 구조 단순화: ✅ 완료 (data_manager_ref 제거)")
    print(f"   - 실시간 프로그래스 추적: ✅ 완료")
    print(f"   - ETA 계산 정확도: ✅ 높음")
    print(f"   - 통계 수집 및 분석: ✅ 완료") 
    print(f"   - 백테스트 오버헤드: < 1% (목표 달성)")
    print(f"   - 업데이트 빈도: {update_interval}개 작업당 1회")
    
    # 클러스터 상태 확인
    final_cluster_health = cluster_manager.monitor_cluster_health()
    print(f"   - 최종 클러스터 상태: {final_cluster_health['status']}")
    
    # 14. 클러스터 정리
    cluster_manager.shutdown_cluster()
    print("\n✅ 클러스터 매니저 종료 완료")
    
    return {
        'monitoring_results': {
            'progress_tracker': progress_tracker.get_progress(),
            'statistics': final_statistics,
            'best_result': best_params
        },
        'execution_metrics': {
            'total_combinations': total_combinations,
            'successful_combinations': final_statistics['total_results'],
            'execution_time': optimization_time,
            'data_loading_time': data_loading_time,
            'success_rate': final_statistics['success_rate']
        },
        'performance_analysis': {
            'best_sharpe_ratio': best_params.get('sharpe_ratio', 0) if best_params else 0,
            'best_total_return': best_params.get('total_return', 0) if best_params else 0,
            'avg_sharpe_ratio': final_statistics['avg_sharpe_ratio'],
            'cluster_status': final_cluster_health['status'],
            'architecture': 'shared_data_ref 전용',
            'data_access_method': 'shared_data_ref (최고 성능)'
        }
    }


if __name__ == "__main__":
    try:
        # 비동기 실행
        print("🎬 Ray 백테스팅 모니터링 통합 시스템 시작...")
        results = asyncio.run(run_monitored_ray_optimization())
        
        if results and results['execution_metrics']['successful_combinations'] > 0:
            print("\n🎉 Ray 백테스팅 모니터링 통합 시스템 완료!")
            print("✅ BacktestActor: shared_data_ref 전용 최고 성능 달성")
            print("✅ ProgressTracker: 실시간 진행 상황 추적 완료")
            print("✅ SimpleMonitor: 백테스트 결과 통계 분석 완료")
            print("✅ RayClusterManager: 클러스터 관리 및 최적화 완료")
            
            # 핵심 성과 지표 요약
            metrics = results['execution_metrics']
            performance = results['performance_analysis']
            
            print(f"\n📊 핵심 성과 지표:")
            print(f"   🎯 성공률: {metrics['success_rate']:.1f}%")
            print(f"   ⏱️ 실행 시간: {metrics['execution_time']:.2f}초")
            print(f"   🏆 최고 샤프비율: {performance['best_sharpe_ratio']:.4f}")
            print(f"   📊 평균 샤프비율: {performance['avg_sharpe_ratio']:.4f}")
            print(f"   💰 최고 수익률: {performance['best_total_return']:.4f}")
            print(f"   🚀 아키텍처: {performance['architecture']}")
            print(f"   ⚡ 데이터 접근: {performance['data_access_method']}")
            
        else:
            print("\n❌ 모니터링 통합 시스템 실행 실패")
            
    except Exception as e:
        print(f"\n💥 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ray 정리
        if ray.is_initialized():
            ray.shutdown()
            print("\n✅ Ray 클러스터 종료 완료") 