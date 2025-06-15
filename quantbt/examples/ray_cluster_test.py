"""
SimpleSMAStrategy RayClusterManager 기반 파라메터 최적화 예제

RayClusterManager를 활용한 효율적인 분산 백테스팅을 수행합니다.

핵심 변경사항:
- RayClusterManager를 통한 클러스터 관리
- 리소스 최적화 및 모니터링
- 자동 워커 수 계산
- 클러스터 상태 진단
"""

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
from datetime import datetime
import time
import ray

from quantbt import (
    # Dict Native 전략 시스템
    TradingStrategy,
    BacktestEngine,
    
    # 기본 모듈들
    SimpleBroker, 
    BacktestConfig,
    UpbitDataProvider,
    
    # 주문 관련
    Order, OrderSide, OrderType,
)

# Ray 기반 최적화 시스템 (RayClusterManager 포함)
from quantbt.ray import (
    RayClusterManager,
    RayDataManager, 
    BacktestActor,
    QuantBTEngineAdapter,
    RayResultAggregator,
    RayParameterOptimizer
)


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


async def run_sma_optimization_with_cluster_manager():
    """SimpleSMAStrategy RayClusterManager 기반 파라메터 최적화 (개선된 RayDataManager 사용)"""
    
    print("🚀 RayClusterManager + 개선된 RayDataManager 기반 SimpleSMAStrategy 파라메터 최적화 시작")
    print("=" * 70)
    
    # 1. RayClusterManager 설정 및 초기화
    ray_cluster_config = {
        "num_cpus": 4,
        "object_store_memory": 1000 * 1024 * 1024,  # 1GB
        "ignore_reinit_error": True,
        "logging_level": "ERROR"
    }
    
    cluster_manager = RayClusterManager(ray_cluster_config)
    
    print("🔧 Ray 클러스터 초기화 중...")
    if not cluster_manager.initialize_cluster():
        print("❌ Ray 클러스터 초기화 실패")
        return
    
    print("✅ Ray 클러스터 초기화 완료")
    
    # 2. 클러스터 상태 및 리소스 정보 출력
    cluster_resources = cluster_manager.get_cluster_resources()
    available_resources = cluster_manager.get_available_resources()
    
    print(f"📊 클러스터 리소스:")
    print(f"   - 총 CPU: {cluster_resources['cpu']}")
    print(f"   - Object Store: {cluster_resources['object_store']:,} bytes")
    print(f"   - 노드 수: {cluster_resources['nodes']}")
    print(f"   - 사용 가능한 CPU: {available_resources['cpu']}")
    
    # 3. 최적 워커 수 계산
    optimal_workers = cluster_manager.calculate_optimal_workers(
        memory_per_worker_gb=0.2,  # 워커당 200MB
        max_workers=8
    )
    
    print(f"⚡ 최적 워커 수: {optimal_workers}")
    
    # 4. 백테스트 기본 설정
    config = BacktestConfig(
        symbols=["KRW-BTC"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        timeframe="1m",
        initial_cash=10_000_000,
        commission_rate=0.0,
        slippage_rate=0.0,
        save_portfolio_history=False
    )
    print("✅ 백테스트 설정 완료")
    
    # 5. 개선된 RayDataManager 생성 및 데이터 로딩
    print("\n🔧 개선된 RayDataManager 생성 및 데이터 로딩")
    data_manager = RayDataManager.remote()
    print("✅ RayDataManager 생성 완료")
    
    # 데이터 미리 로딩 (제로카피 방식)
    print("📊 실제 데이터 로딩 중... (제로카피 방식)")
    data_loading_start = time.time()
    
    data_ref = await data_manager.load_real_data.remote(
        symbols=config.symbols,
        start_date=config.start_date,
        end_date=config.end_date,
        timeframe=config.timeframe
    )
    
    data_loading_time = time.time() - data_loading_start
    print(f"✅ 데이터 로딩 완료: {data_loading_time:.2f}초")
    
    # 캐시 통계 확인
    cache_stats = await data_manager.get_cache_stats.remote()
    print(f"📈 캐시 통계: {cache_stats['cache_size']}개 데이터, {cache_stats['total_data_size']:,} bytes")
    
    # 6. 워커 환경 준비
    worker_env = cluster_manager.prepare_worker_environment(
        expected_tasks=16,  # 예상 작업 수
        memory_per_task_mb=200  # 작업당 메모리
    )
    
    print(f"🎯 워커 환경 준비:")
    print(f"   - 최적 워커 수: {worker_env['optimal_workers']}")
    print(f"   - 작업당 메모리: {worker_env['memory_per_task_mb']}MB")
    
    # 7. 개선된 RayDataManager를 사용하는 BacktestActor 생성
    num_actors = worker_env['optimal_workers']
    print(f"\n🎯 {num_actors}개 BacktestActor 생성 중... (개선된 RayDataManager 사용)")
    
    actors = []
    for i in range(num_actors):
        # 개선된 RayDataManager 참조 전달
        actor = BacktestActor.remote(f"actor_{i}", data_manager_ref=data_manager)
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
    }
    
    init_results = await asyncio.gather(*[
        actor.initialize_engine.remote(config_dict) for actor in actors
    ])
    
    successful_actors = sum(init_results)
    print(f"✅ BacktestActor 초기화: {successful_actors}/{num_actors}개 성공")
    
    # 8. 클러스터 상태 모니터링
    cluster_health = cluster_manager.monitor_cluster_health()
    print(f"\n📊 클러스터 상태: {cluster_health['status']}")
    print(f"   - 노드 수: {cluster_health['nodes']}")
    print(f"   - CPU 사용률: {(cluster_resources['cpu'] - cluster_health['available']['cpu']) / cluster_resources['cpu'] * 100:.1f}%")
    
    # 9. 파라메터 그리드 정의
    param_grid = {
        'buy_sma': [10, 15, 20, 25],      # 매수 SMA: 10, 15, 20, 25
        'sell_sma': [25, 30, 35, 40]      # 매도 SMA: 25, 30, 35, 40
    }
    total_combinations = len(param_grid['buy_sma']) * len(param_grid['sell_sma'])
    print(f"\n✅ 파라메터 그리드 정의 완료: {total_combinations}개 조합")
    print(f"   - 매수 SMA: {param_grid['buy_sma']}")
    print(f"   - 매도 SMA: {param_grid['sell_sma']}")
    
    # 10. 파라메터 조합 생성
    from itertools import product
    param_combinations = []
    for buy_sma, sell_sma in product(param_grid['buy_sma'], param_grid['sell_sma']):
        param_combinations.append({
            'buy_sma': buy_sma,
            'sell_sma': sell_sma
        })
    
    # 11. 분산 백테스트 실행 (개선된 RayDataManager 사용)
    print("\n⚡ 분산 백테스트 실행 시작 (개선된 RayDataManager 제로카피)")
    optimization_start = time.time()
    
    # Actor별로 작업 분배
    tasks = []
    for i, params in enumerate(param_combinations):
        actor_idx = i % len(actors)
        actor = actors[actor_idx]
        
        task = actor.execute_backtest.remote(params, SimpleSMAStrategy)
        tasks.append((i, params, task))
    
    # 모든 작업 완료 대기
    print(f"📊 {total_combinations}개 백테스트 병렬 실행 중...")
    
    results = []
    for i, params, task in tasks:
        try:
            result = await task
            results.append({
                'params': params,
                'result': result,
                'success': True,
                'task_id': i
            })
        except Exception as e:
            print(f"❌ 작업 {i} 실패: {e}")
            results.append({
                'params': params,
                'result': None,
                'success': False,
                'error': str(e),
                'task_id': i
            })
    
    optimization_time = time.time() - optimization_start
    
    # 12. 최종 클러스터 상태 확인
    final_cluster_health = cluster_manager.monitor_cluster_health()
    print(f"\n📊 최종 클러스터 상태: {final_cluster_health['status']}")
    
    # 13. 결과 분석
    print("\n" + "=" * 70)
    print("📊 개선된 RayClusterManager + RayDataManager 최적화 결과 분석")
    print("=" * 70)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"✅ 총 실행 시간: {optimization_time:.2f}초")
    print(f"✅ 데이터 로딩 시간: {data_loading_time:.2f}초")
    print(f"✅ 백테스트 실행 시간: {optimization_time - data_loading_time:.2f}초")
    print(f"✅ 성공한 조합: {len(successful_results)}/{total_combinations}개")
    print(f"✅ 성공률: {len(successful_results)/total_combinations*100:.1f}%")
    
    if successful_results:
        # 최적 파라메터 찾기
        best_result = max(successful_results, 
                         key=lambda x: x['result'].get('sharpe_ratio', -999))
        
        print(f"\n🏆 최적 파라메터:")
        print(f"   - 매수 SMA: {best_result['params']['buy_sma']}")
        print(f"   - 매도 SMA: {best_result['params']['sell_sma']}")
        
        print(f"\n📈 최고 성과:")
        print(f"   - 샤프 비율: {best_result['result'].get('sharpe_ratio', 0):.4f}")
        print(f"   - 총 수익률: {best_result['result'].get('total_return', 0):.4f}")
        
        # 성능 통계
        sharpe_ratios = [r['result'].get('sharpe_ratio', 0) for r in successful_results]
        returns = [r['result'].get('total_return', 0) for r in successful_results]
        
        print(f"\n📊 성능 통계:")
        print(f"   - 평균 샤프 비율: {sum(sharpe_ratios)/len(sharpe_ratios):.4f}")
        print(f"   - 최고 샤프 비율: {max(sharpe_ratios):.4f}")
        print(f"   - 최저 샤프 비율: {min(sharpe_ratios):.4f}")
        print(f"   - 평균 수익률: {sum(returns)/len(returns):.4f}")
    
    # 14. 개선된 시스템 효율성 분석
    print(f"\n⚡ 개선된 RayClusterManager + RayDataManager 효율성:")
    
    # 클러스터 상세 정보
    detailed_info = cluster_manager.get_detailed_cluster_info()
    print(f"   - 클러스터 가동 시간: {detailed_info['metrics']['uptime']:.1f}초")
    print(f"   - CPU 사용률: {detailed_info['metrics']['cpu_utilization']:.1%}")
    print(f"   - 자동 워커 수 계산: {optimal_workers}개")
    print(f"   - 리소스 모니터링: {cluster_health['status']}")
    
    # 최종 캐시 통계
    final_cache_stats = await data_manager.get_cache_stats.remote()
    print(f"   - 데이터 캐시 히트: {final_cache_stats['total_access_count']}회")
    print(f"   - 제로카피 데이터 공유: 효율적인 메모리 사용")
    print(f"   - 고급 캐싱: 통계 및 모니터링 지원")
    
    # 15. 클러스터 정리
    cluster_manager.shutdown_cluster()
    print("✅ 클러스터 매니저 종료 완료")
    
    return {
        'best_params': best_result['params'] if successful_results else {},
        'best_sharpe_ratio': best_result['result'].get('sharpe_ratio', 0) if successful_results else 0,
        'best_total_return': best_result['result'].get('total_return', 0) if successful_results else 0, 
        'total_combinations': total_combinations,
        'successful_combinations': len(successful_results),
        'execution_time': optimization_time,
        'data_loading_time': data_loading_time,
        'cluster_performance': {
            'optimal_workers': optimal_workers,
            'cluster_status': final_cluster_health['status'],
            'cpu_utilization': detailed_info['metrics']['cpu_utilization'],
            'uptime': detailed_info['metrics']['uptime']
        },
        'cache_stats': final_cache_stats
    }


if __name__ == "__main__":
    try:
        # 비동기 실행
        results = asyncio.run(run_sma_optimization_with_cluster_manager())
        
        if results and results['successful_combinations'] > 0:
            print("\n🎉 SimpleSMAStrategy RayClusterManager 최적화 완료!")
            print("✅ RayClusterManager 기반 클러스터 관리 성공")
            print("✅ 자동 리소스 최적화 및 모니터링 완료")
            print("최적 파라메터로 실제 백테스팅을 실행해보세요.")
        else:
            print("\n❌ 최적화 실패")
            
    except Exception as e:
        print(f"\n💥 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ray 정리
        if ray.is_initialized():
            ray.shutdown()
            print("\n✅ Ray 클러스터 종료 완료")
