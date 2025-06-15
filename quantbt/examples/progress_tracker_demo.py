"""
ProgressTracker와 SimpleMonitor 사용 예제

Phase 1 모니터링 시스템의 실제 사용 방법을 보여주는 데모입니다.
"""

import sys
import time
import random
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_dir = Path.cwd()
if 'examples' in str(current_dir):
    project_root = current_dir.parent
else:
    project_root = current_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ray 순환 import 문제 회피를 위해 직접 import
from quantbt.ray.monitoring.progress_tracker import ProgressTracker
from quantbt.ray.monitoring.simple_monitor import SimpleMonitor


def simulate_backtest_task(task_id: int, progress_tracker: ProgressTracker, monitor: SimpleMonitor):
    """개별 백테스트 작업 시뮬레이션"""
    
    # 작업 시간 시뮬레이션 (0.5초 ~ 2초)
    execution_time = random.uniform(0.5, 2.0)
    time.sleep(execution_time)
    
    # 성공/실패 시뮬레이션 (90% 성공률)
    success = random.random() > 0.1
    
    if success:
        # 성공한 경우 - 가상의 백테스트 결과 생성
        sharpe_ratio = random.uniform(0.5, 2.0)
        total_return = random.uniform(-0.1, 0.3)
        
        result = {
            'task_id': task_id,
            'success': True,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'execution_time': execution_time,
            'params': {
                'buy_sma': random.randint(10, 30),
                'sell_sma': random.randint(20, 50)
            }
        }
    else:
        # 실패한 경우
        result = {
            'task_id': task_id,
            'success': False,
            'error': 'Simulated failure',
            'execution_time': execution_time
        }
    
    # 결과 기록
    monitor.record_result(result)
    
    # 진행률 업데이트
    progress_tracker.update()
    
    return result


def demo_simple_monitoring():
    """간단한 모니터링 시스템 데모"""
    
    print("🚀 Ray 백테스팅 간단한 모니터링 시스템 데모")
    print("=" * 70)
    
    # 총 작업 수 설정 (예: 50개 파라메터 조합)
    total_tasks = 50
    
    # 모니터링 시스템 초기화
    progress_tracker = ProgressTracker(total_tasks)
    monitor = SimpleMonitor()
    
    print(f"📊 총 {total_tasks}개 백테스트 작업 시작\n")
    
    # 진행률 추적 시작
    progress_tracker.start()
    
    try:
        for task_id in range(1, total_tasks + 1):
            # 백테스트 작업 시뮬레이션
            result = simulate_backtest_task(task_id, progress_tracker, monitor)
            
            # 진행률 표시 (5개마다 또는 완료 시)
            if task_id % 5 == 0 or task_id == total_tasks:
                print(f"\r{progress_tracker.format_progress(show_bar=True)}")
                
                # 현재 통계 표시
                if task_id % 10 == 0 or task_id == total_tasks:
                    print(f"\n{monitor.format_summary()}")
                    print("-" * 50)
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단됨")
    
    # 최종 결과 요약
    print("\n" + "=" * 70)
    print("🎉 백테스트 완료!")
    print("=" * 70)
    
    final_progress = progress_tracker.get_progress()
    print(f"📈 최종 진행률: {final_progress['completed_tasks']}/{final_progress['total_tasks']} ({final_progress['percentage']:.1f}%)")
    
    print(f"\n{monitor.format_summary()}")
    
    # 최고 성과 상세 정보
    best_performance = monitor.get_best_performance()
    if best_performance:
        print(f"\n🏆 최고 성과 상세:")
        print(f"   샤프 비율: {best_performance['sharpe_ratio']:.4f}")
        print(f"   수익률: {best_performance['total_return']:.4f}")
        print(f"   파라메터: {best_performance['params']}")
        print(f"   실행 시간: {best_performance['execution_time']:.2f}초")


def demo_real_time_updates():
    """실시간 업데이트 데모"""
    
    print("\n🔄 실시간 업데이트 데모")
    print("=" * 50)
    
    total_tasks = 20
    progress_tracker = ProgressTracker(total_tasks)
    monitor = SimpleMonitor()
    
    progress_tracker.start()
    
    for task_id in range(1, total_tasks + 1):
        # 빠른 작업 시뮬레이션
        time.sleep(0.2)
        
        # 가상 결과 생성
        result = {
            'task_id': task_id,
            'success': True,
            'sharpe_ratio': random.uniform(0.8, 1.5),
            'total_return': random.uniform(0.05, 0.15),
            'execution_time': 0.2
        }
        
        monitor.record_result(result)
        progress_tracker.update()
        
        # 실시간 진행률 표시 (덮어쓰기)
        print(f"\r{progress_tracker.format_progress()}", end="", flush=True)
    
    print(f"\n\n✅ 실시간 업데이트 완료!")


if __name__ == "__main__":
    # 메인 데모 실행
    # demo_simple_monitoring()
    
    # 실시간 업데이트 데모
    demo_real_time_updates()
    
    print("\n💡 Phase 1 모니터링 시스템 데모 완료!")
    print("   다음 단계: 베이지안 최적화 시스템 구현 예정") 