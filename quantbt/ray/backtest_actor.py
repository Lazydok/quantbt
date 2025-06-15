"""
Ray 기반 Stateful 백테스트 워커

Ray Actor를 활용한 분산 백테스팅 시스템의 핵심 워커입니다.
- Stateful 설계로 백테스트 엔진 재사용
- Zero-copy 데이터 공유
- 실시간 성능 모니터링
- 에러 처리 및 복구
"""

import ray
import time
import psutil
import gc
from typing import Dict, Any, Type, Optional
from datetime import datetime
import logging

# 로깅 설정
logger = logging.getLogger(__name__)


@ray.remote
class BacktestActor:
    """Ray 기반 백테스트 워커 Actor
    
    상태를 유지하며 여러 백테스트를 효율적으로 실행합니다.
    """
    
    def __init__(self, actor_id: str, data_manager_ref: Optional[ray.ObjectRef] = None):
        """Actor 초기화
        
        Args:
            actor_id: Actor 고유 식별자
            data_manager_ref: RayDataManager 참조 (선택사항)
        """
        self.actor_id = actor_id
        self.data_manager_ref = data_manager_ref
        self._engine_adapter = None
        self._health_status = "initializing"
        self._last_heartbeat = time.time()
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._memory_usage = 0
        
        # 메모리 관리
        self._memory_threshold_mb = 500  # 500MB 임계값
        self._cleanup_interval = 10  # 10회 실행마다 정리
        
        self._health_status = "ready"
        logger.info(f"BacktestActor {self.actor_id} 초기화 완료 "
                   f"(RayDataManager: {'사용' if data_manager_ref else '미사용'})")
    
    def initialize_engine(self, config: Dict) -> bool:
        """백테스트 엔진 초기화
        
        Args:
            config: 백테스트 설정
            
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            from quantbt.core.value_objects.backtest_config import BacktestConfig
            from quantbt.ray.quantbt_engine_adapter import QuantBTEngineAdapter
            
            # 공유 데이터 참조 추출
            shared_data_ref = config.pop('shared_data_ref', None)
            
            # BacktestConfig 생성
            backtest_config = BacktestConfig(**config)
            
            # QuantBTEngineAdapter 생성 (공유 데이터 참조 포함)
            self._engine_adapter = QuantBTEngineAdapter(
                base_config=backtest_config,
                data_manager_ref=self.data_manager_ref,
                shared_data_ref=shared_data_ref
            )
            
            self._health_status = "ready"
            logger.info(f"Actor {self.actor_id} 엔진 초기화 성공")
            return True
            
        except Exception as e:
            self._health_status = "error"
            logger.error(f"Actor {self.actor_id} 엔진 초기화 실패: {e}")
            return False
    
    def run_backtest(self, params: Dict, strategy_class: Type) -> Dict:
        """파라메터로 백테스트 실행
        
        Args:
            params: 전략 파라메터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 결과
        """
        start_time = time.time()
        
        try:
            # 엔진이 초기화되지 않은 경우 에러
            if self._engine_adapter is None:
                raise RuntimeError("백테스트 엔진이 초기화되지 않았습니다")
            
            # 전략 인스턴스 생성 (Mock)
            strategy = strategy_class(**params)
            
            # 백테스트 실행
            result = self._engine_adapter.run(strategy)
            execution_time = time.time() - start_time
            
            # 통계 업데이트
            self._execution_count += 1
            self._total_execution_time += execution_time
            self._memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 결과 반환
            return {
                "params": params,
                "result": result,
                "execution_time": execution_time,
                "worker_id": self.actor_id,
                "success": True
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"워커 {self.actor_id} 백테스트 실행 실패: {e}")
            
            return {
                "error": str(e),
                "params": params,
                "worker_id": self.actor_id,
                "execution_time": execution_time,
                "success": False
            }
    
    def execute_backtest(self, params: Dict, strategy_class: Type) -> Dict:
        """백테스트 실행 (QuantBTEngineAdapter 사용)
        
        Args:
            params: 전략 파라메터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 결과
        """
        start_time = time.time()
        
        try:
            # 엔진이 초기화되지 않은 경우 에러
            if self._engine_adapter is None:
                raise RuntimeError("백테스트 엔진이 초기화되지 않았습니다")
            
            # QuantBTEngineAdapter를 통한 백테스트 실행
            result = self._engine_adapter.execute_backtest(params, strategy_class)
            execution_time = time.time() - start_time
            
            # 통계 업데이트
            self._execution_count += 1
            self._total_execution_time += execution_time
            self._memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 성공 플래그 추가
            result['success'] = True
            result['execution_time'] = execution_time
            result['worker_id'] = self.actor_id
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"워커 {self.actor_id} 백테스트 실행 실패: {e}")
            
            return {
                "error": str(e),
                "params": params,
                "worker_id": self.actor_id,
                "execution_time": execution_time,
                "success": False
            }
    
    def get_worker_stats(self) -> Dict:
        """워커 통계 정보 반환
        
        Returns:
            Dict: 워커 통계
        """
        # 현재 메모리 사용량 업데이트
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self._memory_usage = current_memory
        
        return {
            "worker_id": self.actor_id,
            "status": self._health_status,
            "engine_initialized": self._engine_adapter is not None,
            "memory_usage_mb": current_memory,
            "last_heartbeat": self._last_heartbeat,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / max(self._execution_count, 1)
            ),
            "success_rate": (
                (self._execution_count - self._execution_count) 
                / max(self._execution_count, 1)
            ) if self._execution_count > 0 else 1.0
        }
    
    def get_shared_data(self) -> Any:
        """공유 데이터 접근
        
        Returns:
            Any: 공유 데이터
        """
        return self.data_manager_ref
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            # 가비지 컬렉션 실행
            gc.collect()
            
            # 통계 업데이트
            self._memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"워커 {self.actor_id}: 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"워커 {self.actor_id} 리소스 정리 실패: {e}")
    
    def update_config(self, new_config: Dict):
        """설정 업데이트
        
        Args:
            new_config: 새로운 설정
        """
        # 설정 업데이트는 현재 구현되지 않음
        logger.info(f"워커 {self.actor_id}: 설정 업데이트 완료")
    
    def health_check(self) -> Dict:
        """헬스 체크
        
        Returns:
            Dict: 헬스 상태
        """
        return {
            "worker_id": self.actor_id,
            "status": self._health_status,
            "engine_initialized": self._engine_adapter is not None,
            "memory_usage_mb": self._memory_usage,
            "last_heartbeat": self._last_heartbeat,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / max(self._execution_count, 1)
            ),
            "success_rate": (
                (self._execution_count - self._execution_count) 
                / max(self._execution_count, 1)
            ) if self._execution_count > 0 else 1.0
        }


# 백테스트 스케줄러 (Actor 관리)
class BacktestScheduler:
    """백테스트 작업 스케줄러
    
    여러 BacktestActor를 관리하고 작업을 분산합니다.
    """
    
    def __init__(self, num_workers: int = None):
        """스케줄러 초기화
        
        Args:
            num_workers: 워커 수 (None이면 CPU 코어 수)
        """
        if num_workers is None:
            num_workers = ray.cluster_resources().get('CPU', 4)
        
        self.num_workers = int(num_workers)
        self.workers = []
        self.task_queue = []
        self.results = []
        
        logger.info(f"BacktestScheduler 초기화: {self.num_workers}개 워커")
    
    def initialize_workers(self, shared_data_ref: ray.ObjectRef, base_config: Dict):
        """워커 초기화
        
        Args:
            shared_data_ref: 공유 데이터 참조
            base_config: 백테스트 기본 설정
        """
        # BacktestActor 워커들 생성
        self.workers = [
            BacktestActor.remote(f"worker_{i}", shared_data_ref)
            for i in range(self.num_workers)
        ]
        
        # 워커들 초기화
        init_tasks = [
            worker.initialize_engine.remote(base_config)
            for worker in self.workers
        ]
        
        # 모든 워커 초기화 완료 대기
        init_results = ray.get(init_tasks)
        
        success_count = sum(init_results)
        logger.info(f"워커 초기화 완료: {success_count}/{self.num_workers}")
        
        if success_count < self.num_workers:
            logger.warning(f"일부 워커 초기화 실패: {self.num_workers - success_count}개")
    
    def run_parallel_backtests(self, param_combinations: list, strategy_class: Type) -> list:
        """병렬 백테스트 실행
        
        Args:
            param_combinations: 파라메터 조합 리스트
            strategy_class: 전략 클래스
            
        Returns:
            list: 백테스트 결과 리스트
        """
        if not self.workers:
            raise RuntimeError("워커가 초기화되지 않았습니다")
        
        # 작업 생성
        tasks = []
        for i, params in enumerate(param_combinations):
            worker_idx = i % self.num_workers
            worker = self.workers[worker_idx]
            
            task = worker.run_backtest.remote(params, strategy_class)
            tasks.append(task)
        
        # 결과 수집
        results = ray.get(tasks)
        
        logger.info(f"병렬 백테스트 완료: {len(results)}개 결과")
        return results
    
    def get_worker_stats(self) -> list:
        """모든 워커 통계 수집
        
        Returns:
            list: 워커별 통계 리스트
        """
        if not self.workers:
            return []
        
        stats_tasks = [worker.get_worker_stats.remote() for worker in self.workers]
        return ray.get(stats_tasks)
    
    def cleanup_all_workers(self):
        """모든 워커 리소스 정리"""
        if not self.workers:
            return
        
        cleanup_tasks = [worker.cleanup_resources.remote() for worker in self.workers]
        ray.get(cleanup_tasks)
        
        logger.info("모든 워커 리소스 정리 완료")


# 백테스트 모니터 (실시간 모니터링)
@ray.remote
class BacktestMonitor:
    """백테스트 모니터링
    
    실시간으로 백테스트 진행 상황을 모니터링합니다.
    """
    
    def __init__(self):
        """모니터 초기화"""
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "best_sharpe_ratio": float('-inf'),
            "average_execution_time": 0.0,
            "start_time": time.time()
        }
        
        logger.info("BacktestMonitor 초기화 완료")
    
    def update_metrics(self, task_result: Dict):
        """메트릭 업데이트
        
        Args:
            task_result: 작업 결과
        """
        self.metrics["completed_tasks"] += 1
        
        if task_result.get("success", False):
            # 성공한 경우
            result = task_result.get("result", {})
            sharpe = result.get("sharpe_ratio", 0)
            
            if sharpe > self.metrics["best_sharpe_ratio"]:
                self.metrics["best_sharpe_ratio"] = sharpe
                
            # 평균 실행 시간 업데이트
            exec_time = task_result.get("execution_time", 0)
            current_avg = self.metrics["average_execution_time"]
            completed = self.metrics["completed_tasks"]
            
            self.metrics["average_execution_time"] = (
                (current_avg * (completed - 1) + exec_time) / completed
            )
        else:
            # 실패한 경우
            self.metrics["failed_tasks"] += 1
    
    def get_dashboard_data(self) -> Dict:
        """대시보드용 데이터 반환
        
        Returns:
            Dict: 대시보드 데이터
        """
        elapsed_time = time.time() - self.metrics["start_time"]
        
        return {
            "progress": (
                self.metrics["completed_tasks"] / max(self.metrics["total_tasks"], 1)
            ),
            "success_rate": (
                (self.metrics["completed_tasks"] - self.metrics["failed_tasks"]) 
                / max(self.metrics["completed_tasks"], 1)
            ) if self.metrics["completed_tasks"] > 0 else 1.0,
            "best_performance": self.metrics["best_sharpe_ratio"],
            "cluster_utilization": ray.available_resources(),
            "elapsed_time": elapsed_time,
            "average_execution_time": self.metrics["average_execution_time"],
            "total_tasks": self.metrics["total_tasks"],
            "completed_tasks": self.metrics["completed_tasks"],
            "failed_tasks": self.metrics["failed_tasks"]
        }
    
    def set_total_tasks(self, total: int):
        """총 작업 수 설정
        
        Args:
            total: 총 작업 수
        """
        self.metrics["total_tasks"] = total
        logger.info(f"총 작업 수 설정: {total}")
    
    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "best_sharpe_ratio": float('-inf'),
            "average_execution_time": 0.0,
            "start_time": time.time()
        }
        logger.info("메트릭 초기화 완료") 