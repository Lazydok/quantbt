"""
Ray 기반 단순화된 백테스트 워커 (v2)

shared_data_ref만 사용하는 단순화된 구조로 최고 성능을 제공합니다.
- shared_data_ref 전용 설계
- 526배 빠른 데이터 접근
- 단순하고 명확한 구조
- 메모리 효율성 극대화
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
    """Ray 기반 단순화된 백테스트 워커 Actor (v2)
    
    shared_data_ref만 사용하여 최고 성능을 제공합니다.
    """
    
    def __init__(self, actor_id: str, shared_data_ref: ray.ObjectRef):
        """Actor 초기화
        
        Args:
            actor_id: Actor 고유 식별자
            shared_data_ref: 공유 데이터 참조 (필수)
        """
        self.actor_id = actor_id
        self.shared_data_ref = shared_data_ref
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
        logger.info(f"BacktestActor {self.actor_id} 초기화 완료 (shared_data_ref 전용)")
    
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
            
            # BacktestConfig 생성
            backtest_config = BacktestConfig(**config)
            
            # QuantBTEngineAdapter 생성 (shared_data_ref만 사용)
            self._engine_adapter = QuantBTEngineAdapter(
                base_config=backtest_config,
                shared_data_ref=self.shared_data_ref
            )
            
            self._health_status = "ready"
            logger.info(f"Actor {self.actor_id} 엔진 초기화 성공")
            return True
            
        except Exception as e:
            self._health_status = "error"
            logger.error(f"Actor {self.actor_id} 엔진 초기화 실패: {e}")
            return False
    
    def execute_backtest(self, params: Dict, strategy_class: Type) -> Dict:
        """백테스트 실행
        
        Args:
            params: 전략 파라미터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 결과
        """
        if not self._engine_adapter:
            raise RuntimeError(f"Actor {self.actor_id}: 엔진이 초기화되지 않았습니다")
        
        execution_start = time.time()
        self._last_heartbeat = execution_start
        
        try:
            # 백테스트 실행
            result = self._engine_adapter.execute_backtest(params, strategy_class)
            
            # 통계 업데이트
            execution_time = time.time() - execution_start
            self._execution_count += 1
            self._total_execution_time += execution_time
            
            # 주기적 메모리 정리
            if self._execution_count % self._cleanup_interval == 0:
                self.cleanup_resources()
            
            logger.debug(f"Actor {self.actor_id}: 백테스트 완료 ({execution_time:.3f}초)")
            return result
            
        except Exception as e:
            logger.error(f"Actor {self.actor_id} 백테스트 실행 실패: {e}")
            self._health_status = "error"
            raise
    
    def get_health_status(self) -> Dict:
        """Actor 상태 확인
        
        Returns:
            Dict: 상태 정보
        """
        current_time = time.time()
        
        return {
            "actor_id": self.actor_id,
            "status": self._health_status,
            "last_heartbeat": self._last_heartbeat,
            "uptime": current_time - (self._last_heartbeat - self._total_execution_time),
            "engine_initialized": self._engine_adapter is not None,
            "shared_data_available": self.shared_data_ref is not None
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
            "data_access_method": "shared_data_ref (최고 성능)"
        }
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            # 가비지 컬렉션 실행
            gc.collect()
            
            # 통계 업데이트
            self._memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.debug(f"워커 {self.actor_id}: 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"워커 {self.actor_id} 리소스 정리 실패: {e}") 