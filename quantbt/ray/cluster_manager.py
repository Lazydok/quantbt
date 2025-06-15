"""
Ray 클러스터 매니저

Ray 클러스터의 초기화, 관리, 모니터링을 담당하는 핵심 클래스
TDD Green 단계: 테스트를 통과시키는 최소한의 구현
"""

import ray
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RayClusterManager:
    """Ray 클러스터 관리자
    
    Ray 클러스터의 생명주기를 관리하고 리소스 상태를 모니터링합니다.
    백테스팅 워커들을 위한 최적의 실행 환경을 제공합니다.
    
    주요 기능:
    - Ray 클러스터 초기화 및 종료
    - 리소스 모니터링 및 최적화
    - 워커 수 자동 계산
    - 클러스터 상태 진단
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Ray 클러스터 매니저 초기화
        
        Args:
            config: Ray 클러스터 설정 딕셔너리
                - num_cpus: 사용할 CPU 수
                - object_store_memory: Object Store 메모리 크기
                - 기타 Ray 초기화 옵션들
        
        Raises:
            ValueError: 잘못된 설정값이 포함된 경우
        """
        self._validate_config(config)
        
        self.config = config.copy()
        self.status = 'not_initialized'
        self.cluster_info = None
        self._initialization_time = None
        
        logger.info(f"Ray 클러스터 매니저 생성: {config}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """설정값 검증
        
        Args:
            config: 검증할 설정 딕셔너리
            
        Raises:
            ValueError: 잘못된 설정값이 포함된 경우
        """
        if 'num_cpus' in config and config['num_cpus'] < 0:
            raise ValueError("num_cpus는 0 이상이어야 합니다")
        
        if 'object_store_memory' in config and config['object_store_memory'] < 0:
            raise ValueError("object_store_memory는 0 이상이어야 합니다")
    
    def initialize_cluster(self) -> bool:
        """Ray 클러스터 초기화
        
        이미 초기화된 Ray 클러스터가 있으면 해당 클러스터를 사용하고,
        없으면 새로운 클러스터를 생성합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            if self.status == 'initialized':
                logger.info("클러스터가 이미 초기화되어 있습니다")
                return True
            
            # Ray가 이미 초기화되어 있지 않으면 새로 초기화
            if not ray.is_initialized():
                ray.init(**self.config)
                logger.info(f"새로운 Ray 클러스터 초기화: {self.config}")
            else:
                logger.info("기존 Ray 클러스터 사용")
            
            # 클러스터 정보 수집
            self._collect_cluster_info()
            
            self.status = 'initialized'
            self._initialization_time = datetime.now()
            
            logger.info("Ray 클러스터 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"클러스터 초기화 실패: {e}")
            self.status = 'error'
            return False
    
    def _collect_cluster_info(self) -> None:
        """클러스터 정보 수집"""
        if not ray.is_initialized():
            self.cluster_info = None
            return
        
        cluster_resources = ray.cluster_resources()
        
        self.cluster_info = {
            'num_cpus': cluster_resources.get('CPU', 0),
            'object_store_memory': cluster_resources.get('object_store_memory', 0),
            'nodes': len(ray.nodes()),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """클러스터 리소스 정보 조회
        
        Returns:
            Dict[str, Any]: 클러스터 리소스 정보
                - cpu: 총 CPU 수
                - object_store: Object Store 메모리 크기
                - nodes: 노드 수
        """
        if not ray.is_initialized():
            return {'cpu': 0, 'object_store': 0, 'nodes': 0}
        
        cluster_resources = ray.cluster_resources()
        
        return {
            'cpu': cluster_resources.get('CPU', 0),
            'object_store': cluster_resources.get('object_store_memory', 0),
            'nodes': len(ray.nodes())
        }
    
    def get_available_resources(self) -> Dict[str, Any]:
        """사용 가능한 리소스 조회
        
        Returns:
            Dict[str, Any]: 사용 가능한 리소스 정보
        """
        if not ray.is_initialized():
            return {'cpu': 0, 'object_store': 0}
        
        available_resources = ray.available_resources()
        
        return {
            'cpu': available_resources.get('CPU', 0),
            'object_store': available_resources.get('object_store_memory', 0)
        }
    
    def calculate_optimal_workers(self, 
                                memory_per_worker_gb: Optional[float] = None,
                                max_workers: Optional[int] = None) -> int:
        """최적 워커 수 계산
        
        CPU 수와 메모리 제약을 고려하여 최적의 워커 수를 계산합니다.
        
        Args:
            memory_per_worker_gb: 워커당 필요한 메모리 (GB)
            max_workers: 최대 워커 수 제한
            
        Returns:
            int: 최적 워커 수
        """
        if not ray.is_initialized():
            return 1
        
        resources = self.get_cluster_resources()
        cpu_count = int(resources['cpu'])
        
        # CPU 기반 계산
        optimal_workers = max(1, cpu_count)
        
        # 메모리 제약 고려
        if memory_per_worker_gb and resources['object_store'] > 0:
            # Object Store 메모리를 GB로 변환
            available_memory_gb = resources['object_store'] / (1024 ** 3)
            max_workers_by_memory = int(available_memory_gb / memory_per_worker_gb)
            optimal_workers = min(optimal_workers, max_workers_by_memory)
        
        # 사용자 정의 최대값 적용
        if max_workers:
            optimal_workers = min(optimal_workers, max_workers)
        
        return max(1, optimal_workers)
    
    def monitor_cluster_health(self) -> Dict[str, Any]:
        """클러스터 상태 모니터링
        
        Returns:
            Dict[str, Any]: 클러스터 상태 정보
        """
        if not ray.is_initialized():
            return {
                'status': 'error',
                'nodes': 0,
                'resources': {'cpu': 0, 'object_store': 0}
            }
        
        try:
            nodes = ray.nodes()
            resources = self.get_cluster_resources()
            available = self.get_available_resources()
            
            # 간단한 상태 판단 로직
            if len(nodes) == 0:
                status = 'error'
            elif available['cpu'] < resources['cpu'] * 0.1:  # CPU 90% 이상 사용 중
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'nodes': len(nodes),
                'resources': resources,
                'available': available,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"클러스터 상태 모니터링 실패: {e}")
            return {
                'status': 'error',
                'nodes': 0,
                'resources': {'cpu': 0, 'object_store': 0},
                'error': str(e)
            }
    
    def get_detailed_cluster_info(self) -> Dict[str, Any]:
        """상세한 클러스터 정보 조회
        
        Returns:
            Dict[str, Any]: 상세 클러스터 정보
        """
        if not ray.is_initialized():
            return {
                'cluster_id': None,
                'nodes': [],
                'resources': {},
                'status': 'not_initialized',
                'metrics': {}
            }
        
        # 기본 정보 수집
        nodes = ray.nodes()
        resources = self.get_cluster_resources()
        
        # 성능 메트릭 계산
        uptime = (datetime.now() - self._initialization_time).total_seconds() if self._initialization_time else 0
        available = self.get_available_resources()
        cpu_utilization = 1.0 - (available['cpu'] / max(resources['cpu'], 1))
        
        return {
            'cluster_id': id(self),  # 임시로 객체 ID 사용
            'nodes': [{'node_id': node['NodeID'], 'alive': node['Alive']} for node in nodes],
            'resources': resources,
            'status': self.status,
            'metrics': {
                'uptime': uptime,
                'cpu_utilization': cpu_utilization
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def prepare_worker_environment(self, 
                                 expected_tasks: int,
                                 memory_per_task_mb: int) -> Dict[str, Any]:
        """워커 환경 준비
        
        백테스트 작업을 위한 최적의 워커 환경을 설정합니다.
        
        Args:
            expected_tasks: 예상 작업 수
            memory_per_task_mb: 작업당 메모리 사용량 (MB)
            
        Returns:
            Dict[str, Any]: 워커 환경 설정
        """
        memory_per_worker_gb = memory_per_task_mb / 1024
        optimal_workers = self.calculate_optimal_workers(
            memory_per_worker_gb=memory_per_worker_gb
        )
        
        resources_per_worker = {
            'cpu': 1.0,
            'memory_mb': memory_per_task_mb
        }
        
        return {
            'optimal_workers': optimal_workers,
            'resources_per_worker': resources_per_worker,
            'expected_tasks': expected_tasks,
            'memory_per_task_mb': memory_per_task_mb
        }
    
    def shutdown_cluster(self) -> bool:
        """클러스터 종료
        
        Returns:
            bool: 종료 성공 여부
        """
        try:
            # 세션 레벨에서 Ray를 관리하므로 실제로는 종료하지 않음
            # 상태만 업데이트
            self.status = 'shutdown'
            self.cluster_info = None
            self._initialization_time = None
            
            logger.info("클러스터 매니저 종료 완료")
            return True
            
        except Exception as e:
            logger.error(f"클러스터 종료 실패: {e}")
            return False 