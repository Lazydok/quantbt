"""
백테스트 워커 프로세스

Phase 1 멀티프로세싱 백테스팅 최적화를 위한 워커 프로세스 구현
"""

import sys
import os
import time
import traceback
import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


def backtest_worker_function(
    params: Dict[str, Any], 
    shared_data_path: str, 
    base_config: 'BacktestConfig',
    strategy_class: Type
) -> Dict[str, Any]:
    """워커 프로세스에서 실행되는 백테스팅 함수
    
    각 프로세스에서 독립적으로 실행되는 백테스팅 워커입니다.
    공유 데이터를 로드하고, 파라메터를 적용한 전략으로 백테스팅을 수행합니다.
    
    Args:
        params: 전략 파라메터 딕셔너리
        shared_data_path: 공유 데이터 파일 경로 (Parquet)
        base_config: 기본 백테스팅 설정
        strategy_class: 전략 클래스
        
    Returns:
        백테스팅 결과 딕셔너리
        {
            "parameters": params,
            "success": bool,
            "metrics": {...} or None,
            "error": str or None,
            "execution_time": float,
            "worker_info": {...}
        }
    """
    worker_start_time = time.time()
    worker_id = os.getpid()
    
    try:
        # 프로젝트 루트 패스 추가 (워커 프로세스용)
        _add_project_root_to_path()
        
        # 필요한 모듈 가져오기 (워커 프로세스 내에서)
        from quantbt import BacktestEngine, SimpleBroker
        
        logger.debug(f"워커 {worker_id} 시작: 파라메터 {params}")
        
        # 1. 공유 데이터 로딩
        load_start = time.time()
        shared_data = _load_shared_data(shared_data_path)
        load_time = time.time() - load_start
        
        # 2. 파라메터 적용된 전략 생성
        strategy_start = time.time()
        strategy = _create_strategy_with_params(strategy_class, params)
        strategy_time = time.time() - strategy_start
        
        # 3. 독립적인 백테스트 엔진 구성
        setup_start = time.time()
        broker, data_provider, engine = _setup_backtest_components(
            shared_data, base_config, strategy
        )
        setup_time = time.time() - setup_start
        
        # 4. 백테스팅 실행
        backtest_start = time.time()
        result = engine.run(base_config)
        backtest_time = time.time() - backtest_start
        
        # 5. 결과 메트릭 추출
        metrics_start = time.time()
        metrics = _extract_metrics_from_result(result)
        metrics_time = time.time() - metrics_start
        
        total_time = time.time() - worker_start_time
        
        logger.debug(f"워커 {worker_id} 성공 완료: {total_time:.3f}초")
        
        return {
            "parameters": params,
            "success": True,
            "metrics": metrics,
            "error": None,
            "execution_time": total_time,
            "worker_info": {
                "worker_id": worker_id,
                "timing": {
                    "load_time": load_time,
                    "strategy_time": strategy_time,
                    "setup_time": setup_time,
                    "backtest_time": backtest_time,
                    "metrics_time": metrics_time,
                    "total_time": total_time
                },
                "data_info": {
                    "data_rows": len(shared_data),
                    "data_symbols": shared_data["symbol"].unique().to_list(),
                    "date_range": {
                        "start": shared_data["timestamp"].min(),
                        "end": shared_data["timestamp"].max()
                    }
                }
            }
        }
        
    except Exception as e:
        error_msg = f"워커 {worker_id} 실행 실패: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        total_time = time.time() - worker_start_time
        
        return {
            "parameters": params,
            "success": False,
            "metrics": None,
            "error": error_msg,
            "execution_time": total_time,
            "worker_info": {
                "worker_id": worker_id,
                "error_traceback": traceback.format_exc()
            }
        }


def _add_project_root_to_path() -> None:
    """프로젝트 루트를 Python 경로에 추가 (워커 프로세스용)"""
    current_dir = Path.cwd()
    
    # 다양한 실행 경로 지원
    if 'examples' in str(current_dir):
        project_root = current_dir.parent.parent
    elif 'tests' in str(current_dir):
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    logger.debug(f"프로젝트 루트 추가: {project_root}")


def _load_shared_data(shared_data_path: str) -> pl.DataFrame:
    """공유 데이터 로딩
    
    Args:
        shared_data_path: 공유 데이터 파일 경로
        
    Returns:
        로딩된 시장 데이터
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        Exception: 데이터 로딩 실패 시
    """
    if not os.path.exists(shared_data_path):
        raise FileNotFoundError(f"공유 데이터 파일을 찾을 수 없습니다: {shared_data_path}")
    
    try:
        shared_data = pl.read_parquet(shared_data_path)
        
        if len(shared_data) == 0:
            raise ValueError("공유 데이터가 비어있습니다")
        
        # 필수 컬럼 확인
        required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in shared_data.columns]
        
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        logger.debug(f"공유 데이터 로딩 완료: {len(shared_data):,}행, {len(shared_data.columns)}열")
        
        return shared_data
        
    except Exception as e:
        logger.error(f"공유 데이터 로딩 실패: {e}")
        raise


def _create_strategy_with_params(strategy_class: Type, params: Dict[str, Any]):
    """파라메터를 적용한 전략 인스턴스 생성
    
    Args:
        strategy_class: 전략 클래스
        params: 전략 파라메터
        
    Returns:
        설정된 전략 인스턴스
    """
    try:
        # 전략 클래스의 __init__ 시그니처에 맞게 파라메터 전달
        strategy = strategy_class(**params)
        
        logger.debug(f"전략 생성 완료: {strategy_class.__name__} with {params}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"전략 생성 실패: {e}")
        raise


def _setup_backtest_components(shared_data: pl.DataFrame, base_config: 'BacktestConfig', strategy):
    """백테스트 구성요소 설정
    
    Args:
        shared_data: 공유 시장 데이터
        base_config: 기본 백테스트 설정
        strategy: 전략 인스턴스
        
    Returns:
        Tuple[SimpleBroker, MockDataProvider, BacktestEngine]
    """
    from quantbt import BacktestEngine, SimpleBroker
    
    # 1. 브로커 설정 (독립적인 인스턴스)
    broker = SimpleBroker(
        initial_cash=base_config.initial_cash,
        commission_rate=base_config.commission_rate,
        slippage_rate=base_config.slippage_rate
    )
    
    # 2. Mock 데이터 프로바이더 (이미 로딩된 데이터 사용)
    data_provider = MockDataProvider(shared_data)
    
    # 3. 백테스트 엔진 설정
    engine = BacktestEngine()
    engine.set_strategy(strategy)
    engine.set_broker(broker)
    engine.set_data_provider(data_provider)
    
    logger.debug("백테스트 구성요소 설정 완료")
    
    return broker, data_provider, engine


def _extract_metrics_from_result(result: 'BacktestResult') -> Dict[str, Any]:
    """백테스트 결과에서 메트릭 추출
    
    Args:
        result: 백테스트 결과
        
    Returns:
        메트릭 딕셔너리
    """
    try:
        metrics = {
            "total_return": getattr(result, 'total_return', 0.0),
            "annual_return": getattr(result, 'annual_return', 0.0),
            "sharpe_ratio": getattr(result, 'sharpe_ratio', 0.0),
            "max_drawdown": getattr(result, 'max_drawdown', 0.0),
            "win_rate": getattr(result, 'win_rate', 0.0),
            "profit_factor": getattr(result, 'profit_factor', 0.0),
            "total_trades": getattr(result, 'total_trades', 0),
            "avg_trade_return": getattr(result, 'avg_trade_return', 0.0),
            "volatility": getattr(result, 'volatility', 0.0),
            "calmar_ratio": getattr(result, 'calmar_ratio', 0.0),
        }
        
        # None 값을 0으로 변환
        for key, value in metrics.items():
            if value is None:
                metrics[key] = 0.0
        
        logger.debug(f"메트릭 추출 완료: {len(metrics)}개 메트릭")
        
        return metrics
        
    except Exception as e:
        logger.error(f"메트릭 추출 실패: {e}")
        # 기본 메트릭 반환
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_trade_return": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
        }


class MockDataProvider:
    """Mock 데이터 프로바이더
    
    이미 로딩된 데이터를 사용하는 데이터 프로바이더입니다.
    실제 네트워크 호출 없이 메모리의 데이터를 반환합니다.
    """
    
    def __init__(self, data: pl.DataFrame):
        """Mock 데이터 프로바이더 초기화
        
        Args:
            data: 사용할 시장 데이터
        """
        self.data = data
        logger.debug(f"Mock 데이터 프로바이더 초기화: {len(data):,}행")
    
    async def get_data(self, **kwargs) -> pl.DataFrame:
        """데이터 조회 (비동기)
        
        실제로는 이미 로딩된 데이터를 그대로 반환합니다.
        
        Returns:
            시장 데이터
        """
        logger.debug("Mock 데이터 반환")
        return self.data
    
    def get_data_sync(self, **kwargs) -> pl.DataFrame:
        """데이터 조회 (동기)
        
        Returns:
            시장 데이터
        """
        logger.debug("Mock 데이터 반환 (동기)")
        return self.data


class WorkerProcessManager:
    """워커 프로세스 관리자
    
    워커 프로세스의 생성, 모니터링, 정리를 담당합니다.
    """
    
    def __init__(self):
        """워커 프로세스 관리자 초기화"""
        self.active_workers = {}
        self.completed_workers = {}
        self.failed_workers = {}
        
    def create_worker_config(
        self,
        params: Dict[str, Any],
        shared_data_path: str,
        base_config: 'BacktestConfig',
        strategy_class: Type
    ) -> Dict[str, Any]:
        """워커 설정 생성
        
        Args:
            params: 전략 파라메터
            shared_data_path: 공유 데이터 경로
            base_config: 백테스트 설정
            strategy_class: 전략 클래스
            
        Returns:
            워커 설정 딕셔너리
        """
        worker_config = {
            "params": params,
            "shared_data_path": shared_data_path,
            "base_config": base_config,
            "strategy_class": strategy_class,
            "worker_id": None,  # 실행 시 할당
            "created_at": datetime.now(),
            "status": "created"
        }
        
        return worker_config
    
    def estimate_worker_resource_usage(
        self,
        data_size_mb: float,
        num_workers: int
    ) -> Dict[str, Any]:
        """워커 리소스 사용량 추정
        
        Args:
            data_size_mb: 데이터 크기 (MB)
            num_workers: 워커 개수
            
        Returns:
            리소스 사용량 추정 정보
        """
        # 워커당 메모리 사용량 추정
        base_memory_mb = 50  # 기본 Python 프로세스 메모리
        data_memory_mb = data_size_mb * 1.2  # 데이터 + 처리 오버헤드
        worker_memory_mb = base_memory_mb + data_memory_mb
        
        total_memory_mb = worker_memory_mb * num_workers
        
        return {
            "worker_memory_mb": worker_memory_mb,
            "total_memory_mb": total_memory_mb,
            "total_memory_gb": total_memory_mb / 1024,
            "data_size_mb": data_size_mb,
            "num_workers": num_workers,
            "recommendation": self._get_resource_recommendation(total_memory_mb)
        }
    
    def _get_resource_recommendation(self, total_memory_mb: float) -> str:
        """리소스 사용량 기반 추천사항"""
        available_memory_gb = self._get_available_memory_gb()
        required_memory_gb = total_memory_mb / 1024
        
        if required_memory_gb > available_memory_gb * 0.8:
            return f"경고: 메모리 부족 위험. 필요: {required_memory_gb:.1f}GB, 사용가능: {available_memory_gb:.1f}GB"
        elif required_memory_gb > available_memory_gb * 0.6:
            return f"주의: 메모리 사용량 높음. 필요: {required_memory_gb:.1f}GB, 사용가능: {available_memory_gb:.1f}GB"
        else:
            return f"안전: 메모리 사용량 적절. 필요: {required_memory_gb:.1f}GB, 사용가능: {available_memory_gb:.1f}GB"
    
    def _get_available_memory_gb(self) -> float:
        """사용 가능한 메모리 조회 (GB)"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # psutil이 없는 경우 기본값 반환
            return 8.0  # 8GB 가정


# 편의 함수들
def create_worker_function_wrapper(
    shared_data_path: str,
    base_config: 'BacktestConfig',
    strategy_class: Type
):
    """워커 함수 래퍼 생성 (멀티프로세싱용)
    
    멀티프로세싱에서 사용하기 쉽도록 파라메터만 전달하면 되는 함수를 생성합니다.
    
    Args:
        shared_data_path: 공유 데이터 경로
        base_config: 백테스트 설정
        strategy_class: 전략 클래스
        
    Returns:
        파라메터만 받는 워커 함수
    """
    def worker_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
        return backtest_worker_function(
            params=params,
            shared_data_path=shared_data_path,
            base_config=base_config,
            strategy_class=strategy_class
        )
    
    return worker_wrapper


def validate_worker_environment() -> Dict[str, Any]:
    """워커 실행 환경 검증
    
    Returns:
        환경 검증 결과
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    try:
        # Python 버전 확인
        python_version = sys.version_info
        validation_result["info"]["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            validation_result["errors"].append("Python 3.8 이상 필요")
            validation_result["is_valid"] = False
        
        # 필수 모듈 확인
        required_modules = ["polars", "multiprocessing", "concurrent.futures"]
        missing_modules = []
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        if missing_modules:
            validation_result["errors"].append(f"필수 모듈 누락: {missing_modules}")
            validation_result["is_valid"] = False
        
        # 메모리 확인
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            validation_result["info"]["available_memory_gb"] = available_gb
            
            if available_gb < 2.0:
                validation_result["warnings"].append(f"사용 가능한 메모리가 부족: {available_gb:.1f}GB")
            
        except ImportError:
            validation_result["warnings"].append("psutil 모듈이 없어 메모리 확인 불가")
        
        # CPU 코어 수 확인
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        validation_result["info"]["cpu_cores"] = cpu_count
        
        if cpu_count < 2:
            validation_result["warnings"].append("CPU 코어가 2개 미만: 멀티프로세싱 효과 제한적")
        
    except Exception as e:
        validation_result["errors"].append(f"환경 검증 중 오류: {e}")
        validation_result["is_valid"] = False
    
    return validation_result 