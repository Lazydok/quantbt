"""
Ray Parameter Optimizer

모든 Ray 컴포넌트를 통합하여 파라메터 최적화를 수행하는 통합 API
Phase 5: SimpleSMAStrategy Ray 기반 실행 지원
"""

import ray
import asyncio
import time
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import logging
from itertools import product

from .cluster_manager import RayClusterManager
from .data_manager import RayDataManager
from .backtest_actor import BacktestScheduler, BacktestMonitor
from .result_aggregator import RayResultAggregator
from .quantbt_engine_adapter import QuantBTEngineAdapter


logger = logging.getLogger(__name__)


class RayParameterOptimizer:
    """Ray 기반 파라메터 최적화 통합 시스템
    
    모든 Ray 컴포넌트를 하나의 API로 통합하여 
    SimpleSMAStrategy 등의 전략을 분산 환경에서 최적화합니다.
    """
    
    def __init__(self, strategy_class: Type, base_config: Any, cluster_config: Optional[Dict] = None):
        """RayParameterOptimizer 초기화
        
        Args:
            strategy_class: 최적화할 전략 클래스 (예: SimpleSMAStrategy)
            base_config: 백테스트 기본 설정 (BacktestConfig)
            cluster_config: Ray 클러스터 설정 (선택사항)
        """
        self.strategy_class = strategy_class
        self.base_config = base_config
        self.cluster_config = cluster_config or {}
        
        # Ray 컴포넌트들
        self.cluster_manager = None
        self.data_manager = None
        self.scheduler = None
        self.aggregator = None
        self.monitor = None
        self.engine_adapter = None
        
        # 최적화 상태
        self.optimization_status = {
            'initialized': False,
            'running': False,
            'completed': False,
            'total_combinations': 0,
            'completed_combinations': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 결과 저장
        self.optimization_results = None
        
        logger.info(f"RayParameterOptimizer 초기화: {strategy_class.__name__}")
    
    async def initialize_components(self) -> bool:
        """모든 Ray 컴포넌트 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("Ray 컴포넌트 초기화 시작")
            
            # 현재는 간단한 버전으로 구현 (Ray Actor 없이)
            # TODO: 실제 Ray Actor 통합
            
            # 1. QuantBT 엔진 어댑터 생성 (로컬 실행)
            self.engine_adapter = QuantBTEngineAdapter(self.base_config)
            
            # 2. 결과 저장용 리스트 (임시)
            self.results_storage = []
            
            self.optimization_status['initialized'] = True
            logger.info("Ray 컴포넌트 초기화 완료 (간소화 버전)")
            return True
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            return False
    
    async def optimize_parameters(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """파라메터 그리드 최적화 실행
        
        Args:
            param_grid: 파라메터 그리드 (예: {'buy_sma': [10, 15], 'sell_sma': [25, 30]})
            
        Returns:
            Dict: 최적화 결과
        """
        logger.info("파라메터 최적화 시작")
        
        # 1. 컴포넌트 초기화
        if not self.optimization_status['initialized']:
            await self.initialize_components()
        
        # 2. 파라메터 조합 생성
        param_combinations = self._generate_parameter_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        self.optimization_status.update({
            'running': True,
            'total_combinations': total_combinations,
            'start_time': datetime.now()
        })
        
        logger.info(f"총 {total_combinations}개 파라메터 조합 최적화 시작")
        
        try:
            # 3. 백테스트 작업 생성 및 실행
            results = []
            for i, params in enumerate(param_combinations):
                # 각 파라메터 조합으로 백테스트 실행
                result = await self._execute_single_backtest(params)
                results.append(result)
                
                # 결과 저장 (임시)
                self.results_storage.append(result)
                
                # 진행 상황 업데이트
                self.optimization_status['completed_combinations'] = i + 1
                
                # 진행률 출력
                progress = (i + 1) / total_combinations * 100
                if (i + 1) % max(1, total_combinations // 10) == 0:
                    logger.info(f"진행률: {progress:.1f}% ({i + 1}/{total_combinations})")
            
            # 4. 최적 파라메터 추출 (간소화 버전)
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                # 샤프 비율 기준으로 최고 성과 찾기
                best_strategy = max(successful_results, 
                                  key=lambda x: x.get('result', {}).get('sharpe_ratio', -999))
            else:
                best_strategy = None
            
            # 5. 최종 결과 구성
            self.optimization_results = {
                'best_params': best_strategy.get('params', {}) if best_strategy else {},
                'best_sharpe_ratio': best_strategy.get('result', {}).get('sharpe_ratio', 0.0) if best_strategy else 0.0,
                'best_total_return': best_strategy.get('result', {}).get('total_return', 0.0) if best_strategy else 0.0,
                'execution_time': (datetime.now() - self.optimization_status['start_time']).total_seconds(),
                'total_combinations': total_combinations,
                'successful_combinations': len([r for r in results if r.get('success', False)]),
                'optimization_report': {'summary': 'Simplified version'},  # 임시
                'performance_comparison': await self._generate_performance_comparison(results)
            }
            
            self.optimization_status.update({
                'running': False,
                'completed': True,
                'end_time': datetime.now()
            })
            
            logger.info("파라메터 최적화 완료")
            return self.optimization_results
            
        except Exception as e:
            logger.error(f"최적화 실행 중 오류: {e}")
            self.optimization_status['running'] = False
            raise
    
    async def _execute_single_backtest(self, params: Dict) -> Dict:
        """단일 파라메터 조합으로 백테스트 실행
        
        Args:
            params: 파라메터 딕셔너리
            
        Returns:
            Dict: 백테스트 결과
        """
        try:
            start_time = time.time()
            
            # QuantBT 엔진 어댑터를 통해 백테스트 실행
            result = self.engine_adapter.execute_backtest(params, self.strategy_class)
            
            execution_time = time.time() - start_time
            
            return {
                'params': params,
                'result': result,
                'execution_time': execution_time,
                'success': True,
                'worker_id': 'main'  # 단일 스레드 실행
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패 (params={params}): {e}")
            return {
                'params': params,
                'result': {},
                'execution_time': 0.0,
                'success': False,
                'error': str(e),
                'worker_id': 'main'
            }
    
    def _generate_parameter_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """파라메터 조합 생성
        
        Args:
            param_grid: 파라메터 그리드
            
        Returns:
            List[Dict]: 파라메터 조합 리스트
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    async def _generate_performance_comparison(self, results: List[Dict]) -> Dict:
        """성능 비교 데이터 생성
        
        Args:
            results: 백테스트 결과 리스트
            
        Returns:
            Dict: 성능 비교 데이터
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': '성공한 백테스트 결과가 없습니다'}
        
        # 기본 통계
        sharpe_ratios = [r['result'].get('sharpe_ratio', 0) for r in successful_results]
        total_returns = [r['result'].get('total_return', 0) for r in successful_results]
        execution_times = [r.get('execution_time', 0) for r in successful_results]
        
        return {
            'total_backtests': len(results),
            'successful_backtests': len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'sharpe_ratio_stats': {
                'mean': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                'max': max(sharpe_ratios) if sharpe_ratios else 0,
                'min': min(sharpe_ratios) if sharpe_ratios else 0
            },
            'return_stats': {
                'mean': sum(total_returns) / len(total_returns) if total_returns else 0,
                'max': max(total_returns) if total_returns else 0,
                'min': min(total_returns) if total_returns else 0
            },
            'execution_time_stats': {
                'total': sum(execution_times),
                'average': sum(execution_times) / len(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0,
                'min': min(execution_times) if execution_times else 0
            }
        }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """최적화 진행 상황 조회
        
        Returns:
            Dict: 현재 최적화 상태
        """
        status = self.optimization_status.copy()
        
        # 진행률 계산
        if status['total_combinations'] > 0:
            status['progress_percentage'] = (
                status['completed_combinations'] / status['total_combinations'] * 100
            )
        else:
            status['progress_percentage'] = 0.0
        
        # 경과 시간 계산
        if status['start_time']:
            if status['end_time']:
                status['elapsed_time'] = (status['end_time'] - status['start_time']).total_seconds()
            else:
                status['elapsed_time'] = (datetime.now() - status['start_time']).total_seconds()
        else:
            status['elapsed_time'] = 0.0
        
        return status
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """최적 파라메터 반환
        
        Returns:
            Dict: 최적 파라메터 및 성과
        """
        if not self.optimization_results:
            return {'error': '최적화가 완료되지 않았습니다'}
        
        return {
            'best_params': self.optimization_results['best_params'],
            'best_sharpe_ratio': self.optimization_results['best_sharpe_ratio'],
            'best_total_return': self.optimization_results['best_total_return']
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """성능 비교 보고서 생성
        
        Returns:
            Dict: 성능 보고서
        """
        if not self.optimization_results:
            return {'error': '최적화가 완료되지 않았습니다'}
        
        return {
            'optimization_summary': {
                'total_combinations': self.optimization_results['total_combinations'],
                'successful_combinations': self.optimization_results['successful_combinations'],
                'execution_time': self.optimization_results['execution_time'],
                'best_performance': {
                    'params': self.optimization_results['best_params'],
                    'sharpe_ratio': self.optimization_results['best_sharpe_ratio'],
                    'total_return': self.optimization_results['best_total_return']
                }
            },
            'performance_comparison': self.optimization_results['performance_comparison'],
            'optimization_report': self.optimization_results['optimization_report']
        }
    
    def shutdown(self):
        """Ray 컴포넌트 정리"""
        logger.info("RayParameterOptimizer 종료")
        # Ray 컴포넌트들은 자동으로 정리됨 