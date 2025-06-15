"""
SimpleMonitor - 백테스트 성능 통계 수집기

백테스트 결과를 수집하고 기본적인 성능 통계를 계산하는 클래스입니다.
"""

import threading
from typing import Dict, List, Any, Optional


class SimpleMonitor:
    """백테스트 성능 통계 수집 및 관리 클래스
    
    주요 기능:
    - 백테스트 결과 기록
    - 최고 성과 추적
    - 기본 통계 계산
    - 스레드 안전성 보장
    """
    
    def __init__(self):
        """SimpleMonitor 초기화"""
        self.results: List[Dict] = []
        self.best_performance: Optional[Dict] = None
        
        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()
        
        # 통계 캐시
        self._stats_cache: Optional[Dict] = None
        self._cache_valid = False
    
    def record_result(self, result: Dict):
        """백테스트 결과 기록
        
        Args:
            result: 백테스트 결과 딕셔너리
        """
        with self._lock:
            self.results.append(result.copy())
            self._cache_valid = False
            
            # 최고 성과 업데이트 (샤프 비율 기준)
            sharpe_ratio = result.get('sharpe_ratio', float('-inf'))
            if self.best_performance is None or sharpe_ratio > self.best_performance.get('sharpe_ratio', float('-inf')):
                self.best_performance = result.copy()
    
    def get_best_performance(self) -> Optional[Dict]:
        """최고 성과 반환
        
        Returns:
            Dict: 최고 성과 딕셔너리 또는 None
        """
        with self._lock:
            return self.best_performance.copy() if self.best_performance else None
    
    def get_statistics(self) -> Dict:
        """통계 요약 반환
        
        Returns:
            Dict: 통계 요약
        """
        with self._lock:
            if self._cache_valid and self._stats_cache:
                return self._stats_cache.copy()
                
            if not self.results:
                return {
                    'total_results': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'success_rate': 0.0,
                    'avg_sharpe_ratio': 0.0,
                    'avg_return': 0.0,
                    'avg_execution_time': 0.0
                }
            
            # 통계 계산
            total_results = len(self.results)
            success_count = sum(1 for r in self.results if r.get('success', True))
            failure_count = total_results - success_count
            success_rate = success_count / total_results if total_results > 0 else 0.0
            
            # 성공한 결과만 대상으로 평균 계산
            successful_results = [r for r in self.results if r.get('success', True)]
            
            if successful_results:
                avg_sharpe_ratio = sum(r.get('sharpe_ratio', 0) for r in successful_results) / len(successful_results)
                avg_return = sum(r.get('total_return', 0) for r in successful_results) / len(successful_results)
            else:
                avg_sharpe_ratio = 0.0
                avg_return = 0.0
            
            # 실행 시간 평균 (모든 결과 대상)
            avg_execution_time = sum(r.get('execution_time', 0) for r in self.results) / total_results
            
            self._stats_cache = {
                'total_results': total_results,
                'success_count': success_count,
                'failure_count': failure_count,
                'success_rate': success_rate,
                'avg_sharpe_ratio': avg_sharpe_ratio,
                'avg_return': avg_return,
                'avg_execution_time': avg_execution_time
            }
            
            self._cache_valid = True
            return self._stats_cache.copy()
    
    def format_summary(self) -> str:
        """통계 요약 문자열 반환
        
        Returns:
            str: 포맷된 통계 요약
        """
        stats = self.get_statistics()
        best = self.get_best_performance()
        
        summary = f"""📊 현재 성과:
   총 결과: {stats['total_results']}개
   성공률: {stats['success_rate']:.1%}
   평균 샤프비율: {stats['avg_sharpe_ratio']:.4f}
   평균 수익률: {stats['avg_return']:.4f}
   평균 실행시간: {stats['avg_execution_time']:.2f}초"""
        
        if best:
            summary += f"""
   최고 샤프비율: {best.get('sharpe_ratio', 0):.4f}"""
            if 'params' in best:
                summary += f" (파라메터: {best['params']})"
        
        if stats['failure_count'] > 0:
            summary += f"""
   실패: {stats['failure_count']}개"""
        
        return summary 