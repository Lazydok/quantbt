"""
SimpleMonitor - Backtest Performance Statistics Collector

A class that collects backtest results and calculates basic performance statistics.
"""

import threading
from typing import Dict, List, Any, Optional


class SimpleMonitor:
    """Backtest performance statistics collection and management class
    
    Main features:
    - Record backtest results
    - Track best performance
    - Calculate basic statistics
    - Ensure thread safety
    """
    
    def __init__(self):
        """Initialize SimpleMonitor"""
        self.results: List[Dict] = []
        self.best_performance: Optional[Dict] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Statistics cache
        self._stats_cache: Optional[Dict] = None
        self._cache_valid = False
    
    def record_result(self, result: Dict):
        """Record backtest result
        
        Args:
            result: Backtest result dictionary
        """
        with self._lock:
            self.results.append(result.copy())
            self._cache_valid = False
            
            # Update best performance (based on Sharpe ratio)
            sharpe_ratio = result.get('sharpe_ratio', float('-inf'))
            if self.best_performance is None or sharpe_ratio > self.best_performance.get('sharpe_ratio', float('-inf')):
                self.best_performance = result.copy()
    
    def get_best_performance(self) -> Optional[Dict]:
        """Return best performance
        
        Returns:
            Dict: Best performance dictionary or None
        """
        with self._lock:
            return self.best_performance.copy() if self.best_performance else None
    
    def get_statistics(self) -> Dict:
        """Return statistics summary
        
        Returns:
            Dict: Statistics summary
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
            
            # Calculate statistics
            total_results = len(self.results)
            success_count = sum(1 for r in self.results if r.get('success', True))
            failure_count = total_results - success_count
            success_rate = success_count / total_results if total_results > 0 else 0.0
            
            # Calculate averages for successful results only
            successful_results = [r for r in self.results if r.get('success', True)]
            
            if successful_results:
                avg_sharpe_ratio = sum(r.get('sharpe_ratio', 0) for r in successful_results) / len(successful_results)
                avg_return = sum(r.get('total_return', 0) for r in successful_results) / len(successful_results)
            else:
                avg_sharpe_ratio = 0.0
                avg_return = 0.0
            
            # Average execution time (for all results)
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
        """Return formatted statistics summary
        
        Returns:
            str: Formatted statistics summary
        """
        stats = self.get_statistics()
        best = self.get_best_performance()
        
        summary = f"""📊 Current Performance:
   Total Results: {stats['total_results']}
   Success Rate: {stats['success_rate']:.1%}
   Average Sharpe Ratio: {stats['avg_sharpe_ratio']:.4f}
   Average Return: {stats['avg_return']:.4f}
   Average Execution Time: {stats['avg_execution_time']:.2f}s"""
        
        if best:
            summary += f"""
   Best Sharpe Ratio: {best.get('sharpe_ratio', 0):.4f}"""
            if 'params' in best:
                summary += f" (Parameters: {best['params']})"
        
        if stats['failure_count'] > 0:
            summary += f"""
   Failures: {stats['failure_count']}"""
        
        return summary 