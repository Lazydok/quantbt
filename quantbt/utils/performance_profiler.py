"""
성능 측정 시스템

각 모듈별 실행 시간을 정밀하게 측정하여 실제 병목점을 파악하는 도구
"""

import time
import functools
from collections import defaultdict
from typing import Dict, Any, List, Optional
import threading


class PerformanceProfiler:
    """성능 측정 프로파일러"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self._lock = threading.Lock()  # 스레드 안전성
        
    def measure(self, category: str):
        """성능 측정 데코레이터"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    with self._lock:
                        self.timings[category].append(duration)
                        self.call_counts[category] += 1
                        self.total_times[category] += duration
            return wrapper
        return decorator
    
    def time_block(self, category: str):
        """컨텍스트 매니저로 코드 블록 시간 측정"""
        return _TimeBlockContext(self, category)
    
    def add_timing(self, category: str, duration: float):
        """직접 시간 추가"""
        with self._lock:
            self.timings[category].append(duration)
            self.call_counts[category] += 1
            self.total_times[category] += duration
    
    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """성능 리포트 생성"""
        with self._lock:
            report = {}
            for category in self.timings:
                times = self.timings[category]
                if times:
                    report[category] = {
                        'total_time': self.total_times[category],
                        'call_count': self.call_counts[category],
                        'avg_time': self.total_times[category] / self.call_counts[category],
                        'min_time': min(times),
                        'max_time': max(times),
                        'total_percentage': 0  # 나중에 계산
                    }
            
            # 전체 시간 대비 퍼센트 계산
            total_time = sum(self.total_times.values())
            if total_time > 0:
                for category in report:
                    report[category]['total_percentage'] = (
                        report[category]['total_time'] / total_time * 100
                    )
            
            return report
    
    def print_report(self, title: str = "성능 측정 결과", data_size: Optional[int] = None):
        """성능 측정 결과 출력"""
        report = self.get_report()
        
        size_info = f" (데이터: {data_size:,}건)" if data_size else ""
        print(f"\n📊 {title}{size_info}")
        print("=" * 85)
        print(f"{'모듈':<25} {'총 시간(s)':<12} {'호출 횟수':<12} {'평균 시간(ms)':<15} {'비율(%)':<10}")
        print("-" * 85)
        
        # 시간 순으로 정렬
        sorted_report = sorted(report.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for category, stats in sorted_report:
            avg_ms = stats['avg_time'] * 1000  # 밀리초로 변환
            print(f"{category:<25} {stats['total_time']:<12.3f} {stats['call_count']:<12} "
                  f"{avg_ms:<15.6f} {stats['total_percentage']:<10.1f}")
        
        total_time = sum(stats['total_time'] for stats in report.values())
        print("-" * 85)
        print(f"{'총합':<25} {total_time:<12.3f}")
        print("=" * 85)
        
        return report
    
    def reset(self):
        """측정 데이터 초기화"""
        with self._lock:
            self.timings.clear()
            self.call_counts.clear()
            self.total_times.clear()
    
    def get_summary_stats(self) -> Dict[str, float]:
        """요약 통계 반환"""
        with self._lock:
            return {
                'total_time': sum(self.total_times.values()),
                'total_calls': sum(self.call_counts.values()),
                'categories': len(self.timings)
            }


class _TimeBlockContext:
    """시간 측정 컨텍스트 매니저"""
    
    def __init__(self, profiler: PerformanceProfiler, category: str):
        self.profiler = profiler
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.profiler.add_timing(self.category, duration)


# 전역 프로파일러 인스턴스 (싱글톤 패턴)
_global_profiler = None

def get_global_profiler() -> PerformanceProfiler:
    """전역 프로파일러 인스턴스 반환"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def reset_global_profiler():
    """전역 프로파일러 리셋"""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()

def performance_measure(category: str):
    """전역 프로파일러를 사용한 성능 측정 데코레이터"""
    return get_global_profiler().measure(category)

def time_block(category: str):
    """전역 프로파일러를 사용한 시간 측정 컨텍스트"""
    return get_global_profiler().time_block(category)

def print_global_report(title: str = "성능 측정 결과", data_size: Optional[int] = None):
    """전역 프로파일러 결과 출력"""
    return get_global_profiler().print_report(title, data_size) 