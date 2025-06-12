"""
결과 집계 시스템

Phase 1 멀티프로세싱 백테스팅 최적화를 위한 결과 집계 및 분석 시스템
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class ResultAggregator:
    """백테스트 결과 집계 및 분석 시스템
    
    여러 워커 프로세스로부터 받은 백테스트 결과를 수집하고,
    성과 메트릭을 기준으로 분석 및 정렬하는 시스템입니다.
    
    Example:
        >>> aggregator = ResultAggregator()
        >>> aggregator.add_result(worker_result)
        >>> best_results = aggregator.get_best_results(metric="sharpe_ratio", top_n=10)
        >>> summary = aggregator.generate_summary_report()
    """
    
    def __init__(self):
        """결과 집계 시스템 초기화"""
        self.results: List[Dict[str, Any]] = []
        self.creation_time = datetime.now()
        self.last_update_time = None
        
        # 통계 캐시 (성능 최적화용)
        self._stats_cache = {}
        self._cache_dirty = True
        
        logger.debug("결과 집계 시스템 초기화 완료")
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """백테스트 결과 추가
        
        Args:
            result: 워커 프로세스로부터 받은 백테스트 결과
                   {"parameters": {...}, "success": bool, "metrics": {...}, ...}
        
        Raises:
            ValueError: 결과 형식이 올바르지 않은 경우
        """
        # 기본 필드 검증
        required_fields = ["parameters", "success"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"결과에 필수 필드 '{field}'가 누락되었습니다")
        
        # 성공한 결과의 경우 메트릭 확인
        if result["success"] and "metrics" not in result:
            logger.warning("성공한 결과에 메트릭이 없습니다")
        
        # 결과 추가 (복사본 생성하여 원본 보호)
        result_copy = result.copy()
        result_copy["added_at"] = datetime.now()
        
        self.results.append(result_copy)
        self.last_update_time = datetime.now()
        self._cache_dirty = True
        
        logger.debug(f"결과 추가: {len(self.results)}번째, 파라메터: {result['parameters']}")
    
    def add_results_batch(self, results: List[Dict[str, Any]]) -> None:
        """배치로 여러 결과 추가 (성능 최적화)
        
        Args:
            results: 백테스트 결과 리스트
        """
        for result in results:
            self.add_result(result)
        
        logger.info(f"배치 결과 추가 완료: {len(results)}개")
    
    def get_total_results_count(self) -> int:
        """전체 결과 개수 조회"""
        return len(self.results)
    
    def get_successful_results_count(self) -> int:
        """성공한 결과 개수 조회"""
        return sum(1 for r in self.results if r["success"])
    
    def get_failed_results_count(self) -> int:
        """실패한 결과 개수 조회"""
        return sum(1 for r in self.results if not r["success"])
    
    def get_success_rate(self) -> float:
        """성공률 조회"""
        total = self.get_total_results_count()
        if total == 0:
            return 0.0
        return self.get_successful_results_count() / total
    
    def get_successful_results(self) -> List[Dict[str, Any]]:
        """성공한 결과만 조회"""
        return [r for r in self.results if r["success"]]
    
    def get_failed_results(self) -> List[Dict[str, Any]]:
        """실패한 결과만 조회"""
        return [r for r in self.results if not r["success"]]
    
    def get_best_results(
        self, 
        metric: str = "sharpe_ratio", 
        top_n: Optional[int] = None,
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """성과 메트릭 기준으로 최고 결과 조회
        
        Args:
            metric: 정렬 기준 메트릭명 (예: "sharpe_ratio", "total_return")
            top_n: 상위 N개 결과 (None이면 전체)
            ascending: 오름차순 정렬 여부 (기본: 내림차순)
            
        Returns:
            정렬된 성공 결과 리스트
        """
        successful_results = self.get_successful_results()
        
        if not successful_results:
            return []
        
        # 메트릭이 존재하는 결과만 필터링
        filtered_results = []
        for result in successful_results:
            if "metrics" in result and metric in result["metrics"]:
                metric_value = result["metrics"][metric]
                if metric_value is not None:
                    filtered_results.append(result)
        
        if not filtered_results:
            logger.warning(f"메트릭 '{metric}'을 가진 결과가 없습니다")
            return []
        
        # 메트릭 기준 정렬
        try:
            sorted_results = sorted(
                filtered_results,
                key=lambda x: x["metrics"][metric],
                reverse=not ascending
            )
            
            # 상위 N개만 반환
            if top_n is not None:
                sorted_results = sorted_results[:top_n]
            
            logger.debug(f"메트릭 '{metric}' 기준 정렬 완료: {len(sorted_results)}개 결과")
            
            return sorted_results
            
        except (KeyError, TypeError) as e:
            logger.error(f"메트릭 정렬 실패: {e}")
            return []
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """성능 통계 조회
        
        Returns:
            성능 통계 딕셔너리
        """
        if self._cache_dirty:
            self._update_stats_cache()
        
        return self._stats_cache.copy()
    
    def _update_stats_cache(self) -> None:
        """통계 캐시 업데이트"""
        stats = {
            "total_results": self.get_total_results_count(),
            "successful_results": self.get_successful_results_count(),
            "failed_results": self.get_failed_results_count(),
            "success_rate": self.get_success_rate(),
        }
        
        # 실행 시간 통계 (모든 결과)
        execution_times = [r.get("execution_time", 0) for r in self.results if "execution_time" in r]
        if execution_times:
            stats.update({
                "avg_execution_time": statistics.mean(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "total_execution_time": sum(execution_times)
            })
        
        # 메트릭 통계 (성공한 결과만)
        successful_results = self.get_successful_results()
        if successful_results:
            self._calculate_metrics_statistics(stats, successful_results)
        
        self._stats_cache = stats
        self._cache_dirty = False
        
        logger.debug("통계 캐시 업데이트 완료")
    
    def _calculate_metrics_statistics(self, stats: Dict[str, Any], successful_results: List[Dict[str, Any]]) -> None:
        """메트릭 통계 계산"""
        # 모든 메트릭 수집
        all_metrics = set()
        for result in successful_results:
            if "metrics" in result:
                all_metrics.update(result["metrics"].keys())
        
        # 각 메트릭별 통계 계산
        for metric_name in all_metrics:
            values = []
            for result in successful_results:
                if "metrics" in result and metric_name in result["metrics"]:
                    value = result["metrics"][metric_name]
                    if value is not None and isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                stats[f"avg_{metric_name}"] = statistics.mean(values)
                stats[f"best_{metric_name}"] = max(values)
                stats[f"worst_{metric_name}"] = min(values)
                if len(values) > 1:
                    stats[f"std_{metric_name}"] = statistics.stdev(values)
    
    def analyze_parameter_impact(self, metric: str) -> Dict[str, Dict[Any, float]]:
        """파라메터별 성과 영향 분석
        
        각 파라메터 값이 성과 메트릭에 미치는 영향을 분석합니다.
        
        Args:
            metric: 분석할 메트릭명
            
        Returns:
            파라메터별 평균 성과 딕셔너리
            {"param_name": {param_value: avg_metric_value, ...}, ...}
        """
        successful_results = self.get_successful_results()
        if not successful_results:
            return {}
        
        # 파라메터별 값 수집
        param_impacts = defaultdict(lambda: defaultdict(list))
        
        for result in successful_results:
            if "metrics" not in result or metric not in result["metrics"]:
                continue
            
            metric_value = result["metrics"][metric]
            if metric_value is None:
                continue
            
            parameters = result["parameters"]
            for param_name, param_value in parameters.items():
                param_impacts[param_name][param_value].append(metric_value)
        
        # 평균 계산
        impact_analysis = {}
        for param_name, param_values in param_impacts.items():
            impact_analysis[param_name] = {}
            for param_value, metric_values in param_values.items():
                if metric_values:
                    impact_analysis[param_name][param_value] = statistics.mean(metric_values)
        
        logger.debug(f"파라메터 영향 분석 완료: {len(impact_analysis)}개 파라메터")
        
        return impact_analysis
    
    def find_optimal_parameter_ranges(
        self,
        metric: str,
        top_percentile: float = 0.2
    ) -> Dict[str, Dict[str, Any]]:
        """최적 파라메터 범위 찾기
        
        상위 성과를 보인 결과들의 파라메터 범위를 분석합니다.
        
        Args:
            metric: 분석 기준 메트릭
            top_percentile: 상위 몇 % 결과를 분석할지 (0.2 = 상위 20%)
            
        Returns:
            파라메터별 최적 범위 정보
        """
        # 상위 성과 결과 추출
        best_results = self.get_best_results(metric=metric)
        if not best_results:
            return {}
        
        top_count = max(1, int(len(best_results) * top_percentile))
        top_results = best_results[:top_count]
        
        # 파라메터별 값 범위 분석
        param_ranges = {}
        
        # 모든 파라메터 이름 수집
        all_params = set()
        for result in top_results:
            all_params.update(result["parameters"].keys())
        
        for param_name in all_params:
            values = []
            for result in top_results:
                if param_name in result["parameters"]:
                    value = result["parameters"][param_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                param_ranges[param_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "count": len(values),
                    "values": values
                }
                
                if len(values) > 1:
                    param_ranges[param_name]["std"] = statistics.stdev(values)
        
        logger.info(f"최적 파라메터 범위 분석 완료: 상위 {top_percentile*100}% ({top_count}개) 결과 기준")
        
        return param_ranges
    
    def generate_heatmap_data(
        self,
        x_param: str,
        y_param: str,
        metric: str
    ) -> Dict[str, Any]:
        """히트맵 시각화를 위한 데이터 생성
        
        Args:
            x_param: X축 파라메터명
            y_param: Y축 파라메터명  
            metric: 색상 기준 메트릭명
            
        Returns:
            히트맵 데이터 딕셔너리
        """
        successful_results = self.get_successful_results()
        if not successful_results:
            return {}
        
        # 파라메터 값 수집
        x_values = set()
        y_values = set()
        data_points = []
        
        for result in successful_results:
            params = result["parameters"]
            if x_param not in params or y_param not in params:
                continue
            
            if "metrics" not in result or metric not in result["metrics"]:
                continue
            
            x_val = params[x_param]
            y_val = params[y_param]
            metric_val = result["metrics"][metric]
            
            if metric_val is not None:
                x_values.add(x_val)
                y_values.add(y_val)
                data_points.append((x_val, y_val, metric_val))
        
        # 정렬된 고유값 생성
        x_sorted = sorted(x_values)
        y_sorted = sorted(y_values)
        
        # 2D 그리드 생성
        z_values = []
        for y_val in y_sorted:
            row = []
            for x_val in x_sorted:
                # 해당 좌표의 메트릭 값 찾기
                metric_val = None
                for x, y, z in data_points:
                    if x == x_val and y == y_val:
                        metric_val = z
                        break
                row.append(metric_val)
            z_values.append(row)
        
        heatmap_data = {
            "x_values": x_sorted,
            "y_values": y_sorted,
            "z_values": z_values,
            "x_label": x_param,
            "y_label": y_param,
            "metric_label": metric,
            "data_points_count": len(data_points)
        }
        
        logger.debug(f"히트맵 데이터 생성: {len(x_sorted)}x{len(y_sorted)} 그리드, {len(data_points)}개 데이터포인트")
        
        return heatmap_data
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """종합 요약 보고서 생성
        
        Returns:
            요약 보고서 딕셔너리
        """
        stats = self.get_performance_statistics()
        successful_results = self.get_successful_results()
        
        summary = {
            "generation_time": datetime.now(),
            "total_results": stats["total_results"],
            "successful_results": stats["successful_results"],
            "failed_results": stats["failed_results"],
            "success_rate": stats["success_rate"],
        }
        
        # 실행 시간 정보
        if "avg_execution_time" in stats:
            summary["execution_time_stats"] = {
                "average": stats["avg_execution_time"],
                "minimum": stats["min_execution_time"],
                "maximum": stats["max_execution_time"],
                "total": stats["total_execution_time"]
            }
        
        # 메트릭 통계
        if successful_results:
            summary["metrics_statistics"] = self._generate_metrics_summary(stats, successful_results)
            
            # 최고 결과 (샤프 비율 기준)
            best_by_sharpe = self.get_best_results(metric="sharpe_ratio", top_n=1)
            if best_by_sharpe:
                summary["best_result"] = best_by_sharpe[0]
            
            # 상위 10개 결과
            summary["top_10_results"] = self.get_best_results(metric="sharpe_ratio", top_n=10)
        
        # 실패 원인 분석
        failed_results = self.get_failed_results()
        if failed_results:
            summary["failure_analysis"] = self._analyze_failures(failed_results)
        
        logger.info("종합 요약 보고서 생성 완료")
        
        return summary
    
    def _generate_metrics_summary(self, stats: Dict[str, Any], successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """메트릭 요약 생성"""
        metrics_summary = {}
        
        # 공통 메트릭들
        common_metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "total_trades"]
        
        for metric in common_metrics:
            if f"avg_{metric}" in stats:
                metrics_summary[metric] = {
                    "mean": stats[f"avg_{metric}"],
                    "max": stats[f"best_{metric}"],
                    "min": stats[f"worst_{metric}"]
                }
                
                if f"std_{metric}" in stats:
                    metrics_summary[metric]["std"] = stats[f"std_{metric}"]
        
        return metrics_summary
    
    def _analyze_failures(self, failed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """실패 원인 분석"""
        error_counts = defaultdict(int)
        
        for result in failed_results:
            error_msg = result.get("error", "Unknown error")
            # 에러 메시지에서 주요 키워드 추출
            if "파일" in error_msg or "File" in error_msg:
                error_counts["파일 관련 오류"] += 1
            elif "메모리" in error_msg or "Memory" in error_msg:
                error_counts["메모리 부족"] += 1
            elif "데이터" in error_msg or "Data" in error_msg:
                error_counts["데이터 오류"] += 1
            else:
                error_counts["기타 오류"] += 1
        
        return {
            "total_failures": len(failed_results),
            "error_categories": dict(error_counts),
            "failure_rate": len(failed_results) / (len(failed_results) + len(self.get_successful_results()))
        }
    
    def export_results(self, file_path: str) -> None:
        """결과를 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (.json)
        """
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_results": len(self.results),
            "results": self.results,
            "statistics": self.get_performance_statistics()
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"결과 내보내기 완료: {file_path} ({len(self.results)}개 결과)")
            
        except Exception as e:
            logger.error(f"결과 내보내기 실패: {e}")
            raise
    
    def import_results(self, file_path: str) -> None:
        """파일에서 결과 가져오기
        
        Args:
            file_path: 가져올 파일 경로 (.json)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if "results" not in import_data:
                raise ValueError("유효하지 않은 결과 파일 형식")
            
            # 기존 결과 초기화
            self.results.clear()
            self._cache_dirty = True
            
            # 결과 추가
            for result in import_data["results"]:
                self.results.append(result)
            
            self.last_update_time = datetime.now()
            
            logger.info(f"결과 가져오기 완료: {file_path} ({len(self.results)}개 결과)")
            
        except Exception as e:
            logger.error(f"결과 가져오기 실패: {e}")
            raise
    
    def clear_results(self) -> None:
        """모든 결과 삭제"""
        self.results.clear()
        self._stats_cache.clear()
        self._cache_dirty = True
        self.last_update_time = datetime.now()
        
        logger.info("모든 결과 삭제 완료")
    
    def get_result_by_parameters(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """특정 파라메터 조합의 결과 조회
        
        Args:
            parameters: 찾을 파라메터 딕셔너리
            
        Returns:
            일치하는 결과 (없으면 None)
        """
        for result in self.results:
            if result["parameters"] == parameters:
                return result
        
        return None
    
    def filter_results(self, **criteria) -> List[Dict[str, Any]]:
        """조건에 맞는 결과 필터링
        
        Args:
            **criteria: 필터링 조건들
                       success=True, metric_min={"sharpe_ratio": 1.0}, etc.
        
        Returns:
            필터링된 결과 리스트
        """
        filtered = []
        
        for result in self.results:
            match = True
            
            # 성공 여부 확인
            if "success" in criteria and result["success"] != criteria["success"]:
                match = False
                continue
            
            # 메트릭 최소값 확인
            if "metric_min" in criteria and result["success"]:
                for metric, min_val in criteria["metric_min"].items():
                    if "metrics" not in result or metric not in result["metrics"]:
                        match = False
                        break
                    if result["metrics"][metric] < min_val:
                        match = False
                        break
            
            # 파라메터 조건 확인
            if "parameters" in criteria:
                for param, value in criteria["parameters"].items():
                    if param not in result["parameters"] or result["parameters"][param] != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
        
        return filtered


# 편의 함수들
def create_result_aggregator() -> ResultAggregator:
    """결과 집계기 생성 편의 함수"""
    return ResultAggregator()


def merge_aggregators(aggregators: List[ResultAggregator]) -> ResultAggregator:
    """여러 집계기를 하나로 병합
    
    Args:
        aggregators: 병합할 집계기 리스트
        
    Returns:
        병합된 집계기
    """
    merged = ResultAggregator()
    
    for aggregator in aggregators:
        for result in aggregator.results:
            merged.add_result(result)
    
    logger.info(f"집계기 병합 완료: {len(aggregators)}개 -> 1개 ({merged.get_total_results_count()}개 결과)")
    
    return merged


def compare_parameter_sets(
    aggregator: ResultAggregator,
    param_sets: List[Dict[str, Any]],
    metric: str = "sharpe_ratio"
) -> Dict[str, Any]:
    """파라메터 세트 성과 비교
    
    Args:
        aggregator: 결과 집계기
        param_sets: 비교할 파라메터 세트 리스트
        metric: 비교 기준 메트릭
        
    Returns:
        비교 결과 딕셔너리
    """
    comparison = {
        "metric": metric,
        "comparisons": [],
        "best_set": None,
        "best_value": float('-inf')
    }
    
    for i, param_set in enumerate(param_sets):
        result = aggregator.get_result_by_parameters(param_set)
        
        comparison_item = {
            "index": i,
            "parameters": param_set,
            "found": result is not None
        }
        
        if result and result["success"]:
            metric_value = result["metrics"].get(metric)
            comparison_item["metric_value"] = metric_value
            
            if metric_value is not None and metric_value > comparison["best_value"]:
                comparison["best_value"] = metric_value
                comparison["best_set"] = param_set
        
        comparison["comparisons"].append(comparison_item)
    
    return comparison 