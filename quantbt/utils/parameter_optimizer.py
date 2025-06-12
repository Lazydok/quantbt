"""
파라메터 최적화 유틸리티

Phase 1 멀티프로세싱 백테스팅 최적화를 위한 파라메터 관련 유틸리티들
"""

from typing import Dict, List, Any, Optional, Iterator, Tuple
from itertools import product
import time
import logging

logger = logging.getLogger(__name__)


class ParameterCombinationGenerator:
    """파라메터 조합 생성기
    
    파라메터 그리드에서 모든 가능한 조합을 생성하는 클래스입니다.
    
    Example:
        >>> generator = ParameterCombinationGenerator()
        >>> param_grid = {
        ...     "buy_sma": [10, 15, 20],
        ...     "sell_sma": [25, 30, 35]
        ... }
        >>> combinations = generator.generate_combinations(param_grid)
        >>> len(combinations)
        9
        >>> combinations[0]
        {"buy_sma": 10, "sell_sma": 25}
    """
    
    def __init__(self):
        """파라메터 조합 생성기 초기화"""
        self.total_combinations_generated = 0
        self.generation_stats = {
            "total_time": 0.0,
            "total_combinations": 0,
            "avg_time_per_1000": 0.0
        }
    
    def generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """파라메터 그리드에서 모든 조합 생성
        
        Args:
            param_grid: 파라메터명을 키로, 가능한 값들의 리스트를 값으로 하는 딕셔너리
                       예: {"buy_sma": [10, 15, 20], "sell_sma": [25, 30]}
        
        Returns:
            모든 파라메터 조합의 리스트
            각 조합은 파라메터명을 키로, 특정 값을 값으로 하는 딕셔너리
        
        Raises:
            ValueError: 파라메터 그리드가 잘못된 형식인 경우
            
        Time Complexity: O(M^N) where M is average values per parameter, N is number of parameters
        Space Complexity: O(M^N) for storing all combinations
        """
        start_time = time.time()
        
        try:
            # 입력 검증
            if not isinstance(param_grid, dict):
                raise ValueError("파라메터 그리드는 딕셔너리여야 합니다")
            
            # 빈 그리드 처리
            if not param_grid:
                logger.debug("빈 파라메터 그리드 - 빈 조합 하나 반환")
                return [{}]
            
            # 모든 파라메터 값이 리스트인지 확인
            for param_name, values in param_grid.items():
                if not isinstance(values, (list, tuple)):
                    raise ValueError(f"파라메터 '{param_name}'의 값은 리스트 또는 튜플이어야 합니다")
                if not values:
                    raise ValueError(f"파라메터 '{param_name}'에 최소 하나의 값이 있어야 합니다")
            
            # 파라메터명과 값들 추출 (순서 보장)
            param_names = list(param_grid.keys())
            param_values = [param_grid[name] for name in param_names]
            
            # 총 조합 수 계산 (메모리 사전 검증용)
            total_combinations = 1
            for values in param_values:
                total_combinations *= len(values)
            
            # 메모리 사용량 사전 검증 (10만개 조합 이상시 경고)
            if total_combinations > 100_000:
                logger.warning(f"대용량 파라메터 그리드: {total_combinations:,}개 조합 생성 예정")
                
            # 조합 생성 (itertools.product 사용)
            combinations = []
            for combo_values in product(*param_values):
                combination = dict(zip(param_names, combo_values))
                combinations.append(combination)
            
            # 통계 업데이트
            generation_time = time.time() - start_time
            self._update_generation_stats(len(combinations), generation_time)
            
            logger.debug(f"파라메터 조합 생성 완료: {len(combinations):,}개, {generation_time:.3f}초")
            
            return combinations
            
        except Exception as e:
            logger.error(f"파라메터 조합 생성 실패: {e}")
            raise
    
    def generate_combinations_lazy(self, param_grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
        """파라메터 조합을 지연 생성 (메모리 효율적)
        
        대용량 파라메터 그리드의 경우 메모리 사용량을 줄이기 위해 
        제너레이터를 사용하여 필요할 때마다 조합을 생성합니다.
        
        Args:
            param_grid: 파라메터 그리드
            
        Yields:
            파라메터 조합 딕셔너리
        """
        # 입력 검증 (동일)
        if not isinstance(param_grid, dict):
            raise ValueError("파라메터 그리드는 딕셔너리여야 합니다")
        
        if not param_grid:
            yield {}
            return
        
        # 파라메터명과 값들 추출
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # 조합 지연 생성
        for combo_values in product(*param_values):
            combination = dict(zip(param_names, combo_values))
            yield combination
    
    def estimate_combinations_count(self, param_grid: Dict[str, List[Any]]) -> int:
        """파라메터 그리드의 총 조합 수 추정
        
        실제 조합을 생성하지 않고 총 개수만 계산합니다.
        
        Args:
            param_grid: 파라메터 그리드
            
        Returns:
            총 조합 수
        """
        if not param_grid:
            return 1
        
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        return total_combinations
    
    def estimate_memory_usage(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """파라메터 그리드의 예상 메모리 사용량 추정
        
        Args:
            param_grid: 파라메터 그리드
            
        Returns:
            메모리 사용량 추정 정보
        """
        total_combinations = self.estimate_combinations_count(param_grid)
        
        # 각 조합당 평균 메모리 사용량 추정 (대략적)
        avg_param_count = len(param_grid)
        avg_combination_size_bytes = avg_param_count * 100  # 파라메터당 ~100바이트 가정
        
        total_memory_bytes = total_combinations * avg_combination_size_bytes
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        return {
            "total_combinations": total_combinations,
            "estimated_memory_mb": total_memory_mb,
            "estimated_memory_gb": total_memory_mb / 1024,
            "avg_combination_size_bytes": avg_combination_size_bytes,
            "recommendation": self._get_memory_recommendation(total_memory_mb)
        }
    
    def _get_memory_recommendation(self, memory_mb: float) -> str:
        """메모리 사용량 기반 추천사항"""
        if memory_mb < 100:
            return "일반 생성 (generate_combinations) 사용 권장"
        elif memory_mb < 1000:
            return "주의: 메모리 사용량 높음. 지연 생성 (generate_combinations_lazy) 고려"
        else:
            return "경고: 높은 메모리 사용량. 지연 생성 필수 또는 파라메터 그리드 축소 권장"
    
    def _update_generation_stats(self, combinations_count: int, generation_time: float) -> None:
        """생성 통계 업데이트"""
        self.total_combinations_generated += combinations_count
        self.generation_stats["total_time"] += generation_time
        self.generation_stats["total_combinations"] += combinations_count
        
        # 1000개당 평균 시간 계산
        if self.generation_stats["total_combinations"] > 0:
            self.generation_stats["avg_time_per_1000"] = (
                self.generation_stats["total_time"] * 1000 / 
                self.generation_stats["total_combinations"]
            )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """생성 통계 조회
        
        Returns:
            생성 통계 정보
        """
        return self.generation_stats.copy()
    
    def reset_stats(self) -> None:
        """통계 초기화"""
        self.total_combinations_generated = 0
        self.generation_stats = {
            "total_time": 0.0,
            "total_combinations": 0,
            "avg_time_per_1000": 0.0
        }


class ParameterGridValidator:
    """파라메터 그리드 검증기
    
    파라메터 그리드의 유효성과 최적화 가능성을 검증합니다.
    """
    
    @staticmethod
    def validate_parameter_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """파라메터 그리드 검증
        
        Args:
            param_grid: 검증할 파라메터 그리드
            
        Returns:
            검증 결과 정보
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "statistics": {}
        }
        
        try:
            # 기본 구조 검증
            if not isinstance(param_grid, dict):
                validation_result["errors"].append("파라메터 그리드는 딕셔너리여야 합니다")
                validation_result["is_valid"] = False
                return validation_result
            
            if not param_grid:
                validation_result["warnings"].append("빈 파라메터 그리드입니다")
                return validation_result
            
            # 각 파라메터 검증
            total_combinations = 1
            param_count = len(param_grid)
            value_counts = []
            
            for param_name, values in param_grid.items():
                # 값 타입 검증
                if not isinstance(values, (list, tuple)):
                    validation_result["errors"].append(f"파라메터 '{param_name}'의 값은 리스트여야 합니다")
                    validation_result["is_valid"] = False
                    continue
                
                # 빈 값 검증
                if not values:
                    validation_result["errors"].append(f"파라메터 '{param_name}'에 값이 없습니다")
                    validation_result["is_valid"] = False
                    continue
                
                value_count = len(values)
                value_counts.append(value_count)
                total_combinations *= value_count
                
                # 중복값 검증
                unique_values = set(values)
                if len(unique_values) < len(values):
                    duplicate_count = len(values) - len(unique_values)
                    validation_result["warnings"].append(
                        f"파라메터 '{param_name}'에 {duplicate_count}개 중복값이 있습니다"
                    )
                
                # 값 개수 검증
                if value_count == 1:
                    validation_result["recommendations"].append(
                        f"파라메터 '{param_name}'은 단일값입니다. 고정값으로 처리 가능합니다"
                    )
                elif value_count > 20:
                    validation_result["warnings"].append(
                        f"파라메터 '{param_name}'에 {value_count}개 값이 있습니다 (많음)"
                    )
            
            # 통계 정보
            validation_result["statistics"] = {
                "parameter_count": param_count,
                "total_combinations": total_combinations,
                "avg_values_per_parameter": sum(value_counts) / len(value_counts) if value_counts else 0,
                "max_values_per_parameter": max(value_counts) if value_counts else 0,
                "min_values_per_parameter": min(value_counts) if value_counts else 0
            }
            
            # 조합 수 기반 추천
            if total_combinations > 10000:
                validation_result["warnings"].append(
                    f"총 {total_combinations:,}개 조합이 생성됩니다 (높은 계산 비용)"
                )
                validation_result["recommendations"].append("멀티프로세싱 사용을 강력히 권장합니다")
            elif total_combinations > 1000:
                validation_result["recommendations"].append("멀티프로세싱 사용을 권장합니다")
            
        except Exception as e:
            validation_result["errors"].append(f"검증 중 오류 발생: {e}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    @staticmethod
    def suggest_optimizations(param_grid: Dict[str, List[Any]]) -> List[str]:
        """파라메터 그리드 최적화 제안
        
        Args:
            param_grid: 파라메터 그리드
            
        Returns:
            최적화 제안 목록
        """
        suggestions = []
        
        validator = ParameterGridValidator()
        validation = validator.validate_parameter_grid(param_grid)
        
        if not validation["is_valid"]:
            return ["먼저 파라메터 그리드 오류를 수정하세요"]
        
        stats = validation["statistics"]
        total_combinations = stats["total_combinations"]
        
        # 조합 수 기반 제안
        if total_combinations > 50000:
            suggestions.append("매우 큰 파라메터 공간: 베이지안 최적화 또는 랜덤 서치 고려")
        elif total_combinations > 10000:
            suggestions.append("큰 파라메터 공간: 멀티프로세싱과 조기 중단 기법 사용")
        elif total_combinations > 1000:
            suggestions.append("중간 크기 파라메터 공간: 멀티프로세싱 사용 권장")
        
        # 불균형 파라메터 검출
        value_counts = [len(values) for values in param_grid.values()]
        max_values = max(value_counts)
        min_values = min(value_counts)
        
        if max_values / min_values > 5:
            suggestions.append("파라메터 값 개수가 불균형: 값이 많은 파라메터 우선 축소 고려")
        
        return suggestions


# 편의 함수들
def generate_parameter_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """파라메터 조합 생성 편의 함수
    
    Args:
        param_grid: 파라메터 그리드
        
    Returns:
        파라메터 조합 리스트
    """
    generator = ParameterCombinationGenerator()
    return generator.generate_combinations(param_grid)


def validate_parameter_grid(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """파라메터 그리드 검증 편의 함수
    
    Args:
        param_grid: 파라메터 그리드
        
    Returns:
        검증 결과
    """
    validator = ParameterGridValidator()
    return validator.validate_parameter_grid(param_grid) 