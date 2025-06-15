"""
타임프레임 변환 및 리샘플링 유틸리티

1분봉 기반 데이터를 다양한 타임프레임으로 변환하는 기능을 제공합니다.
동적 타임프레임 지원: 1분의 배수로 임의의 값 설정 가능 (예: 7m, 13h, 45d 등)
"""

from typing import Dict, List, Optional
import polars as pl
import re


class TimeframeUtils:
    """타임프레임 변환 및 리샘플링 유틸리티
    
    동적 타임프레임 지원:
    - 숫자 + 단위 형태로 임의의 타임프레임 지원 (예: 7m, 13h, 45d)
    - 지원 단위: m(분), h(시간), d(일), w(주), M(월)
    - 모든 값은 1분의 배수로 변환되어 처리
    """
    
    # 단위별 분 단위 변환 상수
    UNIT_TO_MINUTES = {
        'm': 1,          # 분
        'h': 60,         # 시간 -> 분
        'd': 1440,       # 일 -> 분 (24 * 60)
        'w': 10080,      # 주 -> 분 (7 * 24 * 60)
        'M': 43200       # 월 -> 분 (30 * 24 * 60, 근사값)
    }
    
    @classmethod
    def parse_timeframe(cls, timeframe: str) -> tuple[int, str]:
        """타임프레임 문자열을 숫자와 단위로 분리
        
        Args:
            timeframe: 타임프레임 문자열 (예: "5m", "2h", "1d")
            
        Returns:
            (숫자, 단위) 튜플
            
        Raises:
            ValueError: 잘못된 형식인 경우
        """
        if not isinstance(timeframe, str):
            raise ValueError(f"Timeframe must be string, got {type(timeframe)}")
        
        timeframe = timeframe.strip()
        
        # 정규식으로 숫자와 단위 분리
        pattern = r'^(\d+)([mhdwM])$'
        match = re.match(pattern, timeframe)
        
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}. Expected format: number + unit (m/h/d/w/M)")
        
        number = int(match.group(1))
        unit = match.group(2)
        
        if number <= 0:
            raise ValueError(f"Timeframe number must be positive, got {number}")
        
        if unit not in cls.UNIT_TO_MINUTES:
            raise ValueError(f"Unsupported unit: {unit}. Supported units: {list(cls.UNIT_TO_MINUTES.keys())}")
        
        return number, unit
    
    @classmethod
    def get_timeframe_minutes(cls, timeframe: str) -> int:
        """타임프레임 문자열을 분 단위로 변환
        
        Args:
            timeframe: 타임프레임 문자열 (예: "5m", "2h", "1d")
            
        Returns:
            분 단위 시간
            
        Raises:
            ValueError: 잘못된 타임프레임인 경우
        """
        try:
            number, unit = cls.parse_timeframe(timeframe)
            return number * cls.UNIT_TO_MINUTES[unit]
        except Exception as e:
            raise ValueError(f"Failed to parse timeframe '{timeframe}': {str(e)}")
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """타임프레임 유효성 검증
        
        Args:
            timeframe: 검증할 타임프레임
            
        Returns:
            유효하면 True, 아니면 False
        """
        try:
            cls.get_timeframe_minutes(timeframe)
            return True
        except ValueError:
            return False
    
    @classmethod
    def normalize_timeframe(cls, timeframe: str) -> str:
        """타임프레임 정규화
        
        Args:
            timeframe: 원본 타임프레임 문자열
            
        Returns:
            정규화된 타임프레임 문자열
        """
        try:
            number, unit = cls.parse_timeframe(timeframe)
            return f"{number}{unit}"
        except ValueError:
            # 파싱 실패 시 원본 반환
            return timeframe.strip()
    
    @classmethod
    def resample_to_timeframe(
        cls,
        data: pl.DataFrame,
        timeframe: str,
        base_timeframe: str = "1m"
    ) -> pl.DataFrame:
        """OHLCV 데이터를 지정된 타임프레임으로 리샘플링
        
        Args:
            data: 원본 OHLCV 데이터 (timestamp, symbol, open, high, low, close, volume 포함)
            timeframe: 목표 타임프레임
            base_timeframe: 원본 데이터의 타임프레임 (기본값: "1m")
            
        Returns:
            리샘플링된 OHLCV 데이터
            
        Raises:
            ValueError: 잘못된 타임프레임이거나 필수 컬럼이 없는 경우
        """
        # 타임프레임 유효성 검증
        if not cls.validate_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        if not cls.validate_timeframe(base_timeframe):
            raise ValueError(f"Invalid base timeframe: {base_timeframe}")
        
        # 필수 컬럼 확인
        required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 같은 타임프레임인 경우 원본 반환
        if cls.normalize_timeframe(timeframe) == cls.normalize_timeframe(base_timeframe):
            return data.sort(["symbol", "timestamp"])
        
        # 목표 타임프레임이 기본 타임프레임보다 작은 경우 에러
        target_minutes = cls.get_timeframe_minutes(timeframe)
        base_minutes = cls.get_timeframe_minutes(base_timeframe)
        
        if target_minutes < base_minutes:
            raise ValueError(
                f"Cannot resample to smaller timeframe: {timeframe} ({target_minutes}m) < {base_timeframe} ({base_minutes}m)"
            )
        
        # Polars의 group_by_dynamic을 사용하여 리샘플링
        try:
            # 타임프레임을 Polars 형식으로 변환
            polars_interval = cls._convert_to_polars_interval(timeframe)
            
            # 지표 컬럼들 식별 (OHLCV가 아닌 모든 컬럼)
            base_columns = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
            indicator_columns = [col for col in data.columns if col not in base_columns]
            
            # 리샘플링 집계 정의
            aggregations = [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume")
            ]
            
            # 지표 컬럼들은 마지막 값으로 집계 (지표는 해당 기간의 마지막 값이 유의미)
            for col in indicator_columns:
                aggregations.append(pl.col(col).last().alias(col))
            
            # 심볼별로 그룹화하여 리샘플링
            resampled = (
                data
                .sort(["symbol", "timestamp"])
                .group_by_dynamic(
                    "timestamp",
                    every=polars_interval,
                    group_by="symbol",
                    closed="left"  # 구간의 시작 시점 포함
                )
                .agg(aggregations)
                .sort(["symbol", "timestamp"])
            )
            
            return resampled
            
        except Exception as e:
            raise ValueError(f"Failed to resample data from {base_timeframe} to {timeframe}: {str(e)}")
    
    @classmethod
    def _convert_to_polars_interval(cls, timeframe: str) -> str:
        """타임프레임을 Polars interval 형식으로 변환
        
        Args:
            timeframe: 타임프레임 문자열
            
        Returns:
            Polars interval 문자열
        """
        try:
            number, unit = cls.parse_timeframe(timeframe)
            
            # Polars 단위 매핑
            polars_unit_map = {
                'm': 'm',      # 분
                'h': 'h',      # 시간
                'd': 'd',      # 일
                'w': 'w',      # 주
                'M': 'mo'      # 월 (Polars는 "mo" 사용)
            }
            
            polars_unit = polars_unit_map.get(unit)
            if polars_unit is None:
                raise ValueError(f"Cannot convert unit '{unit}' to Polars format")
            
            return f"{number}{polars_unit}"
            
        except Exception as e:
            raise ValueError(f"Cannot convert timeframe to Polars interval: {timeframe} - {str(e)}")
    
    @classmethod
    def create_multi_timeframe_data(
        cls,
        base_data: pl.DataFrame,
        timeframes: List[str],
        base_timeframe: str = "1m"
    ) -> Dict[str, pl.DataFrame]:
        """멀티타임프레임 데이터셋 생성
        
        Args:
            base_data: 기본 타임프레임 데이터
            timeframes: 생성할 타임프레임 리스트
            base_timeframe: 기본 데이터의 타임프레임
            
        Returns:
            타임프레임별 데이터 딕셔너리
        """
        result = {}
        
        # 기본 타임프레임이 요청된 타임프레임에 포함되어 있으면 추가
        base_tf_normalized = cls.normalize_timeframe(base_timeframe)
        normalized_timeframes = [cls.normalize_timeframe(tf) for tf in timeframes]
        
        if base_tf_normalized in normalized_timeframes:
            result[base_tf_normalized] = base_data.sort(["symbol", "timestamp"])
        
        # 각 타임프레임별로 리샘플링
        for timeframe in timeframes:
            normalized_tf = cls.normalize_timeframe(timeframe)
            
            # 이미 처리된 기본 타임프레임은 스킵
            if normalized_tf == base_tf_normalized:
                continue
                
            try:
                resampled_data = cls.resample_to_timeframe(
                    base_data, 
                    timeframe, 
                    base_timeframe
                )
                result[normalized_tf] = resampled_data
            except ValueError as e:
                print(f"⚠️ 타임프레임 '{timeframe}' 리샘플링 실패: {e}")
                continue
        
        return result
    
    @classmethod
    def get_supported_units(cls) -> List[str]:
        """지원되는 단위 목록 반환
        
        Returns:
            지원되는 단위 리스트
        """
        return list(cls.UNIT_TO_MINUTES.keys())
    
    @classmethod
    def filter_valid_timeframes(cls, timeframes: List[str]) -> List[str]:
        """유효한 타임프레임만 필터링
        
        Args:
            timeframes: 검증할 타임프레임 리스트
            
        Returns:
            유효한 타임프레임 리스트
        """
        valid_timeframes = []
        for tf in timeframes:
            if cls.validate_timeframe(tf):
                normalized_tf = cls.normalize_timeframe(tf)
                if normalized_tf not in valid_timeframes:
                    valid_timeframes.append(normalized_tf)
            else:
                print(f"⚠️ 잘못된 타임프레임 무시: {tf}")
        
        return valid_timeframes
    
    @classmethod
    def sort_timeframes_by_duration(cls, timeframes: List[str]) -> List[str]:
        """타임프레임을 시간 순서로 정렬 (작은 것부터)
        
        Args:
            timeframes: 정렬할 타임프레임 리스트
            
        Returns:
            시간 순으로 정렬된 타임프레임 리스트
        """
        valid_timeframes = cls.filter_valid_timeframes(timeframes)
        
        # 시간(분)과 함께 튜플로 만들어 정렬
        timeframe_minutes = []
        for tf in valid_timeframes:
            try:
                minutes = cls.get_timeframe_minutes(tf)
                timeframe_minutes.append((tf, minutes))
            except ValueError:
                continue
        
        # 시간 순으로 정렬
        timeframe_minutes.sort(key=lambda x: x[1])
        
        return [tf for tf, _ in timeframe_minutes]
    
    @classmethod
    def get_timeframe_info(cls, timeframe: str) -> Dict[str, any]:
        """타임프레임 상세 정보 반환
        
        Args:
            timeframe: 타임프레임 문자열
            
        Returns:
            타임프레임 정보 딕셔너리
        """
        try:
            number, unit = cls.parse_timeframe(timeframe)
            minutes = cls.get_timeframe_minutes(timeframe)
            
            unit_names = {
                'm': '분',
                'h': '시간',
                'd': '일',
                'w': '주',
                'M': '월'
            }
            
            return {
                'timeframe': timeframe,
                'number': number,
                'unit': unit,
                'unit_name': unit_names.get(unit, unit),
                'total_minutes': minutes,
                'description': f"{number}{unit_names.get(unit, unit)}"
            }
        except ValueError as e:
            return {
                'timeframe': timeframe,
                'error': str(e),
                'valid': False
            } 