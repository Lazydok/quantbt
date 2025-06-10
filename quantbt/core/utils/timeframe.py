"""
타임프레임 변환 및 리샘플링 유틸리티

1분봉 기반 데이터를 다양한 타임프레임으로 변환하는 기능을 제공합니다.
"""

from typing import Dict, List, Optional
import polars as pl
import re


class TimeframeUtils:
    """타임프레임 변환 및 리샘플링 유틸리티"""
    
    # 지원되는 타임프레임과 분 단위 매핑
    TIMEFRAME_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
        "1M": 43200  # 약 30일
    }
    
    @classmethod
    def get_timeframe_minutes(cls, timeframe: str) -> int:
        """타임프레임 문자열을 분 단위로 변환
        
        Args:
            timeframe: 타임프레임 문자열 (예: "1m", "5m", "1h", "1d")
            
        Returns:
            분 단위 시간
            
        Raises:
            ValueError: 지원되지 않는 타임프레임인 경우
        """
        normalized_tf = cls._normalize_timeframe(timeframe)
        
        if normalized_tf not in cls.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return cls.TIMEFRAME_MINUTES[normalized_tf]
    
    @classmethod
    def _normalize_timeframe(cls, timeframe: str) -> str:
        """타임프레임 문자열을 정규화
        
        Args:
            timeframe: 원본 타임프레임 문자열
            
        Returns:
            정규화된 타임프레임 문자열
        """
        # 대소문자 구분하여 월간 먼저 처리
        tf = timeframe.strip()
        
        # 월간 패턴 먼저 확인 (대소문자 구분)
        if re.match(r"^1M$|^1month$|^monthly$", tf):
            return "1M"
        
        # 나머지는 소문자로 변환 후 패턴 매칭
        tf_lower = tf.lower()
        
        # 패턴 매칭을 통한 정규화 (월간 제외)
        patterns = {
            r"^1m$|^1min$|^1minute$": "1m",
            r"^5m$|^5min$|^5minutes$": "5m", 
            r"^15m$|^15min$|^15minutes$": "15m",
            r"^30m$|^30min$|^30minutes$": "30m",
            r"^1h$|^1hr$|^1hour$|^60min$|^60minutes$": "1h",
            r"^4h$|^4hr$|^4hours$|^240min$|^240minutes$": "4h",
            r"^1d$|^1day$|^daily$|^1440min$|^1440minutes$": "1d",
            r"^1w$|^1week$|^weekly$": "1w"
        }
        
        for pattern, normalized in patterns.items():
            if re.match(pattern, tf_lower):
                return normalized
        
        # 매칭되지 않으면 원본 반환
        return timeframe
    
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
        if cls._normalize_timeframe(timeframe) == cls._normalize_timeframe(base_timeframe):
            return data.sort(["symbol", "timestamp"])
        
        # 목표 타임프레임이 기본 타임프레임보다 작은 경우 에러
        target_minutes = cls.get_timeframe_minutes(timeframe)
        base_minutes = cls.get_timeframe_minutes(base_timeframe)
        
        if target_minutes < base_minutes:
            raise ValueError(
                f"Cannot resample from {base_timeframe} to smaller timeframe {timeframe}"
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
            raise ValueError(f"Failed to resample data: {str(e)}")
    
    @classmethod
    def _convert_to_polars_interval(cls, timeframe: str) -> str:
        """타임프레임을 Polars interval 형식으로 변환
        
        Args:
            timeframe: 정규화된 타임프레임
            
        Returns:
            Polars interval 문자열
        """
        normalized_tf = cls._normalize_timeframe(timeframe)
        
        conversion_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m", 
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1mo"  # Polars는 "1mo"를 사용
        }
        
        if normalized_tf not in conversion_map:
            raise ValueError(f"Cannot convert timeframe to Polars interval: {timeframe}")
        
        return conversion_map[normalized_tf]
    
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
        base_tf_normalized = cls._normalize_timeframe(base_timeframe)
        if base_tf_normalized in [cls._normalize_timeframe(tf) for tf in timeframes]:
            result[base_tf_normalized] = base_data.sort(["symbol", "timestamp"])
        
        # 각 타임프레임별로 리샘플링
        for timeframe in timeframes:
            normalized_tf = cls._normalize_timeframe(timeframe)
            
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
                continue
        
        return result
    
    @classmethod
    def get_supported_timeframes(cls) -> List[str]:
        """지원되는 타임프레임 목록 반환
        
        Returns:
            지원되는 타임프레임 리스트
        """
        return list(cls.TIMEFRAME_MINUTES.keys())
    
    @classmethod
    def filter_valid_timeframes(cls, timeframes: List[str]) -> List[str]:
        """유효한 타임프레임만 필터링
        
        Args:
            timeframes: 검증할 타임프레임 리스트
            
        Returns:
            유효한 타임프레임 리스트
        """
        return [tf for tf in timeframes if cls.validate_timeframe(tf)] 