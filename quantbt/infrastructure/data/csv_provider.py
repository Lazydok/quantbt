"""
CSV 데이터 제공자

CSV 파일에서 시장 데이터를 로드하는 데이터 제공자입니다.
"""
from typing import List, Dict, Optional
from datetime import datetime
import polars as pl
from pathlib import Path

from ...core.interfaces.data_provider import DataProviderBase

# 멀티타임프레임 지원을 위한 import 추가
try:
    from ...core.utils.timeframe import TimeframeUtils
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False


class CSVDataProvider(DataProviderBase):
    """
    CSV 파일 기반 데이터 제공자.
    멀티 심볼 및 멀티 타임프레임 데이터를 지원하도록 설계되었습니다.

    Args:
        data_files (Dict[str, Dict[str, str]]):
            심볼과 타임프레임에 따른 파일 경로 딕셔너리.
            예시:
            {
                "BTC-USD": {
                    "1d": "/path/to/btc_daily.csv",
                    "1m": "/path/to/btc_minute.csv"
                },
                "ETH-USD": {
                    "1d": "/path/to/eth_daily.csv"
                }
            }
        date_format (str, optional): CSV 파일의 날짜 형식. 기본값: None (자동 감지).
        timestamp_column (str, optional): 타임스탬프 컬럼명. 기본값: "date".
    """
    
    def __init__(
        self, 
        data_files: Dict[str, Dict[str, str]],
        date_format: Optional[str] = None,
        timestamp_column: str = "date"
    ):
        super().__init__("CSVDataProvider")
        self.data_files = data_files
        self.date_format = date_format
        self.timestamp_column = timestamp_column
        self._resampled_data_cache: Dict[str, pl.DataFrame] = {} # 리샘플링 데이터 캐시
        
        # 데이터 파일 존재 여부 검증
        for symbol, timeframes in self.data_files.items():
            for timeframe, path in timeframes.items():
                if not Path(path).exists():
                    raise ValueError(f"Data file does not exist for {symbol} at {timeframe}: {path}")

    def supports_timeframe(self, timeframe: str) -> bool:
        """타임프레임 지원 여부 확인"""
        # 직접 제공되거나, 분봉에서 리샘플링 가능한 경우 지원
        for symbol_data in self.data_files.values():
            if timeframe in symbol_data:
                return True
            if '1m' in symbol_data and timeframe == '1d': # 분봉 -> 일봉 리샘플링 지원
                return True
        return False
    
    def _load_available_symbols(self) -> List[str]:
        """사용 가능한 심볼 로드"""
        return sorted(list(self.data_files.keys()))
    
    async def _fetch_raw_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> pl.DataFrame:
        """CSV 파일에서 원시 데이터 조회"""
        all_data = []
        
        for symbol in symbols:
            symbol_files = self.data_files.get(symbol)
            if not symbol_files:
                print(f"Warning: No data files configured for symbol {symbol}")
                continue

            file_path = symbol_files.get(timeframe)
            
            # 요청한 타임프레임이 없을 경우, 리샘플링 시도
            if file_path is None:
                if timeframe == '1d' and '1m' in symbol_files:
                    symbol_data = self._get_resampled_data(symbol, start, end)
                else:
                    # 지원하지 않는 타임프레임
                    print(f"Warning: Timeframe '{timeframe}' not available for symbol {symbol}")
                    continue
            else:
                symbol_data = self._read_and_filter_csv(file_path, symbol, start, end)

            if symbol_data is not None and not symbol_data.is_empty():
                all_data.append(symbol_data)
        
        if not all_data:
            return self._get_empty_df()
        
        combined_data = pl.concat(all_data)
        return combined_data

    def _get_resampled_data(self, symbol: str, start: datetime, end: datetime) -> Optional[pl.DataFrame]:
        """
        분봉 데이터를 일봉으로 리샘플링하고 결과를 캐시합니다.
        """
        cache_key = f"{symbol}_1d_from_1m"
        if cache_key in self._resampled_data_cache:
            resampled_df = self._resampled_data_cache[cache_key]
        else:
            minute_file_path = self.data_files[symbol]['1m']
            df = self._read_and_filter_csv(minute_file_path, symbol, datetime.min, datetime.max)
            
            if df is None or df.is_empty():
                return None

            # 일봉으로 리샘플링
            resampled_df = df.group_by_dynamic(
                "timestamp", every="1d", group_by="symbol"
            ).agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ]).sort("timestamp")
            
            # 열 순서 맞추기
            resampled_df = resampled_df.select(
                "timestamp", "open", "high", "low", "close", "volume", "symbol"
            )

            self._resampled_data_cache[cache_key] = resampled_df

        # 날짜 범위 필터링
        return resampled_df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    def _read_and_filter_csv(self, file_path: str, symbol: str, start: datetime, end: datetime) -> Optional[pl.DataFrame]:
        """CSV 파일을 읽고 날짜로 필터링합니다."""
        try:
            # 파일에서 실제 헤더를 먼저 읽어옴
            header = pl.read_csv(Path(file_path), n_rows=0).columns
            
            # dtypes 딕셔너리 생성
            read_dtypes = {col: pl.Utf8 for col in header}
            
            df = pl.read_csv(Path(file_path), schema_overrides=read_dtypes)
            
            # 컬럼명 표준화 (소문자로)
            df = df.rename({col: col.lower() for col in df.columns})
            
            # 타임스탬프 컬럼 이름 맞추기
            current_ts_col = self.timestamp_column.lower()
            if current_ts_col in df.columns and current_ts_col != "timestamp":
                 df = df.rename({current_ts_col: "timestamp"})

            # 필수 컬럼 확인
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                raise ValueError(f"CSV file {file_path} is missing one of required columns: {required}")
                
            # 타임스탬프 파싱
            if df.select("timestamp").dtypes[0] != pl.Datetime:
                if self.date_format:
                    df = df.with_columns(pl.col("timestamp").str.to_datetime(format=self.date_format))
                else:
                    df = df.with_columns(pl.col("timestamp").str.to_datetime())
            
            # 네이티브 타임스탬프로 변환하여 타임존 정보 제거
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

            # 데이터 타입 캐스팅 및 심볼 추가
            df = df.select([
                pl.col("timestamp"),
                pl.col("open").str.strip_chars().cast(pl.Float64),
                pl.col("high").str.strip_chars().cast(pl.Float64),
                pl.col("low").str.strip_chars().cast(pl.Float64),
                pl.col("close").str.strip_chars().cast(pl.Float64),
                pl.col("volume").str.strip_chars().cast(pl.Float64)
            ]).with_columns(pl.lit(symbol).alias("symbol"))

            # 날짜 필터링 및 정렬
            return df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            ).sort("timestamp")

        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")
            return None
            
    def _get_empty_df(self) -> pl.DataFrame:
        """빈 데이터프레임을 표준 스키마와 함께 반환합니다."""
        return pl.DataFrame({
            "timestamp": [], "symbol": [], "open": [], "high": [],
            "low": [], "close": [], "volume": []
        }, schema={
            "timestamp": pl.Datetime, "symbol": pl.Utf8, "open": pl.Float64,
            "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64
        })

    def _post_process_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """데이터 후처리 (정렬, 중복 제거 등)"""
        if data.is_empty():
            return data
        return data.sort(["timestamp", "symbol"]).unique() 