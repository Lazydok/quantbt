"""
CSV 데이터 제공자

CSV 파일에서 시장 데이터를 로드하는 데이터 제공자입니다.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import polars as pl

from ...core.interfaces.data_provider import DataProviderBase


class CSVDataProvider(DataProviderBase):
    """CSV 파일 기반 데이터 제공자"""
    
    def __init__(
        self, 
        data_dir: str,
        date_format: str = "%Y-%m-%d",
        symbol_column: str = "symbol",
        timestamp_column: str = "timestamp"
    ):
        super().__init__("CSVDataProvider")
        self.data_dir = Path(data_dir)
        self.date_format = date_format
        self.symbol_column = symbol_column
        self.timestamp_column = timestamp_column
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
    
    def _load_available_symbols(self) -> List[str]:
        """CSV 파일에서 사용 가능한 심볼 로드"""
        symbols = set()
        
        # CSV 파일들을 스캔하여 심볼 추출
        for csv_file in self.data_dir.glob("*.csv"):
            try:
                # 파일명에서 심볼 추출 (예: AAPL.csv -> AAPL)
                symbol_from_filename = csv_file.stem.upper()
                symbols.add(symbol_from_filename)
                
                # 파일 내용에서도 심볼 확인
                df = pl.read_csv(csv_file, n_rows=10)
                if self.symbol_column in df.columns:
                    file_symbols = df.select(self.symbol_column).unique().to_series().to_list()
                    symbols.update(symbol.upper() for symbol in file_symbols)
                    
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
                continue
        
        return sorted(list(symbols))
    
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
            symbol_data = self._load_symbol_data(symbol, start, end)
            if symbol_data is not None and symbol_data.height > 0:
                all_data.append(symbol_data)
        
        if not all_data:
            # 빈 DataFrame 반환
            return pl.DataFrame({
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }, schema={
                "timestamp": pl.Datetime,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64
            })
        
        # 모든 데이터 결합
        combined_data = pl.concat(all_data)
        
        # 날짜 범위 필터링
        filtered_data = combined_data.filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
        )
        
        return filtered_data
    
    def _load_symbol_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime
    ) -> pl.DataFrame:
        """특정 심볼의 데이터 로드"""
        # 여러 파일 패턴 시도
        possible_files = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol.lower()}.csv",
            self.data_dir / "data.csv",  # 모든 심볼이 하나의 파일에 있는 경우
        ]
        
        for csv_file in possible_files:
            if csv_file.exists():
                try:
                    df = self._read_csv_file(csv_file, symbol)
                    if df is not None and df.height > 0:
                        return df
                except Exception as e:
                    print(f"Warning: Could not read {csv_file} for {symbol}: {e}")
                    continue
        
        print(f"Warning: No data found for symbol {symbol}")
        return None
    
    def _read_csv_file(self, csv_file: Path, target_symbol: str) -> pl.DataFrame:
        """CSV 파일 읽기 및 표준화"""
        try:
            # CSV 파일 읽기
            df = pl.read_csv(csv_file)
            
            # 컬럼명 정규화 (소문자로 변환)
            df = df.rename({col: col.lower() for col in df.columns})
            
            # 필수 컬럼 확인
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            # 타임스탬프 컬럼 대안 확인
            if "timestamp" not in df.columns:
                for alt_col in ["date", "datetime", "time"]:
                    if alt_col in df.columns:
                        df = df.rename({alt_col: "timestamp"})
                        break
            
            if "timestamp" not in df.columns:
                raise ValueError("No timestamp column found")
            
            # 심볼 컬럼 처리
            if "symbol" not in df.columns:
                # 심볼 컬럼이 없으면 파일명에서 추출한 심볼 사용
                df = df.with_columns(pl.lit(target_symbol).alias("symbol"))
            else:
                # 특정 심볼만 필터링
                df = df.filter(pl.col("symbol").str.to_uppercase() == target_symbol.upper())
            
            # 타임스탬프 파싱
            df = self._parse_timestamp(df)
            
            # 필수 컬럼 재확인
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 데이터 타입 변환
            df = df.with_columns([
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("symbol").cast(pl.Utf8)
            ])
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_file}: {e}")
    
    def _parse_timestamp(self, df: pl.DataFrame) -> pl.DataFrame:
        """타임스탬프 파싱"""
        try:
            # 이미 datetime 타입인지 확인
            if df.select("timestamp").dtypes[0] == pl.Datetime:
                return df
            
            # 문자열에서 datetime으로 변환 시도
            try:
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format=self.date_format)
                )
            except:
                # 자동 파싱 시도
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime)
                )
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not parse timestamp column: {e}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터 정보 반환"""
        symbols = self.get_symbols()
        info = {
            "provider": self.name,
            "data_dir": str(self.data_dir),
            "available_symbols": symbols,
            "symbol_count": len(symbols)
        }
        
        # 각 심볼별 데이터 범위 정보
        symbol_info = {}
        for symbol in symbols[:5]:  # 처음 5개만 샘플링
            try:
                data = self._load_symbol_data(symbol, datetime(1900, 1, 1), datetime(2100, 1, 1))
                if data is not None and data.height > 0:
                    timestamps = data.select("timestamp").to_series()
                    symbol_info[symbol] = {
                        "start_date": timestamps.min(),
                        "end_date": timestamps.max(),
                        "record_count": data.height
                    }
            except:
                continue
        
        info["sample_data_ranges"] = symbol_info
        return info

    def _post_process_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """데이터 후처리"""
        # 기본적인 후처리 (정렬, 중복 제거 등)
        return data.sort(["timestamp", "symbol"]).unique() 