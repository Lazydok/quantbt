"""
바이낸스 데이터 프로바이더 - SQLite + Parquet 기반

바이낸스 API를 활용한 암호화폐 데이터 제공자
"""

import asyncio
import aiohttp
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone, timedelta
import polars as pl
import json

from ...core.interfaces.data_provider import DataProviderBase
from ...core.utils.timeframe import TimeframeUtils
from ...config import DB_PATH, CACHE_DIR
from .database_manager import DatabaseManager
from .cache_manager import CacheManager


class BinanceDataProvider(DataProviderBase):
    """바이낸스 API 기반 데이터 제공자 - SQLite + Parquet"""
    
    SUPPORTED_API_TIMEFRAMES = [
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    ]
    PROVIDER_NAME = "binance"
    
    def __init__(
        self,
        db_path: str = None,
        cache_dir: str = None,
        rate_limit_delay: float = 0.05,  # 1200 req/min = 20 req/sec
        max_candles_per_request: int = 1000
    ):
        super().__init__("BinanceDataProvider")
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limit_delay = rate_limit_delay
        self.max_candles_per_request = max_candles_per_request
        
        # config.py에서 절대경로 사용
        if db_path is None:
            db_path = DB_PATH
        if cache_dir is None:
            cache_dir = CACHE_DIR
        
        # 데이터베이스 및 캐시 매니저
        self.db_manager = DatabaseManager(db_path)
        self.cache_manager = CacheManager(cache_dir, db_path)
        
        # HTTP 세션
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 심볼 캐시
        self._all_symbols_cache: Optional[List[str]] = None
        self._usdt_symbols_cache: Optional[List[str]] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 반환 - 타임아웃 및 재시도 설정"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=10,  # 전체 요청 10초 타임아웃
                    connect=3,  # 연결 3초 타임아웃
                    sock_read=5  # 소켓 읽기 5초 타임아웃
                ),
                headers={
                    "accept": "application/json",
                    "user-agent": "QuantBT/1.0"
                },
                connector=aiohttp.TCPConnector(
                    limit=20,  # 동시 연결 제한
                    ttl_dns_cache=300,  # DNS 캐시 5분
                    use_dns_cache=True
                )
            )
        return self._session
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _normalize_timezone(self, dt: datetime) -> datetime:
        """datetime을 UTC로 정규화"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            return dt.astimezone(timezone.utc)
    
    def _to_utc(self, dt: datetime) -> datetime:
        """datetime을 UTC로 변환"""
        return self._normalize_timezone(dt)
    
    def _parse_binance_kline(self, kline_data: list, symbol: str) -> dict:
        """바이낸스 kline 데이터를 표준 포맷으로 변환"""
        return {
            "timestamp": datetime.fromtimestamp(kline_data[0] / 1000, tz=timezone.utc),
            "symbol": symbol,
            "open": float(kline_data[1]),
            "high": float(kline_data[2]),
            "low": float(kline_data[3]),
            "close": float(kline_data[4]),
            "volume": float(kline_data[5])
        }
    
    def _fill_missing_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """빈값을 이전값으로 채우기"""
        if df.is_empty():
            return df
        
        # 심볼별로 그룹화하여 처리
        result_dfs = []
        
        for symbol in df["symbol"].unique():
            symbol_df = df.filter(pl.col("symbol") == symbol).sort("timestamp")
            
            # forward fill 수행
            filled_df = symbol_df.with_columns([
                pl.col("open").fill_null(strategy="forward"),
                pl.col("high").fill_null(strategy="forward"),
                pl.col("low").fill_null(strategy="forward"),
                pl.col("close").fill_null(strategy="forward"),
                pl.col("volume").fill_null(strategy="forward")
            ])
            
            result_dfs.append(filled_df)
        
        if result_dfs:
            return pl.concat(result_dfs).sort(["symbol", "timestamp"])
        else:
            return df
    
    async def get_all_symbols(self) -> List[str]:
        """전체 심볼 목록 조회"""
        if self._all_symbols_cache is not None:
            return self._all_symbols_cache
        
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/ticker/24hr") as response:
                response.raise_for_status()
                tickers = await response.json()
                
                symbols = [ticker["symbol"] for ticker in tickers]
                self._all_symbols_cache = symbols
                return symbols
                
        except Exception as e:
            print(f"⚠️ 전체 심볼 조회 오류: {e}")
            return []
    
    async def get_usdt_symbols(self) -> List[str]:
        """USDT 페어 심볼만 조회"""
        if self._usdt_symbols_cache is not None:
            return self._usdt_symbols_cache
        
        all_symbols = await self.get_all_symbols()
        usdt_symbols = [symbol for symbol in all_symbols if symbol.endswith("USDT")]
        self._usdt_symbols_cache = usdt_symbols
        return usdt_symbols
    
    def _load_available_symbols(self) -> List[str]:
        """사용 가능한 심볼 로드 - 주요 USDT 페어"""
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
            "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT",
            "MANAUSDT", "SANDUSDT", "AXSUSDT", "ICPUSDT", "NEARUSDT"
        ]
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """모든 타임프레임 지원 (리샘플링 처리)"""
        return TimeframeUtils.validate_timeframe(timeframe)
    
    def validate_date_range(self, start: datetime, end: datetime) -> bool:
        """날짜 범위 유효성 검증"""
        start_normalized = self._normalize_timezone(start)
        end_normalized = self._normalize_timezone(end)
        now_utc = datetime.now(timezone.utc)
        
        return start_normalized <= end_normalized and start_normalized <= now_utc
    
    async def _check_network_connection(self) -> bool:
        """네트워크 연결 상태 확인"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/ping") as response:
                return response.status == 200
        except Exception:
            return False
    
    def _get_existing_data_ranges(
        self, 
        symbol: str, 
        timeframe: str, 
        start_utc: datetime, 
        end_utc: datetime
    ) -> List[Dict[str, datetime]]:
        """DB에서 기존 데이터의 실제 범위들을 조회"""
        try:
            import sqlite3
            ranges = []
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.execute('''
                    SELECT timestamp_utc 
                    FROM market_data 
                    WHERE provider = ? AND symbol = ? AND timeframe = ?
                    AND timestamp_utc >= ? AND timestamp_utc <= ?
                    ORDER BY timestamp_utc
                ''', (
                    self.PROVIDER_NAME, 
                    symbol, 
                    timeframe,
                    start_utc.isoformat(),
                    end_utc.isoformat()
                ))
                
                timestamps = []
                for row in cursor.fetchall():
                    timestamp_str = row[0]
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str.replace("Z", "+00:00")
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    timestamps.append(timestamp)
                
                if not timestamps:
                    return ranges
                
                # 연속된 데이터 구간을 찾아서 범위로 그룹핑
                if timeframe == "1m":
                    gap_threshold = timedelta(minutes=10)
                elif timeframe == "1d":
                    gap_threshold = timedelta(days=2)
                else:
                    gap_threshold = timedelta(hours=2)
                
                current_start = timestamps[0]
                current_end = timestamps[0]
                
                for i in range(1, len(timestamps)):
                    current_ts = timestamps[i]
                    
                    if current_ts - current_end <= gap_threshold:
                        current_end = current_ts
                    else:
                        ranges.append({
                            "start": current_start,
                            "end": current_end
                        })
                        current_start = current_ts
                        current_end = current_ts
                
                ranges.append({
                    "start": current_start,
                    "end": current_end
                })
                
            return ranges
            
        except Exception as e:
            print(f"⚠️ 기존 데이터 범위 조회 오류: {e}")
            return []
    
    def _calculate_missing_periods(
        self,
        symbol: str,
        timeframe: str,
        requested_start: datetime,
        requested_end: datetime,
        existing_ranges: List[Dict[str, datetime]]
    ) -> List[Dict[str, datetime]]:
        """요청 기간에서 기존 데이터 범위를 제외한 빈 기간들을 계산"""
        if not existing_ranges:
            return [{
                "start": requested_start,
                "end": requested_end
            }]
        
        missing_periods = []
        
        sorted_ranges = sorted(existing_ranges, key=lambda x: x["start"])
        
        if requested_start.tzinfo is None:
            requested_start = requested_start.replace(tzinfo=timezone.utc)
        if requested_end.tzinfo is None:
            requested_end = requested_end.replace(tzinfo=timezone.utc)
        
        current_time = requested_start
        
        for existing_range in sorted_ranges:
            range_start = existing_range["start"]
            range_end = existing_range["end"]
            
            if range_start.tzinfo is None:
                range_start = range_start.replace(tzinfo=timezone.utc)
            if range_end.tzinfo is None:
                range_end = range_end.replace(tzinfo=timezone.utc)
            
            if current_time < range_start:
                missing_periods.append({
                    "start": current_time,
                    "end": range_start
                })
            
            current_time = max(current_time, range_end)
        
        if current_time < requested_end:
            missing_periods.append({
                "start": current_time,
                "end": requested_end
            })
        
        return missing_periods
    
    async def _fetch_candles_from_api(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """바이낸스 API에서 캔들 데이터 조회"""
        if timeframe not in self.SUPPORTED_API_TIMEFRAMES:
            raise ValueError(f"API timeframe not supported: {timeframe}")
        
        # UTC로 정규화
        start_utc = self._normalize_timezone(start)
        end_utc = self._normalize_timezone(end)
        
        session = await self._get_session()
        all_candles = []
        
        url = f"{self.base_url}/klines"
        current_start = start_utc
        
        max_requests = 10000
        request_count = 0
        start_time = asyncio.get_event_loop().time()
        
        total_days = (end_utc.date() - start_utc.date()).days + 1
        processed_dates = set()
        latest_processed_date = None
        total_candles_received = 0
        
        def update_progress():
            elapsed_time = asyncio.get_event_loop().time() - start_time
            rps = request_count / elapsed_time if elapsed_time > 0 else 0
            cps = total_candles_received / elapsed_time if elapsed_time > 0 else 0
            
            processed_days = len(processed_dates)
            progress_pct = (processed_days / total_days) * 100 if total_days > 0 else 0
            
            latest_date_str = latest_processed_date.strftime("%Y-%m-%d") if latest_processed_date else "N/A"
            
            progress_msg = (f"\r📈 {symbol} {timeframe}: "
                          f"요청 {request_count:3d}/{max_requests} | "
                          f"RPS: {rps:4.1f} | "
                          f"CPS: {cps:5.0f} | "
                          f"진행: {processed_days:3d}/{total_days:3d}일 ({progress_pct:5.1f}%) | "
                          f"최신: {latest_date_str}")
            
            print(progress_msg, end="", flush=True)
        
        try:
            update_progress()
            
            while current_start < end_utc and request_count < max_requests:
                params = {
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(end_utc.timestamp() * 1000),
                    "limit": self.max_candles_per_request
                }
                
                request_count += 1
                
                # 재시도 로직
                max_retries = 3
                retry_delay = 1
                candles = None
                
                for attempt in range(max_retries):
                    try:
                        async with session.get(url, params=params) as response:
                            response.raise_for_status()
                            candles = await response.json()
                            break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                        else:
                            raise e
                
                if not candles:
                    break
                
                # 데이터 파싱 및 필터링
                valid_candles = []
                latest_in_batch = None
                
                for kline in candles:
                    candle_time = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
                    
                    if start_utc <= candle_time <= end_utc:
                        valid_candles.append(kline)
                        processed_dates.add(candle_time.date())
                        if latest_in_batch is None or candle_time > latest_in_batch:
                            latest_in_batch = candle_time
                
                if valid_candles:
                    all_candles.extend(valid_candles)
                    total_candles_received += len(valid_candles)
                    
                    if latest_in_batch:
                        latest_processed_date = latest_in_batch
                        current_start = latest_in_batch + timedelta(milliseconds=1)
                    
                    update_progress()
                else:
                    break
                
                await asyncio.sleep(self.rate_limit_delay)
            
            if request_count >= max_requests:
                print(f"\n⚠️  최대 요청 수 ({max_requests}) 도달")
            else:
                print()
                    
        finally:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            final_rps = request_count / elapsed_time if elapsed_time > 0 else 0
            final_cps = total_candles_received / elapsed_time if elapsed_time > 0 else 0
            processed_days = len(processed_dates)
            
            print(f"✅ {symbol} {timeframe} 완료: "
                  f"{total_candles_received}개 캔들, {processed_days}/{total_days}일, "
                  f"평균 {final_rps:.1f} RPS, {final_cps:.0f} CPS, "
                  f"{elapsed_time:.1f}초 소요")
        
        # DataFrame 변환
        if all_candles:
            df_data = []
            for kline in all_candles:
                parsed_candle = self._parse_binance_kline(kline, symbol)
                df_data.append(parsed_candle)
            
            result_df = pl.DataFrame(df_data).sort("timestamp")
            # 빈값 채우기
            result_df = self._fill_missing_data(result_df)
            return result_df
        
        # 빈 DataFrame 반환
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64
        })
    
    async def _fetch_raw_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> pl.DataFrame:
        """원시 데이터 조회 - 3단계 시스템 (캐시 -> DB -> API)"""
        if timeframe in self.SUPPORTED_API_TIMEFRAMES:
            api_timeframe = timeframe
        else:
            api_timeframe = "1m"
        
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        
        all_data = []
        
        for symbol in symbols:
            # 1단계: Parquet 캐시 확인
            cached_data = self.cache_manager.load_cache(
                self.PROVIDER_NAME, symbol, api_timeframe, start_utc, end_utc
            )
            
            if cached_data is not None:
                all_data.append(cached_data)
                continue
            
            # 2단계: DB에서 기존 데이터 확인 및 누락 기간 계산
            existing_ranges = self._get_existing_data_ranges(
                symbol, api_timeframe, start_utc, end_utc
            )
            
            missing_periods = self._calculate_missing_periods(
                symbol, api_timeframe, start_utc, end_utc, existing_ranges
            )
            
            # 3단계: 누락된 데이터만 API에서 조회
            for period in missing_periods:
                print(f"🔄 {symbol} {api_timeframe} API 조회: "
                      f"{period['start'].strftime('%Y-%m-%d %H:%M')} ~ "
                      f"{period['end'].strftime('%Y-%m-%d %H:%M')}")
                
                try:
                    new_data = await self._fetch_candles_from_api(
                        symbol, api_timeframe, period["start"], period["end"]
                    )
                    
                    if not new_data.is_empty():
                        # 빈값 채우기
                        new_data = self._fill_missing_data(new_data)
                        
                        # DB에 저장
                        self.db_manager.save_market_data(
                            self.PROVIDER_NAME, symbol, api_timeframe, new_data
                        )
                        
                        print(f"💾 {symbol} {api_timeframe}: {len(new_data)}개 캔들 DB 저장 완료")
                    
                except Exception as e:
                    print(f"❌ {symbol} {api_timeframe} API 조회 실패: {e}")
            
            # DB에서 요청 범위의 모든 데이터 로드
            db_data = self.db_manager.get_market_data(
                self.PROVIDER_NAME, symbol, api_timeframe, start_utc, end_utc
            )
            
            if not db_data.is_empty():
                # 캐시 저장은 실제 조회 시에만 수행 (성능 최적화)
                # self.cache_manager.save_cache(...)  # 제거됨
                all_data.append(db_data)
        
        # 모든 심볼 데이터 결합
        if all_data:
            combined_data = pl.concat(all_data)
            
            # 리샘플링이 필요한 경우
            if timeframe != api_timeframe:
                combined_data = TimeframeUtils.resample_data(
                    combined_data, api_timeframe, timeframe
                )
            
            return combined_data.sort(["symbol", "timestamp"])
        
        # 빈 DataFrame 반환
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64
        })
    
    async def _fetch_raw_data_download_only(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Dict[str, int]:
        """다운로드 전용 원시 데이터 조회 - 캐시 저장 없음"""
        if timeframe in self.SUPPORTED_API_TIMEFRAMES:
            api_timeframe = timeframe
        else:
            api_timeframe = "1m"
        
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        
        download_counts = {}
        
        for symbol in symbols:
            total_downloaded = 0
            
            # 다운로드 전용: 캐시 확인 생략, DB에서 직접 확인
            # DB에서 기존 데이터 확인 및 누락 기간 계산
            existing_ranges = self._get_existing_data_ranges(
                symbol, api_timeframe, start_utc, end_utc
            )
            
            missing_periods = self._calculate_missing_periods(
                symbol, api_timeframe, start_utc, end_utc, existing_ranges
            )
            
            # 누락된 데이터만 API에서 조회 (캐시 저장 없음)
            for period in missing_periods:
                print(f"🔄 {symbol} {api_timeframe} API 조회: "
                      f"{period['start'].strftime('%Y-%m-%d %H:%M')} ~ "
                      f"{period['end'].strftime('%Y-%m-%d %H:%M')}")
                
                try:
                    new_data = await self._fetch_candles_from_api(
                        symbol, api_timeframe, period["start"], period["end"]
                    )
                    
                    if not new_data.is_empty():
                        # 빈값 채우기
                        new_data = self._fill_missing_data(new_data)
                        
                        # DB에만 저장 (캐시 저장 안함)
                        saved_count = self.db_manager.save_market_data(
                            self.PROVIDER_NAME, symbol, api_timeframe, new_data
                        )
                        
                        total_downloaded += saved_count
                        print(f"💾 {symbol} {api_timeframe}: {saved_count}개 캔들 DB 저장 완료")
                    
                except Exception as e:
                    print(f"❌ {symbol} {api_timeframe} API 조회 실패: {e}")
                    download_counts[symbol] = -1
                    continue
            
            # 기존 + 새로 다운로드된 데이터 총 개수
            if symbol not in download_counts or download_counts[symbol] != -1:
                total_existing = self.db_manager.get_data_count(
                    self.PROVIDER_NAME, symbol, api_timeframe
                ) if hasattr(self.db_manager, 'get_data_count') else 0
                
                download_counts[symbol] = total_existing
        
        return download_counts
    
    async def export_to_csv(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
        output_dir: str = "./binance_data"
    ) -> Dict[str, str]:
        """데이터를 CSV 파일로 내보내기"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for symbol in symbols:
            print(f"📄 {symbol} CSV 내보내기 시작...")
            
            try:
                # CSV 내보내기용 데이터 조회 (캐시 사용)
                data = await self._fetch_raw_data([symbol], start, end, timeframe)
                
                if data.is_empty():
                    print(f"⚠️ {symbol}: 데이터가 없습니다.")
                    continue
                
                # CSV 파일명 생성
                start_str = start.strftime("%Y%m%d")
                end_str = end.strftime("%Y%m%d")
                filename = f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"
                filepath = output_path / filename
                
                # CSV로 저장
                data.write_csv(str(filepath))
                exported_files[symbol] = str(filepath)
                
                print(f"✅ {symbol}: {filepath} 저장 완료 ({len(data)}개 캔들)")
                
            except Exception as e:
                print(f"❌ {symbol} CSV 내보내기 실패: {e}")
        
        return exported_files
    
    async def download_all_usdt_symbols(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
        batch_size: int = 5
    ) -> Dict[str, int]:
        """전체 USDT 페어 심볼의 데이터 다운로드"""
        print("🔍 USDT 페어 심볼 목록 조회 중...")
        usdt_symbols = await self.get_usdt_symbols()
        
        if not usdt_symbols:
            print("❌ USDT 페어 심볼을 찾을 수 없습니다.")
            return {}
        
        print(f"📊 총 {len(usdt_symbols)}개 USDT 페어 발견")
        print(f"📅 기간: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
        print(f"⏰ 타임프레임: {timeframe}")
        print(f"🔢 배치 크기: {batch_size}")
        
        downloaded_counts = {}
        failed_symbols = []
        
        # 배치 단위로 처리
        for i in range(0, len(usdt_symbols), batch_size):
            batch_symbols = usdt_symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(usdt_symbols) + batch_size - 1) // batch_size
            
            print(f"\n🔄 배치 {batch_num}/{total_batches} 처리 중: {batch_symbols}")
            
            # 배치 내 심볼들을 순차 처리 (동시 처리는 API 제한에 걸릴 수 있음)
            for symbol in batch_symbols:
                try:
                    print(f"\n📈 {symbol} 다운로드 시작...")
                    
                    # 다운로드 전용 메서드 사용 (캐시 저장 없음)
                    symbol_counts = await self._fetch_raw_data_download_only([symbol], start, end, timeframe)
                    
                    count = symbol_counts.get(symbol, 0)
                    if count > 0:
                        downloaded_counts[symbol] = count
                        print(f"✅ {symbol}: {count}개 캔들 다운로드 완료")
                    elif count == 0:
                        print(f"⚠️ {symbol}: 데이터가 없습니다.")
                        downloaded_counts[symbol] = 0
                    else:
                        # 이미 실패 처리됨
                        pass
                    
                except Exception as e:
                    print(f"❌ {symbol} 다운로드 실패: {e}")
                    failed_symbols.append(symbol)
                    downloaded_counts[symbol] = -1
            
            # 배치 간 대기 (API 제한 방지)
            if i + batch_size < len(usdt_symbols):
                print(f"⏱️ 다음 배치까지 {self.rate_limit_delay * 10}초 대기...")
                await asyncio.sleep(self.rate_limit_delay * 10)
        
        # 결과 요약
        print(f"\n📊 다운로드 완료 요약:")
        print(f"✅ 성공: {len([c for c in downloaded_counts.values() if c >= 0])}개 심볼")
        print(f"❌ 실패: {len(failed_symbols)}개 심볼")
        
        if failed_symbols:
            print(f"실패한 심볼들: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")
        
        total_candles = sum(c for c in downloaded_counts.values() if c > 0)
        print(f"📈 총 다운로드된 캔들 수: {total_candles:,}개")
        
        return downloaded_counts
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 조회"""
        return {
            "provider": self.PROVIDER_NAME,
            "db_info": {
                "db_path": str(self.db_manager.db_path),
                "provider": self.PROVIDER_NAME
            },
            "cache_info": self.cache_manager.get_cache_info() if hasattr(self.cache_manager, 'get_cache_info') else {}
        }
    
    def cleanup_storage(self, days: int = 30):
        """오래된 데이터 정리"""
        self.cache_manager.clear_old_cache(days)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 조회"""
        return self.cache_manager.get_cache_info()
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """캐시 정리"""
        self.cache_manager.clear_cache_by_criteria(
            provider=self.PROVIDER_NAME, 
            symbol=symbol, 
            timeframe=timeframe
        ) 