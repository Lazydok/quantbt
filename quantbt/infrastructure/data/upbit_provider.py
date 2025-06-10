"""
업비트 데이터 프로바이더 - SQLite + Parquet 기반

간단하고 효율적인 데이터 관리 시스템으로 업그레이드된 메인 프로바이더
"""

import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import polars as pl

from ...core.interfaces.data_provider import DataProviderBase
from ...core.utils.timeframe import TimeframeUtils
from ...config import DB_PATH, CACHE_DIR
from .database_manager import DatabaseManager
from .cache_manager import CacheManager


class UpbitDataProvider(DataProviderBase):
    """업비트 API 기반 데이터 제공자 - SQLite + Parquet"""
    
    SUPPORTED_API_TIMEFRAMES = ["1m", "1d"]
    PROVIDER_NAME = "upbit"
    
    def __init__(
        self,
        db_path: str = None,
        cache_dir: str = None,
        rate_limit_delay: float = 0.1,
        max_candles_per_request: int = 200
    ):
        super().__init__("UpbitDataProvider")
        self.base_url = "https://api.upbit.com/v1"
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
        

    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 반환 - 타임아웃 및 재시도 설정 강화"""
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
                    limit=10,  # 동시 연결 제한
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
        """datetime을 KST로 정규화"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
        else:
            return dt.astimezone(ZoneInfo("Asia/Seoul"))
    
    def _to_utc(self, dt: datetime) -> datetime:
        """KST datetime을 UTC로 변환"""
        kst_dt = self._normalize_timezone(dt)
        return kst_dt.astimezone(timezone.utc)
    
    def _get_existing_data_ranges(
        self, 
        symbol: str, 
        timeframe: str, 
        start_utc: datetime, 
        end_utc: datetime
    ) -> List[Dict[str, datetime]]:
        """DB에서 기존 데이터의 실제 범위들을 조회
        
        Args:
            symbol: 심볼
            timeframe: 타임프레임
            start_utc: 요청 시작 시간 (UTC)
            end_utc: 요청 종료 시간 (UTC)
            
        Returns:
            기존 데이터 범위 리스트 [{"start": datetime, "end": datetime}, ...]
        """
        try:
            import sqlite3
            ranges = []
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # 요청 기간 내의 모든 데이터를 시간순 조회
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
                    # UTC 타임존 정보가 없으면 추가
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    timestamps.append(timestamp)
                
                if not timestamps:
                    return ranges
                
                # 연속된 데이터 구간을 찾아서 범위로 그룹핑
                if timeframe == "1m":
                    gap_threshold = timedelta(minutes=10)  # 10분 이상 간격이면 새로운 구간 (기존: 2분)
                elif timeframe == "1d":
                    gap_threshold = timedelta(days=2)  # 2일 이상 간격이면 새로운 구간
                else:
                    gap_threshold = timedelta(hours=2)  # 기본값: 2시간
                
                current_start = timestamps[0]
                current_end = timestamps[0]
                
                for i in range(1, len(timestamps)):
                    current_ts = timestamps[i]
                    
                    # 이전 시간과의 간격 확인
                    if current_ts - current_end <= gap_threshold:
                        # 연속된 데이터 - 현재 구간 확장
                        current_end = current_ts
                    else:
                        # 간격이 있음 - 현재 구간 완료하고 새 구간 시작
                        ranges.append({
                            "start": current_start,
                            "end": current_end
                        })
                        current_start = current_ts
                        current_end = current_ts
                
                # 마지막 구간 추가
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
        """요청 기간에서 기존 데이터 범위를 제외한 빈 기간들을 계산
        
        Args:
            symbol: 심볼
            timeframe: 타임프레임  
            requested_start: 요청 시작 시간
            requested_end: 요청 종료 시간
            existing_ranges: 기존 데이터 범위들
            
        Returns:
            빈 기간 리스트 [{"start": datetime, "end": datetime}, ...]
        """
        if not existing_ranges:
            # 기존 데이터가 없으면 전체 기간이 빈 기간
            return [{
                "start": requested_start,
                "end": requested_end
            }]
        
        missing_periods = []
        
        # 기존 범위들을 시간순으로 정렬
        sorted_ranges = sorted(existing_ranges, key=lambda x: x["start"])
        
        # 시간 비교를 위해 모든 시간을 UTC로 통일
        if requested_start.tzinfo is None:
            requested_start = requested_start.replace(tzinfo=timezone.utc)
        if requested_end.tzinfo is None:
            requested_end = requested_end.replace(tzinfo=timezone.utc)
        
        current_time = requested_start
        
        for existing_range in sorted_ranges:
            range_start = existing_range["start"]
            range_end = existing_range["end"]
            
            # 기존 범위 시간들도 UTC로 통일
            if range_start.tzinfo is None:
                range_start = range_start.replace(tzinfo=timezone.utc)
            if range_end.tzinfo is None:
                range_end = range_end.replace(tzinfo=timezone.utc)
            
            # 현재 시간과 기존 범위 시작 사이에 빈 기간이 있는지 확인
            if current_time < range_start:
                missing_periods.append({
                    "start": current_time,
                    "end": range_start
                })
            
            # 다음 확인 시점을 기존 범위 끝으로 이동
            current_time = max(current_time, range_end)
        
        # 마지막 기존 범위 이후부터 요청 종료까지 빈 기간 확인
        if current_time < requested_end:
            missing_periods.append({
                "start": current_time,
                "end": requested_end
            })
        
        return missing_periods
    
    def _load_available_symbols(self) -> List[str]:
        """업비트에서 사용 가능한 심볼 로드"""
        # 주요 암호화폐 심볼들 (실제로는 API에서 조회할 수 있지만 단순화)
        return [
            "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-AVAX",
            "KRW-LINK", "KRW-DOT", "KRW-MATIC", "KRW-SOL", "KRW-DOGE",
            "KRW-SHIB", "KRW-ATOM", "KRW-NEAR", "KRW-APT", "KRW-SUI"
        ]
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """모든 타임프레임 지원 (리샘플링 처리)"""
        return TimeframeUtils.validate_timeframe(timeframe)
    
    def validate_date_range(self, start: datetime, end: datetime) -> bool:
        """날짜 범위 유효성 검증 - 타임존 처리"""
        # 타임존 정규화
        start_normalized = self._normalize_timezone(start)
        end_normalized = self._normalize_timezone(end)
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        
        return start_normalized <= end_normalized and start_normalized <= now_kst
    
    async def _check_network_connection(self) -> bool:
        """네트워크 연결 상태 확인"""
        try:
            session = await self._get_session()
            # 업비트 서버 상태 확인
            async with session.get(f"{self.base_url}/market/all") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _fetch_candles_from_api(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """업비트 API에서 캔들 데이터 조회"""
        if timeframe not in self.SUPPORTED_API_TIMEFRAMES:
            raise ValueError(f"API timeframe not supported: {timeframe}")
        
        # KST로 정규화 (업비트 API는 KST 시간 직접 사용 가능)
        start_kst = self._normalize_timezone(start)
        end_kst = self._normalize_timezone(end)
        
        session = await self._get_session()
        all_candles = []
        
        url = f"{self.base_url}/candles/minutes/1" if timeframe == "1m" else f"{self.base_url}/candles/days"
        params_base = {"market": symbol}
        current_end = end_kst
        
        # 안전장치: 최대 요청 수 제한
        max_requests = 10000  # 최대 10000 요청
        request_count = 0
        start_time = asyncio.get_event_loop().time()  # 시작 시간 기록
        
        # 일자 기반 진행률 추적
        total_days = (end_kst.date() - start_kst.date()).days + 1
        processed_dates = set()  # 처리된 날짜들을 추적
        latest_processed_date = None
        
        # 진행률 로깅을 위한 변수들
        total_candles_received = 0
        
        def update_progress():
            """진행률 업데이트 출력 (한 줄로)"""
            elapsed_time = asyncio.get_event_loop().time() - start_time
            rps = request_count / elapsed_time if elapsed_time > 0 else 0  # 초당 요청 수
            cps = total_candles_received / elapsed_time if elapsed_time > 0 else 0  # 초당 캔들 수
            
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
            # 과거 데이터 요청인지 확인
            now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
            is_historical_data = (now_kst - end_kst).days >= 1
            
            # 초기 진행률 표시
            update_progress()
            
            while current_end > start_kst and request_count < max_requests:
                params = params_base.copy()
                
                # ISO8601 + KST 포맷으로 'to' 파라미터 설정
                params["to"] = current_end.strftime("%Y-%m-%dT%H:%M:%S+09:00")
                
                params["count"] = self.max_candles_per_request
                
                request_count += 1
                
                # 재시도 로직 추가
                max_retries = 3
                retry_delay = 1
                candles = None
                
                for attempt in range(max_retries):
                    try:
                        async with session.get(url, params=params) as response:
                            response.raise_for_status()
                            candles = await response.json()
                            break  # 성공시 재시도 루프 종료
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))  # 지수적 백오프
                        else:
                            raise e
                    
                if not candles:
                    break
                
                # KST 시간으로 파싱하고 필터링 - 요청 범위 내의 데이터만 수집
                valid_candles = []
                oldest_in_batch = None
                
                for candle in candles:
                    candle_time_str = candle["candle_date_time_kst"].replace("T", " ")
                    candle_time = datetime.fromisoformat(candle_time_str)
                    candle_time = candle_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                    
                    # 요청 범위(start_kst <= candle_time <= end_kst) 내의 데이터만 수집
                    if start_kst <= candle_time <= end_kst:
                        valid_candles.append(candle)
                        # 처리된 날짜 추가
                        processed_dates.add(candle_time.date())
                        if oldest_in_batch is None or candle_time < oldest_in_batch:
                            oldest_in_batch = candle_time
                    elif candle_time < start_kst:
                        # 시작 시간보다 이전 데이터 발견시 루프 종료
                        break
                
                if valid_candles:
                    all_candles.extend(valid_candles)
                    total_candles_received += len(valid_candles)
                    
                    # 최신 처리 날짜 업데이트
                    if oldest_in_batch:
                        latest_processed_date = oldest_in_batch
                    
                    # 진행률 업데이트
                    update_progress()
                else:
                    break
                
                # 다음 요청 준비 - 이번 배치의 가장 오래된 시간 기준
                if oldest_in_batch and oldest_in_batch > start_kst:
                    current_end = oldest_in_batch - timedelta(minutes=1)
                else:
                    break
                
                await asyncio.sleep(self.rate_limit_delay)
            
            # 최대 요청 수 도달시 처리
            if request_count >= max_requests:
                print(f"\n⚠️  최대 요청 수 ({max_requests}) 도달")
            else:
                print()  # 다운로드 완료시 줄바꿈
                    
        finally:
            # 최종 통계 출력
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
            for candle in all_candles:
                timestamp_str = candle["candle_date_time_kst"].replace("T", " ")
                timestamp = datetime.fromisoformat(timestamp_str)
                timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                
                df_data.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": float(candle["opening_price"]),
                    "high": float(candle["high_price"]),
                    "low": float(candle["low_price"]),
                    "close": float(candle["trade_price"]),
                    "volume": float(candle["candle_acc_trade_volume"])
                })
            
            result_df = pl.DataFrame(df_data).sort("timestamp")
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
        """원시 데이터 조회 - 새로운 3단계 시스템"""
        # 요청된 타임프레임 확인
        if timeframe in self.SUPPORTED_API_TIMEFRAMES:
            api_timeframe = timeframe
        else:
            api_timeframe = "1m"  # 리샘플링용
        
        # UTC 변환
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
            
            # 2단계: SQLite 데이터베이스 확인 - 빈 기간 계산 방식
            # 기존 데이터 범위 조회
            existing_ranges = self._get_existing_data_ranges(
                symbol, api_timeframe, start_utc, end_utc
            )
            
            # 빈 기간 계산
            missing_periods = self._calculate_missing_periods(
                symbol, api_timeframe, start_utc, end_utc, existing_ranges
            )
            
            # 기존 데이터가 있으면 먼저 추가
            if existing_ranges:
                db_data = self.db_manager.get_market_data(
                    self.PROVIDER_NAME, symbol, api_timeframe, start_utc, end_utc
                )
                if db_data.height > 0:
                    all_data.append(db_data)
            
            # 빈 기간이 있으면 API로 요청
            if missing_periods:
                print(f"📥 {symbol} {api_timeframe}: {len(missing_periods)}개 빈 기간 발견, API 요청 시작...")
                
                # 네트워크 연결 상태 확인
                if not await self._check_network_connection():
                    continue
                
                new_data_parts = []
                
                for i, period in enumerate(missing_periods):
                    period_start_kst = period["start"].astimezone(ZoneInfo("Asia/Seoul"))
                    period_end_kst = period["end"].astimezone(ZoneInfo("Asia/Seoul"))
                    
                    print(f"   📡 빈 기간 {i+1}/{len(missing_periods)}: "
                          f"{period_start_kst.strftime('%Y-%m-%d %H:%M')} ~ "
                          f"{period_end_kst.strftime('%Y-%m-%d %H:%M')} (KST)")
                    
                    # 빈 기간에 대해서만 API 요청
                    period_data = await self._fetch_candles_from_api(
                        symbol, api_timeframe, period_start_kst, period_end_kst
                    )
                    
                    if period_data.height > 0:
                        new_data_parts.append(period_data)
                        print(f"   ✅ {period_data.height}개 캔들 수신")
                    else:
                        print(f"   ⚠️ 데이터 없음")
                
                # 새로운 데이터를 DB에 저장
                if new_data_parts:
                    combined_new_data = pl.concat(new_data_parts)
                    
                    # SQLite에 저장
                    self.db_manager.save_market_data(
                        self.PROVIDER_NAME, symbol, api_timeframe, combined_new_data
                    )
                    
                    all_data.append(combined_new_data)
                    print(f"   💾 총 {combined_new_data.height}개 새 캔들 DB 저장 완료")
            
            # 최종 데이터를 캐시에 저장 (기존 + 새 데이터)
            if all_data:
                # 현재 심볼의 모든 데이터 다시 조회 (기존 + 새 데이터)
                final_data = self.db_manager.get_market_data(
                    self.PROVIDER_NAME, symbol, api_timeframe, start_utc, end_utc
                )
                
                if final_data.height > 0:
                    # Parquet 캐시에 저장
                    self.cache_manager.save_cache(
                        self.PROVIDER_NAME, symbol, api_timeframe, start_utc, end_utc, final_data
                    )
                    
                    # 기존에 추가된 부분 데이터들을 최종 완전한 데이터로 교체
                    # (중복 제거를 위해)
                    all_data = [d for d in all_data if d.select("symbol").unique().item() != symbol]
                    all_data.append(final_data)
        
        # 모든 데이터 결합
        if all_data:
            combined_data = pl.concat(all_data).sort(["symbol", "timestamp"])
            
            # 리샘플링 필요시 처리
            if timeframe != api_timeframe:
                resampled_data = TimeframeUtils.resample_to_timeframe(
                    combined_data, timeframe, api_timeframe
                )
                return resampled_data
            
            return combined_data
        
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
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        info = {
            "provider": self.PROVIDER_NAME,
            "database": {
                "path": str(self.db_manager.db_path),
                "symbols_count": 0,
                "total_records": 0
            },
            "cache": self.cache_manager.get_cache_stats()
        }
        
        # 데이터베이스 통계 추가
        try:
            import sqlite3
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # 심볼 수
                cursor = conn.execute('''
                    SELECT COUNT(DISTINCT symbol) FROM market_data 
                    WHERE provider = ?
                ''', (self.PROVIDER_NAME,))
                info["database"]["symbols_count"] = cursor.fetchone()[0]
                
                # 총 레코드 수
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM market_data 
                    WHERE provider = ?
                ''', (self.PROVIDER_NAME,))
                info["database"]["total_records"] = cursor.fetchone()[0]
        except:
            pass
        
        return info
    
    def cleanup_storage(self, days: int = 30):
        """저장소 정리"""
        # 오래된 데이터 정리
        db_deleted = self.db_manager.cleanup_old_data(days)
        cache_deleted = self.cache_manager.cleanup_old_cache(days // 4)  # 캐시는 더 자주 정리
    
    # 기존 호환성을 위한 메서드들 추가
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 반환 (기존 호환성)"""
        storage_info = self.get_storage_info()
        cache_info = storage_info["cache"]
        
        return {
            "cache_version": "2.0",
            "cache_dir": str(Path(self.cache_manager.cache_dir)),
            "cache_structure": "sqlite_parquet", 
            "cache_files_count": cache_info.get("total_files", 0),
            "cache_size_mb": cache_info.get("total_size_mb", 0.0),
            "symbols": cache_info.get("symbols", {}),
            "total_files": cache_info.get("total_files", 0),
            "total_size_mb": cache_info.get("total_size_mb", 0.0)
        }
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """캐시 삭제 (기존 호환성)"""
        if symbol and timeframe:
            # 특정 심볼의 특정 타임프레임 캐시 삭제
            self.cache_manager.clear_cache(symbol=symbol, timeframe=timeframe)
        elif symbol:
            # 특정 심볼의 모든 캐시 삭제
            self.cache_manager.clear_cache(symbol=symbol)
        else:
            # 모든 캐시 삭제
            self.cache_manager.clear_cache()
    
    async def preload_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        force_download: bool = False
    ) -> Dict[str, int]:
        """데이터 사전 로드 (기존 호환성)"""
        if force_download:
            for symbol in symbols:
                self.clear_cache(symbol=symbol, timeframe=timeframe)
        
        data = await self._fetch_raw_data(symbols, start, end, timeframe)
        
        result = {}
        for symbol in symbols:
            symbol_data = data.filter(pl.col("symbol") == symbol)
            result[symbol] = symbol_data.height
        
        return result
    
    def get_cached_data_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """캐시된 데이터 정보 (기존 호환성)"""
        return self.get_cache_info()
    
    # 새로운 캐시 초기화 기능들
    def clear_all_cache(self) -> Dict[str, Any]:
        """모든 캐시 완전 초기화"""
        return self.cache_manager.clear_all_cache()
    
    def clear_old_cache(self, days: int = 7) -> Dict[str, Any]:
        """오래된 캐시 선택 삭제"""
        return self.cache_manager.clear_old_cache(days)
    
    def clear_cache_by_criteria(self, provider: str = None, symbol: str = None, 
                               timeframe: str = None) -> Dict[str, Any]:
        """조건별 캐시 선택 삭제 (개선된 버전)"""
        if provider is None:
            provider = self.PROVIDER_NAME  # 기본값으로 upbit 사용
        return self.cache_manager.clear_cache_by_criteria(provider, symbol, timeframe)
    
    def find_cache_by_criteria(self, provider: str = None, symbol: str = None, 
                              timeframe: str = None) -> List[Dict[str, Any]]:
        """조건별 캐시 검색"""
        if provider is None:
            provider = self.PROVIDER_NAME  # 기본값으로 upbit 사용
        return self.cache_manager.find_cache_by_criteria(provider, symbol, timeframe)
    
    def rebuild_cache_index(self) -> Dict[str, Any]:
        """캐시 인덱스 재구성 (고아 파일 정리)"""
        return self.cache_manager.rebuild_cache_index()
    
    def get_cache_health_report(self) -> Dict[str, Any]:
        """캐시 상태 건강성 보고서"""
        return self.cache_manager.get_cache_health_report()
    
    def auto_cleanup_cache_by_size(self, max_size_mb: float = 100.0) -> Dict[str, Any]:
        """캐시 크기 기반 자동 정리"""
        return self.cache_manager.auto_cleanup_by_size(max_size_mb)
    
    def cleanup_orphaned_cache(self) -> Dict[str, Any]:
        """고아 파일 정리 전용 메서드"""
        return self.cache_manager.check_and_cleanup_orphans()
    
    def print_cache_health_report(self):
        """캐시 건강성 보고서를 보기 좋게 출력 (timeframe 정보 포함)"""
        report = self.get_cache_health_report()
        
        print("=" * 60)
        print("🔍 캐시 건강성 보고서")
        print("=" * 60)
        
        if "error" in report:
            print(f"❌ 오류: {report['error']}")
            return
        
        print(f"\n📊 캐시 통계:")
        print(f"   • 캐시 파일 수: {report['total_cache_files']:,}개")
        print(f"   • 메타데이터 엔트리: {report['total_metadata_entries']:,}개")
        print(f"   • 총 크기: {report['total_size_mb']:.1f} MB")
        print(f"   • 가장 오래된 캐시: {report['oldest_cache_days']}일 전")
        
        # timeframe별 통계
        if report['timeframe_stats']:
            print(f"\n⏰ 타임프레임별 통계:")
            for timeframe, stats in report['timeframe_stats'].items():
                print(f"   • {timeframe}: {stats['cache_count']}개 캐시, "
                      f"{stats['total_records']:,}개 레코드, "
                      f"평균 {stats['avg_records']:,}개/캐시")
        
        # provider별 통계
        if report['provider_stats']:
            print(f"\n🏢 프로바이더별 통계:")
            for provider, count in report['provider_stats'].items():
                print(f"   • {provider}: {count}개 캐시")
        
        # symbol별 통계 (상위 5개만)
        if report['symbol_stats']:
            print(f"\n📈 심볼별 통계 (상위 5개):")
            for i, (symbol, count) in enumerate(list(report['symbol_stats'].items())[:5]):
                print(f"   • {symbol}: {count}개 캐시")
        
        print(f"\n🔧 캐시 상태:")
        print(f"   • 고아 파일: {report['orphaned_files']}개")
        print(f"   • 누락된 파일: {report['missing_files']}개")
        print(f"   • 캐시 효율성: {report['cache_efficiency']:.1f}%")
        
        if report['recommendations']:
            print(f"\n💡 권장사항:")
            for i, recommendation in enumerate(report['recommendations'], 1):
                print(f"   {i}. {recommendation}")
        else:
            print(f"\n✅ 캐시 상태가 양호합니다!")
        
        print("\n" + "=" * 60)
    
    def print_cache_maintenance_menu(self):
        """캐시 유지보수 메뉴 출력"""
        print("=" * 60)
        print("🛠️  CACHE MAINTENANCE MENU")
        print("=" * 60)
        
        # 현재 상태 요약
        report = self.get_cache_health_report()
        print(f"📊 현재 상태:")
        print(f"   • 캐시 파일: {report.get('total_cache_files', 0)}개")
        print(f"   • 총 크기: {report.get('total_size_mb', 0):.1f}MB")
        print(f"   • 고아 파일: {report.get('orphaned_files', 0)}개")
        print(f"   • 누락 메타데이터: {report.get('missing_files', 0)}개")
        print(f"   • 캐시 효율성: {report.get('cache_efficiency', 0):.1f}%")
        
        print(f"\n🔧 사용 가능한 명령어:")
        print(f"   1. provider.cleanup_orphaned_cache()           # 고아 파일 정리")
        print(f"   2. provider.auto_cleanup_cache_by_size(100)    # 크기 기반 정리 (100MB)")
        print(f"   3. provider.rebuild_cache_index()              # 인덱스 재구성")
        print(f"   4. provider.clear_old_cache(7)                 # 7일 이상 된 캐시 삭제")
        print(f"   5. provider.clear_cache_by_criteria(symbol='KRW-BTC')  # 조건별 삭제")
        
        # 자동 권장사항
        if report.get('orphaned_files', 0) > 0 or report.get('missing_files', 0) > 0:
            print(f"\n💡 권장: cleanup_orphaned_cache() 실행 권장")
        
        if report.get('total_size_mb', 0) > 100:
            print(f"💡 권장: auto_cleanup_cache_by_size(100) 실행 권장")
        
        print("\n" + "=" * 60)
    
    def get_cache_summary_by_timeframe(self) -> Dict[str, Any]:
        """타임프레임별 캐시 요약 정보"""
        summary = {
            "timeframes": {},
            "total_caches": 0,
            "total_size_mb": 0.0
        }
        
        try:
            # 전체 캐시 찾기
            all_caches = self.find_cache_by_criteria()
            summary["total_caches"] = len(all_caches)
            
            # 파일 크기 계산
            cache_files = list(Path(self.cache_manager.cache_dir).glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            summary["total_size_mb"] = total_size / (1024 * 1024)
            
            # 타임프레임별 그룹화
            timeframe_groups = {}
            for cache in all_caches:
                tf = cache.get('timeframe', 'unknown')
                if tf not in timeframe_groups:
                    timeframe_groups[tf] = []
                timeframe_groups[tf].append(cache)
            
            # 각 타임프레임별 통계 계산
            for timeframe, caches in timeframe_groups.items():
                total_records = sum(cache.get('record_count', 0) for cache in caches)
                avg_records = total_records / len(caches) if caches else 0
                
                # 날짜 범위 계산
                start_dates = [cache.get('start_time_utc') for cache in caches if cache.get('start_time_utc')]
                end_dates = [cache.get('end_time_utc') for cache in caches if cache.get('end_time_utc')]
                
                summary["timeframes"][timeframe] = {
                    "cache_count": len(caches),
                    "total_records": total_records,
                    "avg_records": int(avg_records),
                    "earliest_data": min(start_dates) if start_dates else None,
                    "latest_data": max(end_dates) if end_dates else None,
                    "symbols": list(set(cache.get('symbol') for cache in caches if cache.get('symbol')))
                }
        
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def print_cache_summary_by_timeframe(self):
        """타임프레임별 캐시 요약을 보기 좋게 출력"""
        summary = self.get_cache_summary_by_timeframe()
        
        print("=" * 60)
        print("⏰ 타임프레임별 캐시 요약")
        print("=" * 60)
        
        if "error" in summary:
            print(f"❌ 오류: {summary['error']}")
            return
        
        print(f"\n📊 전체 요약:")
        print(f"   • 총 캐시 수: {summary['total_caches']:,}개")
        print(f"   • 총 크기: {summary['total_size_mb']:.1f} MB")
        
        if summary["timeframes"]:
            print(f"\n⏰ 타임프레임별 상세:")
            
            for timeframe, stats in summary["timeframes"].items():
                print(f"\n📅 {timeframe} 타임프레임:")
                print(f"   • 캐시 수: {stats['cache_count']:,}개")
                print(f"   • 총 레코드: {stats['total_records']:,}개")
                print(f"   • 평균 레코드/캐시: {stats['avg_records']:,}개")
                
                if stats['earliest_data'] and stats['latest_data']:
                    print(f"   • 데이터 범위: {stats['earliest_data'][:10]} ~ {stats['latest_data'][:10]}")
                
                if stats['symbols']:
                    symbol_count = len(stats['symbols'])
                    if symbol_count <= 5:
                        print(f"   • 심볼: {', '.join(stats['symbols'])}")
                    else:
                        print(f"   • 심볼: {', '.join(stats['symbols'][:3])} 외 {symbol_count-3}개")
        
        print("\n" + "=" * 60)
    
    def cache_maintenance(self, auto_fix: bool = False) -> Dict[str, Any]:
        """캐시 자동 유지보수"""
        result = {
            "health_report": None,
            "actions_taken": [],
            "success": True,
            "errors": []
        }
        
        try:
            # 건강성 보고서 생성
            health_report = self.get_cache_health_report()
            result["health_report"] = health_report
            
            if "error" in health_report:
                result["errors"].append(f"건강성 보고서 생성 실패: {health_report['error']}")
                result["success"] = False
                return result
            
            actions_needed = []
            
            # 자동 수정이 필요한 문제들 식별
            if health_report['orphaned_files'] > 0 or health_report['missing_files'] > 0:
                actions_needed.append("index_rebuild")
            
            if health_report['oldest_cache_days'] > 30:
                actions_needed.append("old_cache_cleanup")
            
            if health_report['total_size_mb'] > 1000:  # 1GB 이상
                actions_needed.append("size_optimization")
            
            if auto_fix and actions_needed:
                print("🔧 캐시 자동 유지보수 시작...")
                
                # 인덱스 재구성
                if "index_rebuild" in actions_needed:
                    rebuild_result = self.rebuild_cache_index()
                    if rebuild_result["success"]:
                        result["actions_taken"].append("인덱스 재구성 완료")
                    else:
                        result["errors"].extend(rebuild_result["errors"])
                
                # 오래된 캐시 정리
                if "old_cache_cleanup" in actions_needed:
                    cleanup_result = self.clear_old_cache(30)  # 30일 이상 된 캐시 정리
                    if cleanup_result["success"]:
                        result["actions_taken"].append("오래된 캐시 정리 완료")
                    else:
                        result["errors"].extend(cleanup_result["errors"])
                
                # 크기 최적화 (추가 정리)
                if "size_optimization" in actions_needed:
                    cleanup_result = self.clear_old_cache(14)  # 14일 이상 된 캐시 정리
                    if cleanup_result["success"]:
                        result["actions_taken"].append("캐시 크기 최적화 완료")
                    else:
                        result["errors"].extend(cleanup_result["errors"])
                
                print("✅ 캐시 자동 유지보수 완료")
            
            elif actions_needed:
                result["actions_taken"].append(f"권장 작업: {', '.join(actions_needed)}")
                print("💡 수동으로 캐시 유지보수를 수행하세요. auto_fix=True로 설정하면 자동 수행됩니다.")
            
        except Exception as e:
            result["errors"].append(f"캐시 유지보수 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    # 새로운 데이터 현황 확인 기능들
    def get_data_summary(self) -> Dict[str, Any]:
        """전체 데이터 현황 요약"""
        summary = {
            "provider": self.PROVIDER_NAME,
            "database": {},
            "cache": {},
            "symbols": {}
        }
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # 데이터베이스 전체 통계
                cursor = conn.execute('''
                    SELECT 
                        COUNT(DISTINCT symbol) as symbol_count,
                        COUNT(*) as total_records,
                        MIN(timestamp) as earliest_date,
                        MAX(timestamp) as latest_date
                    FROM market_data 
                    WHERE provider = ?
                ''', (self.PROVIDER_NAME,))
                
                row = cursor.fetchone()
                summary["database"] = {
                    "symbol_count": row[0],
                    "total_records": row[1],
                    "earliest_date": row[2],
                    "latest_date": row[3],
                    "path": str(self.db_manager.db_path)
                }
                
                # 심볼별 통계
                cursor = conn.execute('''
                    SELECT 
                        symbol,
                        timeframe,
                        COUNT(*) as record_count,
                        MIN(timestamp) as start_date,
                        MAX(timestamp) as end_date
                    FROM market_data 
                    WHERE provider = ?
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                ''', (self.PROVIDER_NAME,))
                
                for row in cursor.fetchall():
                    symbol, timeframe, count, start_date, end_date = row
                    
                    if symbol not in summary["symbols"]:
                        summary["symbols"][symbol] = {}
                    
                    summary["symbols"][symbol][timeframe] = {
                        "record_count": count,
                        "start_date": start_date,
                        "end_date": end_date,
                        "data_source": "database"
                    }
        
        except Exception as e:
            summary["database"]["error"] = str(e)
        
        # 캐시 정보 추가
        cache_info = self.cache_manager.get_cache_stats()
        summary["cache"] = cache_info
        
        return summary
    
    def get_symbol_data_range(self, symbol: str) -> Dict[str, Any]:
        """특정 심볼의 데이터 범위 상세 정보"""
        symbol_info = {
            "symbol": symbol,
            "database": {},
            "cache": {},
            "timeframes": {}
        }
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # 데이터베이스에서 해당 심볼 정보 조회
                cursor = conn.execute('''
                    SELECT 
                        timeframe,
                        COUNT(*) as record_count,
                        MIN(timestamp) as start_date,
                        MAX(timestamp) as end_date,
                        MIN(close) as min_price,
                        MAX(close) as max_price,
                        AVG(volume) as avg_volume
                    FROM market_data 
                    WHERE provider = ? AND symbol = ?
                    GROUP BY timeframe
                    ORDER BY timeframe
                ''', (self.PROVIDER_NAME, symbol))
                
                for row in cursor.fetchall():
                    timeframe, count, start_date, end_date, min_price, max_price, avg_volume = row
                    
                    # 데이터 연속성 확인
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    
                    # 예상 캔들 수 계산
                    time_diff = end_dt - start_dt
                    if timeframe == "1m":
                        expected_count = int(time_diff.total_seconds() / 60)
                    elif timeframe == "1d":
                        expected_count = time_diff.days + 1
                    else:
                        expected_count = count
                    
                    coverage_rate = (count / expected_count * 100) if expected_count > 0 else 0
                    
                    symbol_info["timeframes"][timeframe] = {
                        "record_count": count,
                        "expected_count": expected_count,
                        "coverage_rate": round(coverage_rate, 2),
                        "start_date": start_date,
                        "end_date": end_date,
                        "price_range": {
                            "min": float(min_price) if min_price else None,
                            "max": float(max_price) if max_price else None
                        },
                        "avg_volume": float(avg_volume) if avg_volume else None,
                        "data_gaps": coverage_rate < 95  # 95% 미만이면 데이터 누락 의심
                    }
                
                # 전체 데이터베이스 통계
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM market_data 
                    WHERE provider = ? AND symbol = ?
                ''', (self.PROVIDER_NAME, symbol))
                
                symbol_info["database"]["total_records"] = cursor.fetchone()[0]
        
        except Exception as e:
            symbol_info["database"]["error"] = str(e)
        
        return symbol_info
    
    def list_available_data(self) -> Dict[str, Any]:
        """사용 가능한 모든 데이터 목록"""
        available_data = {
            "provider": self.PROVIDER_NAME,
            "symbols": [],
            "timeframes": set(),
            "date_ranges": {},
            "total_records": 0
        }
        
        try:
            import sqlite3
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # 모든 심볼 목록
                cursor = conn.execute('''
                    SELECT DISTINCT symbol FROM market_data 
                    WHERE provider = ?
                    ORDER BY symbol
                ''', (self.PROVIDER_NAME,))
                
                available_data["symbols"] = [row[0] for row in cursor.fetchall()]
                
                # 모든 타임프레임 목록
                cursor = conn.execute('''
                    SELECT DISTINCT timeframe FROM market_data 
                    WHERE provider = ?
                    ORDER BY timeframe
                ''', (self.PROVIDER_NAME,))
                
                available_data["timeframes"] = set(row[0] for row in cursor.fetchall())
                
                # 전체 날짜 범위
                cursor = conn.execute('''
                    SELECT 
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest,
                        COUNT(*) as total
                    FROM market_data 
                    WHERE provider = ?
                ''', (self.PROVIDER_NAME,))
                
                row = cursor.fetchone()
                if row[0]:  # 데이터가 있는 경우
                    available_data["date_ranges"]["earliest"] = row[0]
                    available_data["date_ranges"]["latest"] = row[1]
                    available_data["total_records"] = row[2]
        
        except Exception as e:
            available_data["error"] = str(e)
        
        return available_data
    
    def check_data_coverage(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> Dict[str, Any]:
        """특정 기간의 데이터 커버리지 확인"""
        # 타임존 정규화
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        
        coverage_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "requested_period": {
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat()
            },
            "database": {
                "available": False,
                "record_count": 0,
                "coverage_rate": 0.0,
                "missing_periods": [],
                "existing_ranges": []
            },
            "cache": {
                "available": False,
                "cache_files": []
            }
        }
        
        try:
            # 기존 데이터 범위 조회 (스마트 갭 감지 로직 활용)
            existing_ranges = self._get_existing_data_ranges(
                symbol, timeframe, start_utc, end_utc
            )
            
            # 빈 기간 계산 (스마트 갭 감지 로직 활용)
            missing_periods = self._calculate_missing_periods(
                symbol, timeframe, start_utc, end_utc, existing_ranges
            )
            
            # 데이터베이스 확인
            db_data = self.db_manager.get_market_data(
                self.PROVIDER_NAME, symbol, timeframe, start_utc, end_utc
            )
            
            if db_data.height > 0:
                coverage_info["database"]["available"] = True
                coverage_info["database"]["record_count"] = db_data.height
                
                # 예상 캔들 수 계산
                time_diff = end_utc - start_utc
                if timeframe == "1m":
                    expected_count = int(time_diff.total_seconds() / 60)
                elif timeframe == "1d":
                    expected_count = time_diff.days + 1
                else:
                    expected_count = db_data.height
                
                coverage_rate = (db_data.height / expected_count * 100) if expected_count > 0 else 0
                coverage_info["database"]["coverage_rate"] = round(coverage_rate, 2)
                coverage_info["database"]["expected_count"] = expected_count
                
                # 실제 데이터 범위
                timestamps = db_data.select("timestamp").to_series().to_list()
                if timestamps:
                    coverage_info["database"]["actual_start"] = min(timestamps).isoformat()
                    coverage_info["database"]["actual_end"] = max(timestamps).isoformat()
                
                # 기존 데이터 범위 정보 추가
                coverage_info["database"]["existing_ranges"] = [
                    {
                        "start": range_info["start"].isoformat(),
                        "end": range_info["end"].isoformat()
                    }
                    for range_info in existing_ranges
                ]
                
                # 빈 기간 정보 추가
                coverage_info["database"]["missing_periods"] = [
                    {
                        "start": period["start"].isoformat(),
                        "end": period["end"].isoformat()
                    }
                    for period in missing_periods
                ]
            else:
                # 데이터가 없으면 전체 기간이 빈 기간
                coverage_info["database"]["missing_periods"] = [{
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat()
                }]
            
            # 캐시 확인
            cached_data = self.cache_manager.load_cache(
                self.PROVIDER_NAME, symbol, timeframe, start_utc, end_utc
            )
            
            if cached_data is not None:
                coverage_info["cache"]["available"] = True
                coverage_info["cache"]["record_count"] = cached_data.height
        
        except Exception as e:
            coverage_info["error"] = str(e)
        
        return coverage_info
    
    def print_data_summary(self):
        """데이터 현황을 보기 좋게 출력"""
        summary = self.get_data_summary()
        
        print("=" * 60)
        print(f"📊 {summary['provider'].upper()} 데이터 현황 요약")
        print("=" * 60)
        
        # 데이터베이스 정보
        db_info = summary.get("database", {})
        if "error" not in db_info:
            print(f"\n💾 데이터베이스 ({db_info.get('path', 'N/A')})")
            print(f"   • 심볼 수: {db_info.get('symbol_count', 0):,}개")
            print(f"   • 총 레코드: {db_info.get('total_records', 0):,}개")
            print(f"   • 데이터 기간: {db_info.get('earliest_date', 'N/A')} ~ {db_info.get('latest_date', 'N/A')}")
        
        # 캐시 정보
        cache_info = summary.get("cache", {})
        if cache_info:
            print(f"\n💨 캐시")
            print(f"   • 캐시 파일: {cache_info.get('total_files', 0)}개")
            print(f"   • 캐시 크기: {cache_info.get('total_size_mb', 0):.1f} MB")
        
        # 심볼별 상세 정보
        symbols = summary.get("symbols", {})
        if symbols:
            print(f"\n📈 심볼별 데이터 ({len(symbols)}개)")
            print("-" * 60)
            
            for symbol, timeframes in symbols.items():
                print(f"\n{symbol}")
                for timeframe, info in timeframes.items():
                    count = info.get("record_count", 0)
                    start_date = info.get("start_date", "N/A")[:10] if info.get("start_date") else "N/A"
                    end_date = info.get("end_date", "N/A")[:10] if info.get("end_date") else "N/A"
                    print(f"   {timeframe:>4}: {count:>6,}개 ({start_date} ~ {end_date})")
        
        print("\n" + "=" * 60) 