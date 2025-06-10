"""
Parquet 기반 캐시 관리자

SQLite에서 조회된 데이터를 Parquet 파일로 캐시하여 빠른 재조회를 지원하는 클래스
"""

import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
import polars as pl

# config.py에서 절대경로 import
try:
    from ...config import CACHE_DIR, DB_PATH
except ImportError:
    # config.py가 없는 경우 기본값
    CACHE_DIR = "/home/lazydok/src/quantbt/data/cache/"
    DB_PATH = "/home/lazydok/src/quantbt/data/quantbt.db"


class CacheManager:
    """Parquet 기반 캐시 관리자"""
    
    def __init__(self, cache_dir: str = None, db_path: str = None):
        if cache_dir is None:
            cache_dir = CACHE_DIR
        if db_path is None:
            db_path = DB_PATH
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
    
    def _generate_cache_key(self, provider: str, symbol: str, timeframe: str,
                           start_utc: datetime, end_utc: datetime) -> str:
        """캐시 키 생성"""
        # 시간을 정규화하여 동일한 요청에 대해 같은 키 생성
        start_str = start_utc.replace(tzinfo=None).isoformat()
        end_str = end_utc.replace(tzinfo=None).isoformat()
        
        key_string = f"{provider}:{symbol}:{timeframe}:{start_str}:{end_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]  # 16자리 해시
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """캐시 파일 경로 생성"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def has_cache(self, provider: str, symbol: str, timeframe: str,
                  start_utc: datetime, end_utc: datetime) -> bool:
        """캐시 존재 여부 확인"""
        cache_key = self._generate_cache_key(provider, symbol, timeframe, start_utc, end_utc)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return False
        
        # 메타데이터에서도 확인
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT file_path FROM cache_metadata 
                WHERE cache_key = ? AND file_path = ?
            ''', (cache_key, str(cache_file)))
            
            return cursor.fetchone() is not None
    
    def save_cache(self, provider: str, symbol: str, timeframe: str,
                   start_utc: datetime, end_utc: datetime, data: pl.DataFrame, 
                   max_cache_size_mb: float = 100.0) -> bool:
        """데이터를 Parquet 캐시로 저장 (자동 크기 관리 포함)"""
        if data.height == 0:
            return False
        
        cache_key = self._generate_cache_key(provider, symbol, timeframe, start_utc, end_utc)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            # 캐시 저장 전 크기 확인 및 자동 정리
            self._check_and_manage_cache_size(max_cache_size_mb)
            
            # Parquet 파일로 저장
            data.write_parquet(cache_file)
            
            # 향상된 메타데이터 저장 (provider, symbol, timeframe 포함)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_metadata 
                    (cache_key, file_path, provider, symbol, timeframe, 
                     start_time_utc, end_time_utc, record_count, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key,
                    str(cache_file),
                    provider,
                    symbol,
                    timeframe,
                    start_utc.replace(tzinfo=None).isoformat(),
                    end_utc.replace(tzinfo=None).isoformat(),
                    data.height,
                    "api"  # 데이터 소스
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"❌ 캐시 저장 실패: {e}")
            return False
    
    def _check_and_manage_cache_size(self, max_size_mb: float):
        """캐시 크기 확인 및 자동 관리 (내부 메서드)"""
        try:
            # 현재 캐시 크기 계산
            cache_files = list(self.cache_dir.glob("*.parquet"))
            current_size_mb = sum(f.stat().st_size for f in cache_files if f.exists()) / (1024 * 1024)
            
            if current_size_mb > max_size_mb:
                print(f"🚨 캐시 크기 한계 초과: {current_size_mb:.1f}MB > {max_size_mb}MB")
                cleanup_result = self.auto_cleanup_by_size(max_size_mb * 0.8)  # 80%까지 정리
                
                if cleanup_result["success"]:
                    print(f"🧹 자동 정리 완료: {cleanup_result['space_freed_mb']:.1f}MB 확보")
                else:
                    print(f"⚠️ 자동 정리 실패: {cleanup_result['errors']}")
        except Exception as e:
            print(f"⚠️ 캐시 크기 관리 중 오류: {e}")
    
    def find_cache_by_criteria(self, provider: str = None, symbol: str = None, 
                              timeframe: str = None) -> List[dict]:
        """조건별 캐시 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                where_conditions = []
                params = []
                
                if provider:
                    where_conditions.append("provider = ?")
                    params.append(provider)
                
                if symbol:
                    where_conditions.append("symbol = ?")
                    params.append(symbol)
                
                if timeframe:
                    where_conditions.append("timeframe = ?")
                    params.append(timeframe)
                
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                query = f'''
                    SELECT cache_key, file_path, provider, symbol, timeframe,
                           start_time_utc, end_time_utc, record_count,
                           created_at, last_accessed, data_source
                    FROM cache_metadata 
                    {where_clause}
                    ORDER BY created_at DESC
                '''
                
                cursor = conn.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                results = []
                for row in cursor.fetchall():
                    result_dict = dict(zip(columns, row))
                    results.append(result_dict)
                
                return results
                
        except Exception as e:
            print(f"❌ 캐시 검색 실패: {e}")
            return []
    
    def clear_cache_by_criteria(self, provider: str = None, symbol: str = None, 
                               timeframe: str = None) -> dict:
        """조건별 캐시 선택 삭제 (개선된 버전)"""
        result = {
            "files_deleted": 0,
            "metadata_deleted": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # 조건에 맞는 캐시 찾기
            matching_caches = self.find_cache_by_criteria(provider, symbol, timeframe)
            
            if not matching_caches:
                result["success"] = True
                print(f"💡 삭제할 캐시가 없습니다 (조건: provider={provider}, symbol={symbol}, timeframe={timeframe})")
                return result
            
            with sqlite3.connect(self.db_path) as conn:
                deleted_keys = []
                
                for cache_entry in matching_caches:
                    cache_key = cache_entry['cache_key']
                    file_path = cache_entry['file_path']
                    
                    try:
                        # 파일 삭제
                        cache_file = Path(file_path)
                        if cache_file.exists():
                            cache_file.unlink()
                            result["files_deleted"] += 1
                        
                        deleted_keys.append(cache_key)
                        
                    except Exception as e:
                        result["errors"].append(f"파일 삭제 실패 {cache_key}: {str(e)}")
                
                # 메타데이터 삭제
                if deleted_keys:
                    placeholders = ','.join(['?' for _ in deleted_keys])
                    cursor = conn.execute(f'''
                        DELETE FROM cache_metadata 
                        WHERE cache_key IN ({placeholders})
                    ''', deleted_keys)
                    result["metadata_deleted"] = cursor.rowcount
                    conn.commit()
            
            result["success"] = True
            
            # 조건 정보 포함한 메시지 출력
            condition_parts = []
            if provider:
                condition_parts.append(f"provider={provider}")
            if symbol:
                condition_parts.append(f"symbol={symbol}")
            if timeframe:
                condition_parts.append(f"timeframe={timeframe}")
            
            condition_str = ", ".join(condition_parts) if condition_parts else "모든 조건"
            
            print(f"✅ 조건별 캐시 삭제 완료 ({condition_str}): "
                  f"{result['files_deleted']}개 파일, {result['metadata_deleted']}개 메타데이터 삭제")
            
        except Exception as e:
            result["errors"].append(f"조건별 캐시 삭제 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    def load_cache(self, provider: str, symbol: str, timeframe: str,
                   start_utc: datetime, end_utc: datetime) -> Optional[pl.DataFrame]:
        """Parquet 캐시에서 데이터 로드 - 범위 포함 방식으로 개선"""
        # 1단계: 정확한 매칭 시도
        cache_key = self._generate_cache_key(provider, symbol, timeframe, start_utc, end_utc)
        cache_file = self._get_cache_file_path(cache_key)
        
        if cache_file.exists():
            try:
                data = pl.read_parquet(cache_file)
                self._update_cache_access(cache_key)
                return data
            except Exception as e:
                self._delete_cache_file(cache_key)
        
        # 2단계: 범위 포함 캐시 검색 - 해시 기반 캐시 키로는 패턴 매칭 불가
        # 정확한 매칭만 사용 (캐시 키가 해시값이므로 패턴 검색 불가능)
        return None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 주석: 이 로직은 해시 기반 캐시 키에서는 작동하지 않음
                # 향후 캐시 키 구조 변경시 재활성화 가능
                cursor = conn.execute('''
                    SELECT cache_key, file_path, start_time_utc, end_time_utc
                    FROM cache_metadata 
                    WHERE start_time_utc <= ? 
                    AND end_time_utc >= ?
                    ORDER BY record_count DESC
                    LIMIT 1
                ''', (
                    start_utc.replace(tzinfo=None).isoformat(),
                    end_utc.replace(tzinfo=None).isoformat()
                ))
                
                result = cursor.fetchone()
                if result:
                    cached_key, file_path, cached_start, cached_end = result
                    cache_file_path = Path(file_path)
                    
                    if cache_file_path.exists():
                        data = pl.read_parquet(cache_file_path)
                        
                        # 요청된 범위로 필터링
                        filtered_data = data.filter(
                            (pl.col("timestamp") >= start_utc) & 
                            (pl.col("timestamp") <= end_utc)
                        )
                        
                        if filtered_data.height > 0:
                            self._update_cache_access(cached_key)
                            print(f"⚡ 범위 캐시 히트: {filtered_data.height}개 레코드 (전체 {data.height}개 중)")
                            return filtered_data
        
        except Exception as e:
            print(f"❌ 범위 캐시 검색 실패: {e}")
        
        return None
    
    def _update_cache_access(self, cache_key: str):
        """캐시 접근 시간 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE cache_metadata 
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
        except Exception:
            pass
    
    def _delete_cache_file(self, cache_key: str):
        """캐시 파일 삭제"""
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            
            # 메타데이터에서도 삭제
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM cache_metadata WHERE cache_key = ?', (cache_key,))
                conn.commit()
                
        except Exception:
            pass
    
    def cleanup_old_cache(self, days: int = 7):
        """오래된 캐시 파일 정리"""
        cutoff_date = datetime.now(timezone.utc) - timezone.timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT cache_key, file_path FROM cache_metadata 
                WHERE last_accessed < ?
            ''', (cutoff_date.replace(tzinfo=None).isoformat(),))
            
            old_caches = cursor.fetchall()
            
            for cache_key, file_path in old_caches:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    conn.execute('DELETE FROM cache_metadata WHERE cache_key = ?', (cache_key,))
                except Exception:
                    pass
            
            conn.commit()
            
        return len(old_caches)
    
    def clear_all_cache(self) -> dict:
        """모든 캐시 완전 초기화"""
        result = {
            "files_deleted": 0,
            "metadata_deleted": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # 1. 모든 parquet 파일 삭제
            cache_files = list(self.cache_dir.glob("*.parquet"))
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    result["files_deleted"] += 1
                except Exception as e:
                    result["errors"].append(f"파일 삭제 실패 {cache_file.name}: {str(e)}")
            
            # 2. 메타데이터 모두 삭제
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM cache_metadata')
                result["metadata_deleted"] = cursor.rowcount
                conn.commit()
            
            result["success"] = True
            print(f"✅ 캐시 완전 초기화 완료: {result['files_deleted']}개 파일, {result['metadata_deleted']}개 메타데이터 삭제")
            
        except Exception as e:
            result["errors"].append(f"캐시 초기화 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    def clear_old_cache(self, days: int = 7) -> dict:
        """오래된 캐시만 선택 삭제"""
        from datetime import timedelta
        
        result = {
            "files_deleted": 0,
            "metadata_deleted": 0,
            "errors": [],
            "success": False
        }
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                # 오래된 캐시 조회
                cursor = conn.execute('''
                    SELECT cache_key, file_path 
                    FROM cache_metadata 
                    WHERE last_accessed < ?
                ''', (cutoff_date.replace(tzinfo=None).isoformat(),))
                
                old_caches = cursor.fetchall()
                
                # 파일 삭제
                for cache_key, file_path in old_caches:
                    try:
                        cache_file = Path(file_path)
                        if cache_file.exists():
                            cache_file.unlink()
                            result["files_deleted"] += 1
                    except Exception as e:
                        result["errors"].append(f"파일 삭제 실패 {cache_key}: {str(e)}")
                
                # 메타데이터 삭제
                cursor = conn.execute('''
                    DELETE FROM cache_metadata 
                    WHERE last_accessed < ?
                ''', (cutoff_date.replace(tzinfo=None).isoformat(),))
                
                result["metadata_deleted"] = cursor.rowcount
                conn.commit()
            
            result["success"] = True
            print(f"✅ 오래된 캐시 삭제 완료 ({days}일 이전): {result['files_deleted']}개 파일, {result['metadata_deleted']}개 메타데이터 삭제")
            
        except Exception as e:
            result["errors"].append(f"오래된 캐시 삭제 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    def rebuild_cache_index(self) -> dict:
        """캐시 인덱스 재구성 (고아 파일 정리)"""
        result = {
            "orphaned_files": 0,
            "missing_files": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # 파일 시스템의 모든 parquet 파일
            existing_files = {f.name for f in self.cache_dir.glob("*.parquet")}
            
            with sqlite3.connect(self.db_path) as conn:
                # 메타데이터의 모든 파일
                cursor = conn.execute('SELECT cache_key, file_path FROM cache_metadata')
                metadata_entries = cursor.fetchall()
                
                metadata_files = set()
                missing_entries = []
                
                for cache_key, file_path in metadata_entries:
                    file_name = Path(file_path).name
                    metadata_files.add(file_name)
                    
                    # 파일이 실제로 존재하지 않는 메타데이터 엔트리
                    if file_name not in existing_files:
                        missing_entries.append(cache_key)
                
                # 고아 파일들 (메타데이터에 없는 파일들)
                orphaned_files = existing_files - metadata_files
                
                # 고아 파일 삭제
                for orphaned_file in orphaned_files:
                    try:
                        (self.cache_dir / orphaned_file).unlink()
                        result["orphaned_files"] += 1
                    except Exception as e:
                        result["errors"].append(f"고아 파일 삭제 실패 {orphaned_file}: {str(e)}")
                
                # 누락된 파일의 메타데이터 삭제
                if missing_entries:
                    placeholders = ','.join(['?' for _ in missing_entries])
                    cursor = conn.execute(f'''
                        DELETE FROM cache_metadata 
                        WHERE cache_key IN ({placeholders})
                    ''', missing_entries)
                    result["missing_files"] = cursor.rowcount
                    conn.commit()
            
            result["success"] = True
            print(f"✅ 캐시 인덱스 재구성 완료: 고아 파일 {result['orphaned_files']}개, 누락 메타데이터 {result['missing_files']}개 정리")
            
        except Exception as e:
            result["errors"].append(f"캐시 인덱스 재구성 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    def get_cache_health_report(self) -> dict:
        """캐시 상태 건강성 보고서 (timeframe 정보 포함)"""
        report = {
            "total_cache_files": 0,
            "total_metadata_entries": 0,
            "orphaned_files": 0,
            "missing_files": 0,
            "total_size_mb": 0.0,
            "oldest_cache_days": 0,
            "cache_efficiency": 0.0,
            "timeframe_stats": {},
            "provider_stats": {},
            "symbol_stats": {},
            "recommendations": []
        }
        
        try:
            # 파일 시스템 통계
            cache_files = list(self.cache_dir.glob("*.parquet"))
            report["total_cache_files"] = len(cache_files)
            
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            report["total_size_mb"] = total_size / (1024 * 1024)
            
            existing_files = {f.name for f in cache_files}
            
            # 메타데이터 통계
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache_metadata')
                report["total_metadata_entries"] = cursor.fetchone()[0]
                
                # timeframe별 통계
                cursor = conn.execute('''
                    SELECT timeframe, COUNT(*) as count, 
                           AVG(record_count) as avg_records,
                           SUM(record_count) as total_records
                    FROM cache_metadata 
                    WHERE timeframe IS NOT NULL
                    GROUP BY timeframe
                    ORDER BY count DESC
                ''')
                
                for row in cursor.fetchall():
                    timeframe, count, avg_records, total_records = row
                    report["timeframe_stats"][timeframe] = {
                        "cache_count": count,
                        "avg_records": int(avg_records) if avg_records else 0,
                        "total_records": int(total_records) if total_records else 0
                    }
                
                # provider별 통계
                cursor = conn.execute('''
                    SELECT provider, COUNT(*) as count
                    FROM cache_metadata 
                    WHERE provider IS NOT NULL
                    GROUP BY provider
                    ORDER BY count DESC
                ''')
                
                for row in cursor.fetchall():
                    provider, count = row
                    report["provider_stats"][provider] = count
                
                # symbol별 통계 (상위 10개)
                cursor = conn.execute('''
                    SELECT symbol, COUNT(*) as count
                    FROM cache_metadata 
                    WHERE symbol IS NOT NULL
                    GROUP BY symbol
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                
                for row in cursor.fetchall():
                    symbol, count = row
                    report["symbol_stats"][symbol] = count
                
                # 메타데이터와 실제 파일 매칭 확인
                cursor = conn.execute('SELECT cache_key, file_path FROM cache_metadata')
                metadata_entries = cursor.fetchall()
                
                metadata_files = set()
                missing_count = 0
                
                for cache_key, file_path in metadata_entries:
                    file_name = Path(file_path).name
                    metadata_files.add(file_name)
                    
                    if file_name not in existing_files:
                        missing_count += 1
                
                report["missing_files"] = missing_count
                report["orphaned_files"] = len(existing_files - metadata_files)
                
                # 가장 오래된 캐시
                cursor = conn.execute('SELECT MIN(created_at) FROM cache_metadata')
                oldest_cache = cursor.fetchone()[0]
                
                if oldest_cache:
                    oldest_date = datetime.fromisoformat(oldest_cache)
                    report["oldest_cache_days"] = (datetime.now() - oldest_date).days
            
            # 캐시 효율성 계산
            if report["total_metadata_entries"] > 0:
                report["cache_efficiency"] = (
                    (report["total_metadata_entries"] - report["missing_files"]) / 
                    report["total_metadata_entries"] * 100
                )
            
            # 권장사항 생성
            if report["orphaned_files"] > 0:
                report["recommendations"].append(f"고아 파일 {report['orphaned_files']}개 정리 필요")
            
            if report["missing_files"] > 0:
                report["recommendations"].append(f"누락된 파일 메타데이터 {report['missing_files']}개 정리 필요")
            
            if report["total_size_mb"] > 1000:  # 1GB 이상
                report["recommendations"].append("캐시 크기가 큽니다. 오래된 캐시 정리를 고려하세요")
            
            if report["oldest_cache_days"] > 30:
                report["recommendations"].append("30일 이상 된 캐시가 있습니다. 정리를 고려하세요")
            
            if report["cache_efficiency"] < 90:
                report["recommendations"].append("캐시 효율성이 낮습니다. 인덱스 재구성을 권장합니다")
            
            # timeframe별 권장사항
            if "1m" in report["timeframe_stats"] and "1d" in report["timeframe_stats"]:
                min_caches = report["timeframe_stats"]["1m"]["cache_count"]
                daily_caches = report["timeframe_stats"]["1d"]["cache_count"]
                if min_caches > daily_caches * 10:  # 1분 캐시가 너무 많은 경우
                    report["recommendations"].append("1분 데이터 캐시가 많습니다. 오래된 1분 캐시 정리를 고려하세요")
        
        except Exception as e:
            report["error"] = str(e)
        
        return report

    def get_cache_stats(self) -> dict:
        """캐시 통계 정보"""
        stats = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "symbols": {},
            "oldest_cache": None,
            "newest_cache": None
        }
        
        try:
            # 파일 시스템에서 캐시 파일 통계
            cache_files = list(self.cache_dir.glob("*.parquet"))
            stats["total_files"] = len(cache_files)
            
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            stats["total_size_mb"] = total_size / (1024 * 1024)
            
            # 메타데이터에서 상세 통계
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest,
                        COUNT(*) as meta_count
                    FROM cache_metadata
                ''')
                
                result = cursor.fetchone()
                if result and result[0]:
                    stats["oldest_cache"] = result[0]
                    stats["newest_cache"] = result[1]
                    stats["metadata_entries"] = result[2]
        
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def auto_cleanup_by_size(self, max_size_mb: float = 100.0) -> dict:
        """캐시 크기 기반 자동 정리 - 크기 초과시 오래된 순서로 삭제"""
        result = {
            "initial_size_mb": 0.0,
            "final_size_mb": 0.0,
            "files_deleted": 0,
            "metadata_deleted": 0,
            "space_freed_mb": 0.0,
            "success": False,
            "errors": []
        }
        
        try:
            # 현재 캐시 크기 계산
            cache_files = list(self.cache_dir.glob("*.parquet"))
            initial_size = sum(f.stat().st_size for f in cache_files if f.exists())
            result["initial_size_mb"] = initial_size / (1024 * 1024)
            
            if result["initial_size_mb"] <= max_size_mb:
                result["success"] = True
                result["final_size_mb"] = result["initial_size_mb"]
                return result
            
            print(f"📦 캐시 크기 확인: {result['initial_size_mb']:.1f}MB (한계: {max_size_mb}MB)")
            print(f"🧹 크기 초과로 오래된 캐시부터 자동 정리 시작...")
            
            # 메타데이터에서 오래된 순서로 캐시 조회
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT cache_key, file_path, created_at, record_count
                    FROM cache_metadata 
                    ORDER BY last_accessed ASC, created_at ASC
                ''')
                
                old_caches = cursor.fetchall()
                
                current_size = result["initial_size_mb"]
                deleted_keys = []
                
                for cache_key, file_path, created_at, record_count in old_caches:
                    if current_size <= max_size_mb:
                        break
                    
                    try:
                        # 파일 크기 확인 후 삭제
                        cache_file = Path(file_path)
                        if cache_file.exists():
                            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                            cache_file.unlink()
                            current_size -= file_size_mb
                            result["files_deleted"] += 1
                            result["space_freed_mb"] += file_size_mb
                            
                            print(f"   🗑️ 삭제: {cache_key} (크기: {file_size_mb:.1f}MB, 레코드: {record_count:,}개)")
                        
                        deleted_keys.append(cache_key)
                        
                    except Exception as e:
                        result["errors"].append(f"파일 삭제 실패 {cache_key}: {str(e)}")
                
                # 메타데이터 삭제
                if deleted_keys:
                    placeholders = ','.join(['?' for _ in deleted_keys])
                    cursor = conn.execute(f'''
                        DELETE FROM cache_metadata 
                        WHERE cache_key IN ({placeholders})
                    ''', deleted_keys)
                    result["metadata_deleted"] = cursor.rowcount
                    conn.commit()
            
            # 최종 크기 계산
            remaining_files = list(self.cache_dir.glob("*.parquet"))
            final_size = sum(f.stat().st_size for f in remaining_files if f.exists())
            result["final_size_mb"] = final_size / (1024 * 1024)
            
            result["success"] = True
            print(f"✅ 크기 기반 정리 완료: {result['files_deleted']}개 파일 삭제, "
                  f"{result['space_freed_mb']:.1f}MB 확보 "
                  f"({result['initial_size_mb']:.1f}MB → {result['final_size_mb']:.1f}MB)")
            
        except Exception as e:
            result["errors"].append(f"크기 기반 정리 실패: {str(e)}")
            result["success"] = False
        
        return result
    
    def check_and_cleanup_orphans(self) -> dict:
        """고아 파일 정리 전용 메서드"""
        result = {
            "orphaned_files_deleted": 0,
            "orphaned_metadata_deleted": 0,
            "success": False,
            "errors": [],
            "orphaned_files_list": [],
            "orphaned_metadata_list": []
        }
        
        try:
            print("🔍 고아 파일 및 메타데이터 검사 시작...")
            
            # 파일 시스템의 모든 parquet 파일
            existing_files = {}
            for f in self.cache_dir.glob("*.parquet"):
                existing_files[f.name] = f
            
            with sqlite3.connect(self.db_path) as conn:
                # 메타데이터의 모든 파일
                cursor = conn.execute('SELECT cache_key, file_path FROM cache_metadata')
                metadata_entries = cursor.fetchall()
                
                metadata_files = set()
                orphaned_metadata_keys = []
                
                for cache_key, file_path in metadata_entries:
                    file_name = Path(file_path).name
                    metadata_files.add(file_name)
                    
                    # 파일이 실제로 존재하지 않는 메타데이터 엔트리
                    if file_name not in existing_files:
                        orphaned_metadata_keys.append(cache_key)
                        result["orphaned_metadata_list"].append({
                            "cache_key": cache_key,
                            "file_path": file_path
                        })
                
                # 고아 파일들 (메타데이터에 없는 파일들)
                orphaned_file_names = set(existing_files.keys()) - metadata_files
                
                if orphaned_file_names:
                    print(f"🗑️ 고아 파일 {len(orphaned_file_names)}개 발견:")
                    for file_name in orphaned_file_names:
                        file_path = existing_files[file_name]
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        result["orphaned_files_list"].append({
                            "file_name": file_name,
                            "file_path": str(file_path),
                            "size_mb": file_size_mb
                        })
                        print(f"   📄 {file_name} (크기: {file_size_mb:.1f}MB)")
                
                if orphaned_metadata_keys:
                    print(f"🗃️ 고아 메타데이터 {len(orphaned_metadata_keys)}개 발견:")
                    for key in orphaned_metadata_keys[:5]:  # 최대 5개만 표시
                        print(f"   🔑 {key}")
                    if len(orphaned_metadata_keys) > 5:
                        print(f"   ... (+{len(orphaned_metadata_keys) - 5}개 더)")
                
                # 고아 파일 삭제
                for file_name in orphaned_file_names:
                    try:
                        existing_files[file_name].unlink()
                        result["orphaned_files_deleted"] += 1
                    except Exception as e:
                        result["errors"].append(f"고아 파일 삭제 실패 {file_name}: {str(e)}")
                
                # 고아 메타데이터 삭제
                if orphaned_metadata_keys:
                    placeholders = ','.join(['?' for _ in orphaned_metadata_keys])
                    cursor = conn.execute(f'''
                        DELETE FROM cache_metadata 
                        WHERE cache_key IN ({placeholders})
                    ''', orphaned_metadata_keys)
                    result["orphaned_metadata_deleted"] = cursor.rowcount
                    conn.commit()
            
            result["success"] = True
            
            if result["orphaned_files_deleted"] > 0 or result["orphaned_metadata_deleted"] > 0:
                print(f"✅ 고아 파일 정리 완료: "
                      f"파일 {result['orphaned_files_deleted']}개, "
                      f"메타데이터 {result['orphaned_metadata_deleted']}개 삭제")
            else:
                print("✅ 고아 파일 없음 - 캐시가 정상 상태입니다")
            
        except Exception as e:
            result["errors"].append(f"고아 파일 정리 실패: {str(e)}")
            result["success"] = False
        
        return result 