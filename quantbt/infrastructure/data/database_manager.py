"""
SQLite 기반 시장 데이터 관리자

업비트 등 다양한 프로바이더의 시장 데이터를 SQLite에 저장하고 관리하는 클래스
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import polars as pl

# config.py에서 절대경로 import
try:
    from ...config import DB_PATH
except ImportError:
    # config.py가 없는 경우 기본값
    DB_PATH = "/home/lazydok/src/quantbt/data/quantbt.db"


class DatabaseManager:
    """SQLite 기반 시장 데이터 관리자"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DB_PATH
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """데이터베이스 및 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            # 시장 데이터 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp_utc DATETIME NOT NULL,
                    timestamp_kst DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(provider, symbol, timeframe, timestamp_utc)
                )
            ''')
            
            # 인덱스 생성
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_data_lookup 
                ON market_data(provider, symbol, timeframe, timestamp_utc)
            ''')
            
            # 캐시 메타데이터 테이블 (기존 버전)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    start_time_utc DATETIME NOT NULL,
                    end_time_utc DATETIME NOT NULL,
                    record_count INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 새로운 컬럼들 추가 (마이그레이션)
            self._migrate_cache_metadata_table(conn)
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                ON cache_metadata(cache_key)
            ''')
            
            # 새로운 인덱스들 추가
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_provider_symbol_timeframe 
                ON cache_metadata(provider, symbol, timeframe)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_timeframe 
                ON cache_metadata(timeframe)
            ''')
            
            conn.commit()
    
    def _migrate_cache_metadata_table(self, conn):
        """캐시 메타데이터 테이블 마이그레이션"""
        try:
            # 기존 컬럼 확인
            cursor = conn.execute("PRAGMA table_info(cache_metadata)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # provider 컬럼 추가
            if 'provider' not in columns:
                conn.execute('ALTER TABLE cache_metadata ADD COLUMN provider TEXT')
                print("📝 캐시 메타데이터에 provider 컬럼 추가")
            
            # symbol 컬럼 추가
            if 'symbol' not in columns:
                conn.execute('ALTER TABLE cache_metadata ADD COLUMN symbol TEXT')
                print("📝 캐시 메타데이터에 symbol 컬럼 추가")
            
            # timeframe 컬럼 추가
            if 'timeframe' not in columns:
                conn.execute('ALTER TABLE cache_metadata ADD COLUMN timeframe TEXT')
                print("📝 캐시 메타데이터에 timeframe 컬럼 추가")
            
            # data_source 컬럼 추가 (추가 정보를 위해)
            if 'data_source' not in columns:
                conn.execute('ALTER TABLE cache_metadata ADD COLUMN data_source TEXT DEFAULT "api"')
                print("📝 캐시 메타데이터에 data_source 컬럼 추가")
            
        except Exception as e:
            print(f"⚠️ 캐시 메타데이터 마이그레이션 중 오류: {e}")
    
    def get_cache_metadata_by_criteria(self, provider: str = None, symbol: str = None, 
                                     timeframe: str = None) -> List[Dict]:
        """조건별 캐시 메타데이터 조회"""
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
    
    def save_market_data(self, provider: str, symbol: str, timeframe: str, 
                        data: pl.DataFrame) -> int:
        """시장 데이터를 SQLite에 저장 (UPSERT)"""
        if data.height == 0:
            return 0
            
        # DataFrame을 딕셔너리 리스트로 변환
        records = []
        for row in data.iter_rows(named=True):
            # KST 시간을 UTC로 변환
            timestamp_kst = row['timestamp']
            if timestamp_kst.tzinfo is None:
                # naive datetime은 KST로 가정
                from zoneinfo import ZoneInfo
                timestamp_kst = timestamp_kst.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            
            timestamp_utc = timestamp_kst.astimezone(timezone.utc)
            
            records.append((
                provider, symbol, timeframe,
                timestamp_utc.replace(tzinfo=None).isoformat(),  # SQLite는 naive datetime 선호
                timestamp_kst.replace(tzinfo=None).isoformat(),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        
        with sqlite3.connect(self.db_path) as conn:
            # UPSERT 쿼리 (중복시 업데이트)
            conn.executemany('''
                INSERT OR REPLACE INTO market_data 
                (provider, symbol, timeframe, timestamp_utc, timestamp_kst,
                 open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            return len(records)
    
    def get_market_data(self, provider: str, symbol: str, timeframe: str,
                       start_utc: datetime, end_utc: datetime) -> pl.DataFrame:
        """SQLite에서 시장 데이터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT timestamp_kst, open_price, high_price, 
                       low_price, close_price, volume
                FROM market_data 
                WHERE provider = ? AND symbol = ? AND timeframe = ?
                  AND timestamp_utc BETWEEN ? AND ?
                ORDER BY timestamp_utc
            '''
            
            cursor = conn.execute(query, (
                provider, symbol, timeframe,
                start_utc.replace(tzinfo=None).isoformat(),
                end_utc.replace(tzinfo=None).isoformat()
            ))
            
            rows = cursor.fetchall()
            
        if not rows:
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64
            })
        
        # DataFrame 생성
        data = []
        for row in rows:
            timestamp_kst = datetime.fromisoformat(row[0])
            # KST 타임존 정보 추가
            from zoneinfo import ZoneInfo
            timestamp_kst = timestamp_kst.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            
            data.append({
                "timestamp": timestamp_kst,
                "symbol": symbol,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5]
            })
        
        result_df = pl.DataFrame(data)
        return result_df
    
    def get_data_range(self, provider: str, symbol: str, timeframe: str) -> Optional[tuple]:
        """데이터 범위 조회 (최소, 최대 시간)"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT MIN(timestamp_utc), MAX(timestamp_utc)
                FROM market_data 
                WHERE provider = ? AND symbol = ? AND timeframe = ?
            '''
            cursor = conn.execute(query, (provider, symbol, timeframe))
            result = cursor.fetchone()
            
            if result[0] and result[1]:
                return (
                    datetime.fromisoformat(result[0]),
                    datetime.fromisoformat(result[1])
                )
            return None
    
    def get_data_count(self, provider: str, symbol: str, timeframe: str) -> int:
        """데이터 개수 조회"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT COUNT(*) FROM market_data 
                WHERE provider = ? AND symbol = ? AND timeframe = ?
            '''
            cursor = conn.execute(query, (provider, symbol, timeframe))
            return cursor.fetchone()[0]
    
    def cleanup_old_data(self, days: int = 180):
        """오래된 데이터 정리"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM market_data 
                WHERE created_at < ?
            ''', (cutoff_date.replace(tzinfo=None).isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                print(f"🗑️ {deleted_count}개 오래된 레코드 삭제 ({days}일 이전)")
            
            return deleted_count 