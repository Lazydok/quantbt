"""
Ray 데이터 매니저

Zero-copy 데이터 공유 및 Object Store 기반 분산 데이터 관리
"""

import ray
import polars as pl
import time
import gc
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataInfo:
    """저장된 데이터 정보"""
    key: str
    ref: ray.ObjectRef
    size: int
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


@ray.remote
class RayDataManager:
    """Ray Object Store 기반 데이터 관리자
    
    Zero-copy 데이터 공유 및 메모리 효율적 분산 데이터 관리를 제공합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """데이터 매니저 초기화
        
        Args:
            config: 데이터 매니저 설정
        """
        self.config = config or {}
        self._data_registry: Dict[str, DataInfo] = {}
        self._cache: Dict[str, ray.ObjectRef] = {}
        self._gc_stats = {
            'gc_count': 0,
            'last_gc_time': 0.0
        }
        
        # 실제 데이터 로딩을 위한 프로바이더
        self._upbit_provider = None
        
        # 설정 값 추출
        self.max_object_store_memory = self.config.get('max_object_store_memory', 500000000)
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_type = self.config.get('compression_type', 'lz4')
        self.memory_threshold = self.config.get('memory_threshold', 0.8)
        self.gc_interval = self.config.get('gc_interval', 60)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size_mb = self.config.get('cache_size_mb', 100)
        
        logger.info(f"Ray 데이터 매니저 초기화 완료: {len(self._data_registry)}개 데이터")
    
    def load_real_data(self, symbols: List[str], start_date: Union[str, datetime], 
                      end_date: Union[str, datetime], timeframe: str) -> ray.ObjectRef:
        """실제 업비트 데이터 로딩 및 Ray Object Store 저장 (개선된 제로카피 방식)
        
        Args:
            symbols: 심볼 목록 (예: ['KRW-BTC'])
            start_date: 시작 날짜 (YYYY-MM-DD 또는 datetime)
            end_date: 종료 날짜 (YYYY-MM-DD 또는 datetime)
            timeframe: 시간 프레임 (1h, 1d 등)
            
        Returns:
            ray.ObjectRef: 데이터 참조
        """
        # 캐시 키 생성 (datetime을 문자열로 변환, 일관성 있게)
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_date, datetime) else end_date
        cache_key = f"{symbols}_{start_str}_{end_str}_{timeframe}"
        
        # 캐시 확인
        if cache_key in self._data_registry:
            # 접근 통계 업데이트
            data_info = self._data_registry[cache_key]
            data_info.access_count += 1
            data_info.last_accessed = time.time()
            logger.info(f"캐시에서 데이터 반환: {cache_key} (접근횟수: {data_info.access_count})")
            return data_info.ref
        
        # 실제 데이터 로딩
        logger.info(f"실제 데이터 로딩 시작: {cache_key}")
        loading_start = time.time()
        
        # UpbitDataProvider 초기화 (지연 로딩)
        if self._upbit_provider is None:
            from quantbt.infrastructure.data.upbit_provider import UpbitDataProvider
            self._upbit_provider = UpbitDataProvider()
        
        # 문자열을 datetime으로 변환
        if isinstance(start_date, str):
            # 다양한 포맷 지원
            if ' ' in start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            else:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            # 다양한 포맷 지원
            if ' ' in end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            else:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        # async 메서드 호출
        try:
            raw_data = asyncio.run(self._upbit_provider.get_data(
                symbols=symbols,
                start=start_dt,
                end=end_dt,
                timeframe=timeframe
            ))
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            raise
        
        # 데이터 전처리 (QuantBT 호환)
        processed_data = self._preprocess_for_quantbt(raw_data)
        
        # 제로카피 방식으로 Ray Object Store에 저장
        data_ref = ray.put(processed_data)
        
        # 데이터 정보 등록 (Polars 방식)
        data_size = processed_data.estimated_size() if hasattr(processed_data, 'estimated_size') else 0
        current_time = time.time()
        loading_time = current_time - loading_start
        
        data_info = DataInfo(
            key=cache_key,
            ref=data_ref,
            size=data_size,
            created_at=current_time,
            access_count=1,
            last_accessed=current_time
        )
        
        self._data_registry[cache_key] = data_info
        
        # 캐시에도 저장 (활성화된 경우)
        if self.enable_caching:
            self._cache[cache_key] = data_ref
        
        logger.info(f"✅ 실제 데이터 로딩 완료: {cache_key}")
        logger.info(f"   - 데이터 크기: {data_size:,} bytes")
        logger.info(f"   - 로딩 시간: {loading_time:.2f}초")
        logger.info(f"   - 데이터 행 수: {len(processed_data):,}")
        
        return data_ref
    
    def _preprocess_for_quantbt(self, data: Any) -> pl.DataFrame:
        """QuantBT 엔진 호환 형식으로 데이터 전처리 (Polars 전용)
        
        Args:
            data: 원본 데이터 (Polars DataFrame)
            
        Returns:
            pl.DataFrame: 전처리된 Polars DataFrame
        """
        if data is None or len(data) == 0:
            raise ValueError("데이터가 비어있습니다")
        
        # 데이터가 이미 Polars DataFrame인지 확인
        if isinstance(data, pl.DataFrame):
            # 이미 Polars DataFrame인 경우
            processed_data = data
        elif hasattr(data, 'to_pandas') and not isinstance(data, pl.DataFrame):
            # Pandas DataFrame을 Polars로 변환
            processed_data = pl.from_pandas(data)
        else:
            # 기타 타입의 경우 Polars로 변환 시도
            processed_data = pl.DataFrame(data)
        
        # 필수 컬럼 확인
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 시간순 정렬
        processed_data = processed_data.sort("timestamp")
        
        # 결측값 처리 (forward fill)
        processed_data = processed_data.fill_null(strategy="forward").drop_nulls()
        
        # 데이터 타입 최적화 (Polars 방식)
        processed_data = processed_data.with_columns([
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64)
        ])
        
        logger.debug(f"데이터 전처리 완료: {len(processed_data)}행, {list(processed_data.columns)}")
        return processed_data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환
        
        Returns:
            Dict[str, Any]: 캐시 통계 정보
        """
        return {
            'cache_size': len(self._data_registry),
            'cached_keys': list(self._data_registry.keys()),
            'total_data_size': sum(info.size for info in self._data_registry.values()),
            'total_access_count': sum(info.access_count for info in self._data_registry.values())
        }
    
    def is_initialized(self) -> bool:
        """초기화 상태 확인
        
        Returns:
            bool: 초기화 여부
        """
        return ray.is_initialized()
    
    def get_stored_data_count(self) -> int:
        """저장된 데이터 개수 조회
        
        Returns:
            int: 저장된 데이터 개수
        """
        return len(self._data_registry)
    
    def get_memory_usage(self) -> int:
        """현재 메모리 사용량 조회 (바이트)
        
        Returns:
            int: 메모리 사용량
        """
        try:
            cluster_resources = ray.cluster_resources()
            object_store_memory = cluster_resources.get('object_store_memory', 0)
            available_resources = ray.available_resources()
            available_object_store = available_resources.get('object_store_memory', 0)
            
            used_memory = object_store_memory - available_object_store
            return int(used_memory)
        except Exception as e:
            logger.warning(f"메모리 사용량 조회 실패: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, int]:
        """메모리 통계 조회
        
        Returns:
            Dict[str, int]: 메모리 통계 정보
        """
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            total_memory = int(cluster_resources.get('object_store_memory', 0))
            available_memory = int(available_resources.get('object_store_memory', 0))
            used_memory = total_memory - available_memory
            
            return {
                'total_memory': total_memory,
                'used_memory': used_memory,
                'available_memory': available_memory
            }
        except Exception as e:
            logger.warning(f"메모리 통계 조회 실패: {e}")
            return {
                'total_memory': self.max_object_store_memory,
                'used_memory': 0,
                'available_memory': self.max_object_store_memory
            }
    
    def store_data(self, data_key: str, data: pl.DataFrame, overwrite: bool = True) -> ray.ObjectRef:
        """데이터를 Object Store에 저장
        
        Args:
            data_key: 데이터 키
            data: 저장할 데이터
            overwrite: 덮어쓰기 허용 여부
            
        Returns:
            ray.ObjectRef: 데이터 참조
            
        Raises:
            ValueError: 키가 이미 존재하고 덮어쓰기가 금지된 경우
        """
        # 중복 키 확인
        if data_key in self._data_registry and not overwrite:
            raise ValueError(f"Data key '{data_key}' already exists and overwrite is disabled")
        
        # 데이터를 Ray Object Store에 저장
        data_ref = ray.put(data)
        
        # 데이터 정보 등록
        data_size = data.estimated_size()
        current_time = time.time()
        
        data_info = DataInfo(
            key=data_key,
            ref=data_ref,
            size=data_size,
            created_at=current_time,
            access_count=0,
            last_accessed=current_time
        )
        
        self._data_registry[data_key] = data_info
        
        # 캐시에도 저장 (활성화된 경우)
        if self.enable_caching:
            self._cache[data_key] = data_ref
        
        logger.debug(f"데이터 저장 완료: {data_key} ({data_size:,} bytes)")
        return data_ref
    
    def has_data(self, data_key: str) -> bool:
        """데이터 존재 여부 확인
        
        Args:
            data_key: 데이터 키
            
        Returns:
            bool: 데이터 존재 여부
        """
        return data_key in self._data_registry
    
    def get_data_size(self, data_key: str) -> int:
        """데이터 크기 조회
        
        Args:
            data_key: 데이터 키
            
        Returns:
            int: 데이터 크기 (바이트)
            
        Raises:
            KeyError: 데이터가 존재하지 않는 경우
        """
        if data_key not in self._data_registry:
            raise KeyError(f"Data key '{data_key}' not found")
        
        return self._data_registry[data_key].size
    
    def get_data_reference(self, data_key: str) -> ray.ObjectRef:
        """데이터 참조 조회
        
        Args:
            data_key: 데이터 키
            
        Returns:
            ray.ObjectRef: 데이터 참조
            
        Raises:
            KeyError: 데이터가 존재하지 않는 경우
        """
        if data_key not in self._data_registry:
            raise KeyError(f"Data key '{data_key}' not found")
        
        # 접근 통계 업데이트
        data_info = self._data_registry[data_key]
        data_info.access_count += 1
        data_info.last_accessed = time.time()
        
        return data_info.ref
    
    def remove_data(self, data_key: str) -> bool:
        """데이터 제거
        
        Args:
            data_key: 데이터 키
            
        Returns:
            bool: 제거 성공 여부
        """
        if data_key not in self._data_registry:
            return False
        
        # 레지스트리에서 제거
        del self._data_registry[data_key]
        
        # 캐시에서도 제거
        if data_key in self._cache:
            del self._cache[data_key]
        
        logger.debug(f"데이터 제거 완료: {data_key}")
        return True
    
    def store_data_chunked(self, data_key: str, data: pl.DataFrame) -> List[ray.ObjectRef]:
        """데이터를 청크 단위로 분할하여 저장
        
        Args:
            data_key: 데이터 키
            data: 저장할 데이터
            
        Returns:
            List[ray.ObjectRef]: 청크 참조 리스트
        """
        chunk_refs = []
        total_rows = data.shape[0]
        
        for i in range(0, total_rows, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_rows)
            chunk_data = data.slice(i, end_idx - i)
            
            chunk_key = f"{data_key}_chunk_{i//self.chunk_size}"
            chunk_ref = self.store_data(chunk_key, chunk_data)
            chunk_refs.append(chunk_ref)
        
        logger.info(f"청크 저장 완료: {data_key} ({len(chunk_refs)}개 청크)")
        return chunk_refs
    
    def get_data_from_chunks(self, data_key: str) -> pl.DataFrame:
        """청크로 분할된 데이터를 재조립
        
        Args:
            data_key: 데이터 키
            
        Returns:
            pl.DataFrame: 재조립된 데이터
        """
        chunk_dfs = []
        chunk_idx = 0
        
        while True:
            chunk_key = f"{data_key}_chunk_{chunk_idx}"
            if not self.has_data(chunk_key):
                break
            
            chunk_ref = self.get_data_reference(chunk_key)
            chunk_data = ray.get(chunk_ref)
            chunk_dfs.append(chunk_data)
            chunk_idx += 1
        
        if not chunk_dfs:
            raise KeyError(f"No chunks found for data key '{data_key}'")
        
        # 청크들을 연결하여 원본 데이터 재구성
        reassembled_data = pl.concat(chunk_dfs)
        logger.debug(f"청크 재조립 완료: {data_key} ({len(chunk_dfs)}개 청크)")
        
        return reassembled_data
    
    def get_gc_stats(self) -> Dict[str, Union[int, float]]:
        """가비지 컬렉션 통계 조회
        
        Returns:
            Dict[str, Union[int, float]]: GC 통계
        """
        return self._gc_stats.copy()
    
    def force_garbage_collection(self) -> None:
        """강제 가비지 컬렉션 실행"""
        gc.collect()
        self._gc_stats['gc_count'] += 1
        self._gc_stats['last_gc_time'] = time.time()
        
        logger.debug("강제 가비지 컬렉션 실행 완료")
    
    def get_cached_data_reference(self, symbols: List[str], start_date: Union[str, datetime], 
                                 end_date: Union[str, datetime], timeframe: str) -> Optional[ray.ObjectRef]:
        """캐시된 데이터 참조 조회 (제로카피 접근)
        
        Args:
            symbols: 심볼 목록
            start_date: 시작 날짜
            end_date: 종료 날짜
            timeframe: 시간 프레임
            
        Returns:
            Optional[ray.ObjectRef]: 캐시된 데이터 참조 (없으면 None)
        """
        # 캐시 키 생성 (일관성 있게)
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_date, datetime) else end_date
        cache_key = f"{symbols}_{start_str}_{end_str}_{timeframe}"
        
        if cache_key in self._data_registry:
            # 접근 통계 업데이트
            data_info = self._data_registry[cache_key]
            data_info.access_count += 1
            data_info.last_accessed = time.time()
            logger.debug(f"캐시 히트: {cache_key} (접근횟수: {data_info.access_count})")
            return data_info.ref
        
        logger.debug(f"캐시 미스: {cache_key}")
        return None
    
    def store_shared_data(self, data_key: str, data_ref: ray.ObjectRef, data_size: int = 0) -> bool:
        """외부에서 생성된 공유 데이터 참조를 저장 (제로카피)
        
        Args:
            data_key: 데이터 키
            data_ref: 이미 생성된 데이터 참조
            data_size: 데이터 크기 (선택사항)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            current_time = time.time()
            
            data_info = DataInfo(
                key=data_key,
                ref=data_ref,
                size=data_size,
                created_at=current_time,
                access_count=0,
                last_accessed=current_time
            )
            
            self._data_registry[data_key] = data_info
            
            # 캐시에도 저장 (활성화된 경우)
            if self.enable_caching:
                self._cache[data_key] = data_ref
            
            logger.info(f"공유 데이터 참조 저장 완료: {data_key}")
            return True
            
        except Exception as e:
            logger.error(f"공유 데이터 참조 저장 실패: {e}")
            return False 