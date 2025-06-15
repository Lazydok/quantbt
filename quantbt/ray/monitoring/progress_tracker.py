"""
ProgressTracker - 백테스트 진행률 추적기

실시간으로 백테스트 진행률을 추적하고 ETA를 계산하는 클래스입니다.
텍스트 기반으로 사용자 친화적인 진행상황을 표시합니다.
"""

import time
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta


class ProgressTracker:
    """백테스트 진행률 추적 및 ETA 계산 클래스
    
    주요 기능:
    - 실시간 진행률 추적
    - ETA (예상 완료 시간) 계산
    - 텍스트 기반 진행률 표시
    - 스레드 안전성 보장
    """
    
    def __init__(self, total_tasks: int):
        """ProgressTracker 초기화
        
        Args:
            total_tasks: 총 작업 수
            
        Raises:
            ValueError: total_tasks가 0 이하인 경우
        """
        if total_tasks <= 0:
            raise ValueError("총 작업 수는 0보다 커야 합니다")
            
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time: Optional[float] = None
        self.is_started = False
        
        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()
        
        # 진행률 계산을 위한 내부 변수들
        self._last_update_time: Optional[float] = None
        
    def start(self):
        """진행률 추적 시작"""
        with self._lock:
            self.start_time = time.time()
            self.is_started = True
            self._last_update_time = self.start_time
    
    def update(self, completed: int = 1):
        """진행률 업데이트
        
        Args:
            completed: 완료된 작업 수 (기본값: 1)
        """
        with self._lock:
            if not self.is_started:
                self.start()
                
            self.completed_tasks += completed
            self._last_update_time = time.time()
    
    def get_progress(self) -> Dict:
        """현재 진행률 정보 반환
        
        Returns:
            Dict: 진행률 정보
            {
                'total_tasks': int,
                'completed_tasks': int,
                'percentage': float,
                'remaining_tasks': int
            }
        """
        with self._lock:
            remaining = max(0, self.total_tasks - self.completed_tasks)
            percentage = (self.completed_tasks / self.total_tasks) * 100
            
            return {
                'total_tasks': self.total_tasks,
                'completed_tasks': self.completed_tasks,
                'percentage': percentage,
                'remaining_tasks': remaining
            }
    
    def get_eta(self) -> str:
        """예상 완료 시간 문자열 반환
        
        Returns:
            str: ETA 문자열 (예: "2분 30초", "추정 불가")
        """
        with self._lock:
            if not self.is_started or self.completed_tasks == 0:
                return "추정 불가"
                
            if self.completed_tasks >= self.total_tasks:
                return "완료"
                
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 완료된 작업 비율
            progress_ratio = self.completed_tasks / self.total_tasks
            
            # 전체 예상 시간
            estimated_total_time = elapsed_time / progress_ratio
            
            # 남은 시간
            remaining_time = estimated_total_time - elapsed_time
            
            return self._format_time(remaining_time)
    
    def get_progress_rate(self) -> float:
        """작업 진행률 (작업/초) 반환
        
        Returns:
            float: 초당 작업 처리 수
        """
        with self._lock:
            if not self.is_started or self.completed_tasks == 0:
                return 0.0
                
            elapsed_time = time.time() - self.start_time
            if elapsed_time <= 0:
                return 0.0
                
            return self.completed_tasks / elapsed_time
    
    def format_progress(self, show_bar: bool = False, bar_width: int = 40) -> str:
        """사용자 친화적 진행률 문자열 반환
        
        Args:
            show_bar: 진행률 바 표시 여부
            bar_width: 진행률 바 너비
            
        Returns:
            str: 포맷된 진행률 문자열
        """
        progress = self.get_progress()
        eta = self.get_eta()
        
        # 기본 정보
        basic_info = f"진행률: {progress['completed_tasks']}/{progress['total_tasks']} ({progress['percentage']:.1f}%)"
        
        # ETA 추가
        if eta != "추정 불가":
            basic_info += f", ETA: {eta}"
        
        # 완료 상태 확인
        if self.is_completed():
            basic_info += " - 완료"
        
        if not show_bar:
            return basic_info
            
        # 진행률 바 생성
        filled_width = int((progress['percentage'] / 100) * bar_width)
        empty_width = bar_width - filled_width
        
        progress_bar = "█" * filled_width + "░" * empty_width
        
        return f"{basic_info}\n{progress_bar}"
    
    def is_completed(self) -> bool:
        """작업 완료 여부 확인
        
        Returns:
            bool: 완료 여부
        """
        with self._lock:
            return self.completed_tasks >= self.total_tasks
    
    def reset(self):
        """진행률 추적 리셋"""
        with self._lock:
            self.completed_tasks = 0
            self.start_time = None
            self.is_started = False
            self._last_update_time = None
    
    def _format_time(self, seconds: float) -> str:
        """시간을 사용자 친화적 문자열로 변환
        
        Args:
            seconds: 초 단위 시간
            
        Returns:
            str: 포맷된 시간 문자열
        """
        if seconds < 0:
            return "완료"
            
        if seconds < 60:
            return f"{int(seconds)}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            if remaining_seconds > 0:
                return f"{minutes}분 {remaining_seconds}초"
            else:
                return f"{minutes}분"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            if remaining_minutes > 0:
                return f"{hours}시간 {remaining_minutes}분"
            else:
                return f"{hours}시간" 