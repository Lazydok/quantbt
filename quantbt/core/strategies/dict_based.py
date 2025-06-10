"""
Dict 기반 고성능 거래 전략 기본 클래스

List[Dict] → Dict 직접 전달 방식으로 최고 성능을 달성하는 전략
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..entities.order import Order, OrderSide, OrderType

class DictTradingStrategy(ABC):
    """Dict 기반 거래 전략 기본 클래스
    
    핵심 특징:
    - Zero Conversion: 중간 변환 완전 제거
    - Direct Access: Dict 데이터 직접 활용
    - Maximum Performance: 순수 Python 성능 극대화
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, 
                 position_size_pct: float = 0.8, max_positions: int = 1):
        """Dict 기반 전략 초기화
        
        Args:
            name: 전략 이름
            config: 전략 설정
            position_size_pct: 포지션 크기 비율 (0.0 ~ 1.0)
            max_positions: 최대 포지션 수
        """
        self.name = name
        self.config = config or {}
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.broker = None
        self.initial_cash = 10_000_000
        
    def set_broker(self, broker):
        """브로커 설정"""
        self.broker = broker
        self.initial_cash = broker.initial_cash
    
    def precompute_indicators_dict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dict 기반 지표 계산
        
        Args:
            data: 원본 캔들 데이터 List[Dict]
            
        Returns:
            지표가 추가된 데이터 List[Dict]
        """
        # 심볼별로 그룹화
        symbol_groups = {}
        for row in data:
            symbol = row['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(row)
        
        # 각 심볼별로 지표 계산
        enriched_data = []
        for symbol, symbol_data in symbol_groups.items():
            # 시간순 정렬
            symbol_data.sort(key=lambda x: x['timestamp'])
            
            # 지표 계산
            enriched_symbol_data = self._compute_indicators_for_symbol_dict(symbol_data)
            enriched_data.extend(enriched_symbol_data)
        
        # 전체 데이터를 시간순으로 정렬
        enriched_data.sort(key=lambda x: x['timestamp'])
        return enriched_data
    
    @abstractmethod
    def _compute_indicators_for_symbol_dict(self, symbol_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """심볼별 지표 계산 (서브클래스에서 구현)
        
        Args:
            symbol_data: 특정 심볼의 시간순 정렬된 캔들 데이터
            
        Returns:
            지표가 추가된 심볼 데이터
        """
        pass
    
    @abstractmethod
    def generate_signals_dict(self, current_data: Dict[str, Any], 
                            historical_data: List[Dict[str, Any]] = None) -> List[Order]:
        """Dict 기반 신호 생성 (서브클래스에서 구현)
        
        Args:
            current_data: 현재 캔들 데이터 Dict
            historical_data: 과거 데이터 (옵션)
            
        Returns:
            생성된 주문 리스트
        """
        pass
    
    def calculate_sma_dict(self, prices: List[float], window: int) -> List[float]:
        """Dict 기반 단순이동평균 계산
        
        Args:
            prices: 가격 리스트
            window: 이동평균 윈도우 크기
            
        Returns:
            이동평균 값 리스트 (초기 window-1개는 None)
        """
        sma_values = []
        for i in range(len(prices)):
            if i < window - 1:
                sma_values.append(None)
            else:
                window_prices = prices[i - window + 1:i + 1]
                sma = sum(window_prices) / len(window_prices)
                sma_values.append(sma)
        return sma_values
    
    def get_current_positions(self) -> Dict[str, float]:
        """현재 포지션 조회
        
        Returns:
            심볼별 포지션 수량 Dict
        """
        if self.broker:
            portfolio = self.broker.get_portfolio()
            positions = {}
            
            # Portfolio의 positions는 Position 객체들의 딕셔너리
            for symbol, position in portfolio.positions.items():
                if hasattr(position, 'quantity') and position.quantity != 0:
                    positions[symbol] = position.quantity
                    
            return positions
        return {}
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 가치 조회
        
        Returns:
            현재 포트폴리오 총 가치
        """
        if self.broker:
            return self.broker.get_portfolio().equity
        return self.initial_cash
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float) -> float:
        """포지션 크기 계산
        
        Args:
            symbol: 심볼
            price: 현재 가격
            portfolio_value: 포트폴리오 가치
            
        Returns:
            계산된 포지션 수량
        """
        target_value = portfolio_value * self.position_size_pct
        return target_value / price if price > 0 else 0.0 