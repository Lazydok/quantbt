"""
멀티 타임프레임 SMA 전략

1분봉과 5분봉의 서로 다른 지표를 활용한 크로스 타임프레임 전략
"""

from typing import List, Dict, Any
import polars as pl
from ...core.interfaces.strategy import MultiTimeframeTradingStrategy
from ...core.entities.order import Order, OrderSide, OrderType


class MultiTimeframeSMAStrategy(MultiTimeframeTradingStrategy):
    """멀티 타임프레임 SMA 전략
    
    특징:
    - 1분봉: SMA + 거래량 지표 (단기 진입 신호)
    - 5분봉: SMA + RSI + 변동성 (중기 트렌드 필터)
    - 크로스 타임프레임: 모든 조건을 만족해야 거래
    
    매수 조건:
    - 1분봉: 가격 > SMA10, SMA10 > SMA20, 거래량 증가
    - 5분봉: SMA5 > SMA15 (상승 추세), RSI 30-70 (과매수/과매도 아님)
    
    매도 조건:
    - 1분봉: 가격 < SMA20 (지지선 이탈)
    - 5분봉: SMA5 < SMA15 (하락 추세) 또는 RSI 극값
    """
    
    def __init__(self):
        timeframe_configs = {
            "1m": {
                "sma_windows": [10, 20],
                "volume_window": 20
            },
            "5m": {
                "sma_windows": [5, 15], 
                "rsi_period": 14,
                "volatility_window": 10
            }
        }
        
        super().__init__(
            name="MultiTimeframeSMA",
            timeframe_configs=timeframe_configs,
            primary_timeframe="1m",
            position_size_pct=0.8,
            max_positions=1
        )
    
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str, 
        config: Dict[str, Any]
    ) -> pl.DataFrame:
        """심볼별 + 타임프레임별 지표 계산 (벡터 연산)"""
        
        data = symbol_data.sort("timestamp")
        indicators = []
        
        if timeframe == "1m":
            # 1분봉: SMA + 볼륨 지표
            indicators.extend([
                pl.col("close").rolling_mean(10).alias("sma_10"),
                pl.col("close").rolling_mean(20).alias("sma_20"),
                pl.col("volume").rolling_mean(20).alias("volume_sma"),
                (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio")
            ])
            
        elif timeframe == "5m":
            # 5분봉: SMA + RSI + 변동성
            indicators.extend([
                pl.col("close").rolling_mean(5).alias("sma_5"),
                pl.col("close").rolling_mean(15).alias("sma_15"),
                self.calculate_rsi(pl.col("close"), 14).alias("rsi_14"),
                pl.col("close").rolling_std(10).alias("volatility_10")
            ])
        
        return data.with_columns(indicators)
    
    def calculate_rsi(self, prices: pl.Expr, period: int = 14) -> pl.Expr:
        """RSI 계산 (최적화된 Polars 벡터 연산)
        
        🚀 성능 최적화: map_elements → pl.when으로 개선
        
        Args:
            prices: 가격 데이터 (Polars Expression)
            period: RSI 계산 기간 (기본값: 14)
            
        Returns:
            RSI 값 (0-100 범위)
        """
        price_change = prices.diff(1)
        
        # 🚀 최적화: 순수 벡터 연산 사용
        gains = pl.when(price_change > 0).then(price_change).otherwise(0)
        losses = pl.when(price_change < 0).then(-price_change).otherwise(0)
        
        # 지수이동평균 계산 (RSI 표준 방식)
        avg_gains = gains.ewm_mean(span=period)
        avg_losses = losses.ewm_mean(span=period)
        
        # 0으로 나누기 방지
        rs = avg_gains / pl.when(avg_losses > 0).then(avg_losses).otherwise(1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals_multi_timeframe(
        self, 
        multi_current_data: Dict[str, Dict[str, Any]]
    ) -> List[Order]:
        """멀티 타임프레임 신호 생성"""
        
        orders = []
        
        # 1분봉과 5분봉 데이터 확인
        data_1m = multi_current_data.get("1m")
        data_5m = multi_current_data.get("5m")
        
        if not data_1m or not data_5m:
            return orders
        
        symbol = data_1m.get('symbol')
        if not symbol:
            return orders
        
        # 멀티 타임프레임 분석
        signal = self._analyze_multi_timeframe_signal(data_1m, data_5m, symbol)
        
        if signal == "BUY":
            orders.append(self._create_buy_order(symbol, data_1m))
        elif signal == "SELL":
            orders.append(self._create_sell_order(symbol))
        
        return orders
    
    def _analyze_multi_timeframe_signal(
        self, 
        data_1m: Dict[str, Any], 
        data_5m: Dict[str, Any], 
        symbol: str
    ) -> str:
        """크로스 타임프레임 신호 분석"""
        
        # 1분봉 조건
        price_1m = data_1m.get('close', 0)
        sma_10_1m = data_1m.get('sma_10', 0)
        sma_20_1m = data_1m.get('sma_20', 0)
        volume_ratio_1m = data_1m.get('volume_ratio', 1)
        
        # 5분봉 조건  
        sma_5_5m = data_5m.get('sma_5', 0)
        sma_15_5m = data_5m.get('sma_15', 0)
        rsi_5m = data_5m.get('rsi_14', 50)
        
        # 유효성 검증
        indicators_valid = all([
            sma_10_1m is not None and sma_10_1m > 0,
            sma_20_1m is not None and sma_20_1m > 0,
            sma_5_5m is not None and sma_5_5m > 0,
            sma_15_5m is not None and sma_15_5m > 0,
            rsi_5m is not None
        ])
        
        if not indicators_valid:
            return "HOLD"
        
        current_positions = self.get_current_positions()
        
        # 매수 조건: 1분봉 + 5분봉 모든 조건 만족
        buy_conditions = [
            price_1m > sma_10_1m,           # 1분봉 단기 상승
            sma_10_1m > sma_20_1m,          # 1분봉 골든크로스
            volume_ratio_1m > 1.2,          # 거래량 증가
            sma_5_5m > sma_15_5m,           # 5분봉 상승 추세
            30 < rsi_5m < 70,               # RSI 적정 구간
            symbol not in current_positions
        ]
        
        # 매도 조건: 포지션이 있어야 하고, 다른 조건 중 하나라도 만족해야 함
        has_position = symbol in current_positions and current_positions[symbol] > 0
        
        if has_position:
            sell_technical_conditions = [
                price_1m < sma_20_1m,           # 1분봉 지지선 이탈
                sma_5_5m < sma_15_5m,           # 5분봉 하락 추세
                rsi_5m > 75 or rsi_5m < 25,     # RSI 극값
            ]
            
            if any(sell_technical_conditions):
                return "SELL"
        
        if all(buy_conditions):
            return "BUY"
        else:
            return "HOLD"
    
    def _create_buy_order(self, symbol: str, current_data: Dict[str, Any]) -> Order:
        """매수 주문 생성"""
        current_price = current_data['close']
        portfolio_value = self.get_portfolio_value()
        quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
        
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
    
    def _create_sell_order(self, symbol: str) -> Order:
        """매도 주문 생성"""
        current_positions = self.get_current_positions()
        quantity = current_positions.get(symbol, 0)
        
        # 수량이 0이면 예외 발생
        if quantity <= 0:
            raise ValueError(f"매도할 수량이 없습니다. 심볼: {symbol}, 현재 포지션: {quantity}")
        
        return Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET
        ) 