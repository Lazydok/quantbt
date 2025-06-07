"""
멀티타임프레임 전략 예제

다양한 타임프레임을 활용한 고급 트레이딩 전략들을 제공합니다.
"""

from typing import List, Dict, Any
import polars as pl

from ..core.interfaces.strategy import MultiTimeframeTradingStrategy
from ..core.entities.market_data import MultiTimeframeDataBatch
from ..core.entities.order import Order, OrderSide, OrderType


class MultiTimeframeSMAStrategy(MultiTimeframeTradingStrategy):
    """멀티타임프레임 이동평균 교차 전략
    
    1시간봉에서 전체 트렌드를 파악하고, 5분봉에서 진입 타이밍을 잡는 전략
    - 1시간봉: SMA 20, 50으로 전체 트렌드 방향 판단
    - 5분봉: SMA 5, 20으로 진입/청산 신호 생성
    """
    
    def __init__(
        self,
        name: str = "MultiTimeframeSMAStrategy",
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
        position_size_pct: float = 0.2,
        max_positions: int = 3
    ):
        """멀티타임프레임 SMA 전략 초기화
        
        Args:
            name: 전략명
            timeframes: 사용할 타임프레임 (기본값: ["5m", "1h"])
            config: 설정 딕셔너리
            position_size_pct: 포지션 크기 비율 (기본값: 20%)
            max_positions: 최대 포지션 수 (기본값: 3개)
        """
        if timeframes is None:
            timeframes = ["5m", "1h"]
        
        if config is None:
            config = {}
        
        # 기본 설정값들
        default_config = {
            "hourly_short_period": 20,    # 1시간봉 단기 SMA
            "hourly_long_period": 50,     # 1시간봉 장기 SMA
            "signal_short_period": 5,     # 5분봉 단기 SMA
            "signal_long_period": 20,     # 5분봉 장기 SMA
            "trend_strength_threshold": 0.02,  # 트렌드 강도 임계값 (2%)
            "entry_confirmation_bars": 2   # 진입 확인 봉 수
        }
        default_config.update(config)
        
        super().__init__(
            name=name,
            timeframes=timeframes,
            config=default_config,
            position_size_pct=position_size_pct,
            max_positions=max_positions,
            primary_timeframe="5m"
        )
        
        # 설정값 추출
        self.hourly_short = self.config["hourly_short_period"]
        self.hourly_long = self.config["hourly_long_period"] 
        self.signal_short = self.config["signal_short_period"]
        self.signal_long = self.config["signal_long_period"]
        self.trend_threshold = self.config["trend_strength_threshold"]
        self.confirmation_bars = self.config["entry_confirmation_bars"]
    
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 호환성을 위한 구현 (주 타임프레임 사용)"""
        # 주 타임프레임(5분봉) 지표만 계산하여 반환
        return self._compute_indicators_for_symbol_and_timeframe(symbol_data, self.primary_timeframe)
    
    def generate_signals(self, data) -> List[Order]:
        """단일 타임프레임 호환성을 위한 구현"""
        # MultiTimeframeDataBatch가 아닌 경우 단순 전략으로 대체
        if not isinstance(data, MultiTimeframeDataBatch):
            # 기본 SMA 교차 전략으로 작동
            orders = []
            
            for symbol in data.symbols:
                current_price_data = data.get_latest(symbol)
                if not current_price_data:
                    continue
                
                latest_data = data.get_latest_with_indicators(symbol)
                if not latest_data:
                    continue
                
                sma_fast = latest_data.get("sma_fast")
                sma_slow = latest_data.get("sma_slow")
                
                if sma_fast and sma_slow:
                    current_positions = self.get_current_positions()
                    
                    # 매수 신호
                    if (sma_fast > sma_slow and 
                        symbol not in current_positions and 
                        len(current_positions) < self.max_positions):
                        
                        # current_price_data는 MarketData 객체이므로 close 속성으로 접근
                        current_price = current_price_data.close
                        quantity = self.calculate_position_size(
                            symbol, current_price, self.get_portfolio_value()
                        )
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                    
                    # 매도 신호
                    elif sma_fast < sma_slow and symbol in current_positions:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_positions[symbol],
                            order_type=OrderType.MARKET
                        ))
            
            return orders
        else:
            # 멀티타임프레임 신호 생성
            return self.generate_signals_multi_timeframe(data)
    
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str
    ) -> pl.DataFrame:
        """심볼별, 타임프레임별 지표 계산"""
        data = symbol_data.sort("timestamp")
        
        if timeframe == "1h":
            # 1시간봉: 트렌드 분석용 지표
            sma_short = self.calculate_sma(data["close"], self.hourly_short)
            sma_long = self.calculate_sma(data["close"], self.hourly_long)
            
            # 트렌드 강도 계산 (SMA 간격의 비율)
            trend_strength = (sma_short - sma_long) / sma_long
            
            return data.with_columns([
                sma_short.alias("sma_short"),
                sma_long.alias("sma_long"),
                trend_strength.alias("trend_strength")
            ])
            
        elif timeframe == "5m":
            # 5분봉: 진입/청산 신호용 지표
            sma_fast = self.calculate_sma(data["close"], self.signal_short)
            sma_slow = self.calculate_sma(data["close"], self.signal_long)
            
            # RSI 추가 (과매수/과매도 확인용)
            rsi = self.calculate_rsi(data["close"], 14)
            
            return data.with_columns([
                sma_fast.alias("sma_fast"),
                sma_slow.alias("sma_slow"),
                rsi.alias("rsi")
            ])
        
        else:
            # 기본 구현
            return super()._compute_indicators_for_symbol_and_timeframe(symbol_data, timeframe)
    
    def generate_signals_multi_timeframe(
        self, 
        multi_data: MultiTimeframeDataBatch
    ) -> List[Order]:
        """멀티타임프레임 분석 기반 신호 생성"""
        orders = []
        
        if not self._has_required_timeframes(multi_data):
            return orders
        
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_current_positions()
        
        for symbol in multi_data.symbols:
            # 1. 1시간봉에서 전체 트렌드 분석
            trend_direction = self._analyze_hourly_trend(multi_data, symbol)
            
            if trend_direction == "neutral":
                continue  # 명확한 트렌드가 없으면 거래하지 않음
            
            # 2. 5분봉에서 진입/청산 신호 분석
            signal_type = self._analyze_signal_timeframe(multi_data, symbol, trend_direction)
            
            # 3. 주문 생성
            if signal_type == "buy" and len(current_positions) < self.max_positions:
                if symbol not in current_positions:
                    current_price = multi_data.get_timeframe_price("5m", symbol, "close")
                    if current_price:
                        quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
                        
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                        

            
            elif signal_type == "sell" and symbol in current_positions:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                ))
                

        
        return orders
    
    def _has_required_timeframes(self, multi_data: MultiTimeframeDataBatch) -> bool:
        """필요한 타임프레임이 모두 있는지 확인"""
        required = {"5m", "1h"}
        available = set(multi_data.available_timeframes)
        return required.issubset(available)
    
    def _analyze_hourly_trend(self, multi_data: MultiTimeframeDataBatch, symbol: str) -> str:
        """1시간봉에서 전체 트렌드 분석
        
        Returns:
            "bullish": 상승 트렌드
            "bearish": 하락 트렌드  
            "neutral": 중립
        """
        # 1시간봉 데이터에서 트렌드 지표들 조회
        sma_short = multi_data.get_timeframe_indicator("1h", "sma_short", symbol)
        sma_long = multi_data.get_timeframe_indicator("1h", "sma_long", symbol)
        trend_strength = multi_data.get_timeframe_indicator("1h", "trend_strength", symbol)
        
        if not all(v is not None for v in [sma_short, sma_long, trend_strength]):
            return "neutral"
        
        # 트렌드 방향 및 강도 판단
        if sma_short > sma_long and trend_strength > self.trend_threshold:
            return "bullish"
        elif sma_short < sma_long and trend_strength < -self.trend_threshold:
            return "bearish"
        else:
            return "neutral"
    
    def _analyze_signal_timeframe(
        self, 
        multi_data: MultiTimeframeDataBatch, 
        symbol: str, 
        trend_direction: str
    ) -> str:
        """5분봉에서 진입/청산 신호 분석
        
        Args:
            multi_data: 멀티타임프레임 데이터
            symbol: 심볼명
            trend_direction: 1시간봉 트렌드 방향
            
        Returns:
            "buy": 매수 신호
            "sell": 매도 신호
            "hold": 보유
        """
        # 5분봉 지표들 조회
        sma_fast = multi_data.get_timeframe_indicator("5m", "sma_fast", symbol)
        sma_slow = multi_data.get_timeframe_indicator("5m", "sma_slow", symbol)
        rsi = multi_data.get_timeframe_indicator("5m", "rsi", symbol)
        current_price = multi_data.get_timeframe_price("5m", symbol, "close")
        
        if not all(v is not None for v in [sma_fast, sma_slow, rsi, current_price]):
            return "hold"
        
        current_positions = self.get_current_positions()
        is_holding = symbol in current_positions
        
        # 매수 신호 (상승 트렌드 + 골든 크로스 + RSI 과매도 아님)
        if (trend_direction == "bullish" and 
            not is_holding and
            sma_fast > sma_slow and 
            current_price > sma_fast and
            rsi < 80):  # 과매수 상태가 아님
            
            return "buy"
        
        # 매도 신호 (포지션 보유 중 + 청산 조건)
        elif is_holding:
            # 트렌드 반전 또는 데드 크로스 또는 RSI 과매수
            if (trend_direction == "bearish" or 
                sma_fast < sma_slow or 
                current_price < sma_slow or
                rsi > 85):
                
                return "sell"
        
        return "hold"


class MultiTimeframeMACDRSIStrategy(MultiTimeframeTradingStrategy):
    """멀티타임프레임 MACD + RSI 전략
    
    4시간봉에서 MACD로 주요 트렌드를 파악하고,
    15분봉에서 RSI로 진입 타이밍을 조절하는 전략
    """
    
    def __init__(
        self,
        name: str = "MultiTimeframeMACDRSIStrategy",
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
        position_size_pct: float = 0.15,
        max_positions: int = 5
    ):
        if timeframes is None:
            timeframes = ["15m", "4h"]
        
        if config is None:
            config = {}
        
        default_config = {
            "macd_fast": 12,        # MACD 빠른 EMA
            "macd_slow": 26,        # MACD 느린 EMA  
            "macd_signal": 9,       # MACD 신호선
            "rsi_period": 14,       # RSI 기간
            "rsi_oversold": 30,     # RSI 과매도 기준
            "rsi_overbought": 70,   # RSI 과매수 기준
            "volume_threshold": 1.5  # 거래량 임계값 (평균 대비)
        }
        default_config.update(config)
        
        super().__init__(
            name=name,
            timeframes=timeframes,
            config=default_config,
            position_size_pct=position_size_pct,
            max_positions=max_positions,
            primary_timeframe="15m"
        )
    
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 호환성을 위한 구현 (주 타임프레임 사용)"""
        return self._compute_indicators_for_symbol_and_timeframe(symbol_data, self.primary_timeframe)
    
    def generate_signals(self, data) -> List[Order]:
        """단일 타임프레임 호환성을 위한 구현"""
        if not isinstance(data, MultiTimeframeDataBatch):
            # 기본 RSI 전략으로 작동
            orders = []
            current_positions = self.get_current_positions()
            portfolio_value = self.get_portfolio_value()
            
            for symbol in data.symbols:
                current_price_data = data.get_latest(symbol)
                if not current_price_data:
                    continue
                
                latest_data = data.get_latest_with_indicators(symbol)
                if not latest_data:
                    continue
                
                rsi = latest_data.get("rsi")
                if rsi is not None:
                    # 매수 신호 (RSI 과매도)
                    if (rsi < self.config["rsi_oversold"] and 
                        symbol not in current_positions and 
                        len(current_positions) < self.max_positions):
                        
                        # current_price_data는 MarketData 객체이므로 close 속성으로 접근
                        current_price = current_price_data.close
                        quantity = self.calculate_position_size(
                            symbol, current_price, portfolio_value
                        )
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                    
                    # 매도 신호 (RSI 과매수)
                    elif rsi > self.config["rsi_overbought"] and symbol in current_positions:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_positions[symbol],
                            order_type=OrderType.MARKET
                        ))
            
            return orders
        else:
            return self.generate_signals_multi_timeframe(data)
    
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str
    ) -> pl.DataFrame:
        """심볼별, 타임프레임별 지표 계산"""
        data = symbol_data.sort("timestamp")
        
        if timeframe == "4h":
            # 4시간봉: MACD 트렌드 분석
            ema_fast = self.calculate_ema(data["close"], self.config["macd_fast"])
            ema_slow = self.calculate_ema(data["close"], self.config["macd_slow"])
            macd_line = ema_fast - ema_slow
            macd_signal = self.calculate_ema(macd_line, self.config["macd_signal"])
            macd_histogram = macd_line - macd_signal
            
            return data.with_columns([
                macd_line.alias("macd_line"),
                macd_signal.alias("macd_signal"),
                macd_histogram.alias("macd_histogram")
            ])
            
        elif timeframe == "15m":
            # 15분봉: RSI + 거래량 분석
            rsi = self.calculate_rsi(data["close"], self.config["rsi_period"])
            
            # 거래량 이동평균 및 비율
            volume_ma = data["volume"].rolling_mean(20)
            volume_ratio = data["volume"] / volume_ma
            
            return data.with_columns([
                rsi.alias("rsi"),
                volume_ma.alias("volume_ma"),
                volume_ratio.alias("volume_ratio")
            ])
        
        else:
            return super()._compute_indicators_for_symbol_and_timeframe(symbol_data, timeframe)
    
    def generate_signals_multi_timeframe(
        self, 
        multi_data: MultiTimeframeDataBatch
    ) -> List[Order]:
        """멀티타임프레임 MACD + RSI 신호 생성"""
        orders = []
        
        if not {"15m", "4h"}.issubset(set(multi_data.available_timeframes)):
            return orders
        
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_current_positions()
        
        # 모든 심볼에 대한 신호 강도 계산
        signals = []
        
        for symbol in multi_data.symbols:
            signal_strength = self._calculate_signal_strength(multi_data, symbol)
            
            if signal_strength != 0:
                current_price = multi_data.get_timeframe_price("15m", symbol, "close")
                if current_price:
                    signals.append((symbol, signal_strength, current_price))
        
        # 신호 강도 순으로 정렬 (강한 신호부터)
        signals.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 매수 신호 처리
        buy_count = 0
        for symbol, strength, price in signals:
            if strength > 0 and symbol not in current_positions:
                if len(current_positions) + buy_count < self.max_positions:
                    quantity = self.calculate_position_size(symbol, price, portfolio_value)
                    
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    ))
                    
                    buy_count += 1
        
        # 매도 신호 처리
        for symbol, strength, price in signals:
            if strength < -0.3 and symbol in current_positions:  # 강한 매도 신호만
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                ))
                

        
        return orders
    
    def _calculate_signal_strength(self, multi_data: MultiTimeframeDataBatch, symbol: str) -> float:
        """신호 강도 계산 (양수: 매수, 음수: 매도, 0: 중립)"""
        # 4시간봉 MACD 분석
        macd_line = multi_data.get_timeframe_indicator("4h", "macd_line", symbol)
        macd_signal = multi_data.get_timeframe_indicator("4h", "macd_signal", symbol)
        macd_histogram = multi_data.get_timeframe_indicator("4h", "macd_histogram", symbol)
        
        # 15분봉 RSI 및 거래량 분석
        rsi = multi_data.get_timeframe_indicator("15m", "rsi", symbol)
        volume_ratio = multi_data.get_timeframe_indicator("15m", "volume_ratio", symbol)
        
        if not all(v is not None for v in [macd_line, macd_signal, macd_histogram, rsi, volume_ratio]):
            return 0.0
        
        signal_strength = 0.0
        
        # MACD 신호 (4시간봉)
        if macd_line > macd_signal and macd_histogram > 0:
            signal_strength += 0.4  # 상승 신호
        elif macd_line < macd_signal and macd_histogram < 0:
            signal_strength -= 0.4  # 하락 신호
        
        # RSI 신호 (15분봉)
        if rsi < self.config["rsi_oversold"]:
            signal_strength += 0.3  # 과매도 (매수 기회)
        elif rsi > self.config["rsi_overbought"]:
            signal_strength -= 0.3  # 과매수 (매도 기회)
        
        # 거래량 확인 (15분봉)
        if volume_ratio > self.config["volume_threshold"]:
            signal_strength *= 1.2  # 거래량 증가 시 신호 강화
        
        return max(-1.0, min(1.0, signal_strength))  # [-1, 1] 범위로 제한


class MultiTimeframeMomentumStrategy(MultiTimeframeTradingStrategy):
    """멀티타임프레임 모멘텀 전략
    
    일간 차트에서 전체적인 모멘텀을 파악하고,
    1시간봉에서 단기 모멘텀을 확인하여
    15분봉에서 정확한 진입점을 찾는 전략
    """
    
    def __init__(
        self,
        name: str = "MultiTimeframeMomentumStrategy", 
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
        position_size_pct: float = 0.25,
        max_positions: int = 4
    ):
        if timeframes is None:
            timeframes = ["15m", "1h", "1d"]
        
        if config is None:
            config = {}
        
        default_config = {
            "momentum_period": 14,      # 모멘텀 계산 기간
            "roc_threshold": 5.0,       # ROC 임계값 (%)
            "volume_period": 20,        # 거래량 이동평균 기간
            "breakout_period": 20,      # 돌파 기준 기간
            "atr_period": 14,           # ATR 기간
            "risk_multiplier": 2.0      # 리스크 배수
        }
        default_config.update(config)
        
        super().__init__(
            name=name,
            timeframes=timeframes,
            config=default_config,
            position_size_pct=position_size_pct,
            max_positions=max_positions,
            primary_timeframe="15m"
        )
    
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 호환성을 위한 구현 (주 타임프레임 사용)"""
        return self._compute_indicators_for_symbol_and_timeframe(symbol_data, self.primary_timeframe)
    
    def generate_signals(self, data) -> List[Order]:
        """단일 타임프레임 호환성을 위한 구현"""
        if not isinstance(data, MultiTimeframeDataBatch):
            # 기본 모멘텀 전략으로 작동
            orders = []
            current_positions = self.get_current_positions()
            portfolio_value = self.get_portfolio_value()
            
            for symbol in data.symbols:
                current_price_data = data.get_latest(symbol)
                if not current_price_data:
                    continue
                
                latest_data = data.get_latest_with_indicators(symbol)
                if not latest_data:
                    continue
                
                sma_5 = latest_data.get("sma_5")
                sma_20 = latest_data.get("sma_20")
                volume_ratio = latest_data.get("volume_ratio")
                
                if all(v is not None for v in [sma_5, sma_20, volume_ratio]):
                    # 매수 신호 (모멘텀 상승 + 거래량 증가)
                    if (sma_5 > sma_20 and volume_ratio > 1.5 and 
                        symbol not in current_positions and 
                        len(current_positions) < self.max_positions):
                        
                        # current_price_data는 MarketData 객체이므로 close 속성으로 접근
                        current_price = current_price_data.close
                        quantity = self.calculate_position_size(
                            symbol, current_price, portfolio_value
                        )
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                    
                    # 매도 신호 (모멘텀 하락)
                    elif sma_5 < sma_20 and symbol in current_positions:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_positions[symbol],
                            order_type=OrderType.MARKET
                        ))
            
            return orders
        else:
            return self.generate_signals_multi_timeframe(data)
    
    def _compute_indicators_for_symbol_and_timeframe(
        self, 
        symbol_data: pl.DataFrame, 
        timeframe: str
    ) -> pl.DataFrame:
        """심볼별, 타임프레임별 지표 계산"""
        data = symbol_data.sort("timestamp")
        
        if timeframe == "1d":
            # 일간: 장기 모멘텀 및 트렌드
            momentum = self._calculate_momentum(data["close"], self.config["momentum_period"])
            roc = self._calculate_roc(data["close"], self.config["momentum_period"])
            
            return data.with_columns([
                momentum.alias("momentum"),
                roc.alias("roc")
            ])
            
        elif timeframe == "1h":
            # 1시간: 중기 모멘텀 및 변동성
            rsi = self.calculate_rsi(data["close"], 14)
            atr = self._calculate_atr(data, self.config["atr_period"])
            
            # 볼린저 밴드 
            sma_20 = self.calculate_sma(data["close"], 20)
            std_20 = data["close"].rolling_std(20)
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            bb_width = (bb_upper - bb_lower) / sma_20
            
            return data.with_columns([
                rsi.alias("rsi"),
                atr.alias("atr"),
                bb_upper.alias("bb_upper"),
                bb_lower.alias("bb_lower"),
                bb_width.alias("bb_width")
            ])
            
        elif timeframe == "15m":
            # 15분: 단기 진입 신호
            sma_5 = self.calculate_sma(data["close"], 5)
            sma_20 = self.calculate_sma(data["close"], 20)
            
            # 거래량 분석
            volume_sma = self.calculate_sma(data["volume"], self.config["volume_period"])
            volume_ratio = data["volume"] / volume_sma
            
            return data.with_columns([
                sma_5.alias("sma_5"),
                sma_20.alias("sma_20"),
                volume_sma.alias("volume_sma"),
                volume_ratio.alias("volume_ratio")
            ])
        
        else:
            return super()._compute_indicators_for_symbol_and_timeframe(symbol_data, timeframe)
    
    def _calculate_momentum(self, prices: pl.Series, period: int) -> pl.Series:
        """모멘텀 계산 (현재가 - N기간전가)"""
        return prices - prices.shift(period)
    
    def _calculate_roc(self, prices: pl.Series, period: int) -> pl.Series:
        """변화율(ROC) 계산"""
        prev_prices = prices.shift(period)
        return ((prices - prev_prices) / prev_prices * 100).fill_null(0)
    
    def _calculate_atr(self, data: pl.DataFrame, period: int) -> pl.Series:
        """Average True Range 계산"""
        # 기본적인 ATR 계산 - 간단한 구현
        # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
        high_low = data["high"] - data["low"]
        
        # 간단한 구현을 위해 high-low만 사용 (실제 ATR의 근사값)
        # 실제 환경에서는 더 정확한 구현이 필요할 수 있음
        return high_low.rolling_mean(period)
    
    def generate_signals_multi_timeframe(
        self, 
        multi_data: MultiTimeframeDataBatch
    ) -> List[Order]:
        """멀티타임프레임 모멘텀 신호 생성"""
        orders = []
        
        required_timeframes = {"15m", "1h", "1d"}
        if not required_timeframes.issubset(set(multi_data.available_timeframes)):
            return orders
        
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_current_positions()
        
        for symbol in multi_data.symbols:
            signal = self._analyze_momentum_signal(multi_data, symbol)
            
            if signal == "strong_buy" and symbol not in current_positions:
                if len(current_positions) < self.max_positions:
                    current_price = multi_data.get_timeframe_price("15m", symbol, "close")
                    if current_price:
                        quantity = self.calculate_position_size(symbol, current_price, portfolio_value)
                        
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        ))
                        

            
            elif signal in ["sell", "stop_loss"] and symbol in current_positions:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_positions[symbol],
                    order_type=OrderType.MARKET
                ))
                

        
        return orders
    
    def _analyze_momentum_signal(self, multi_data: MultiTimeframeDataBatch, symbol: str) -> str:
        """모멘텀 신호 분석"""
        # 일간 모멘텀 확인
        daily_momentum = multi_data.get_timeframe_indicator("1d", "momentum", symbol)
        daily_roc = multi_data.get_timeframe_indicator("1d", "roc", symbol)
        
        # 1시간 상태 확인
        hourly_rsi = multi_data.get_timeframe_indicator("1h", "rsi", symbol)
        hourly_bb_width = multi_data.get_timeframe_indicator("1h", "bb_width", symbol)
        
        # 15분 진입 조건 확인
        sma_5 = multi_data.get_timeframe_indicator("15m", "sma_5", symbol)
        sma_20 = multi_data.get_timeframe_indicator("15m", "sma_20", symbol)
        volume_ratio = multi_data.get_timeframe_indicator("15m", "volume_ratio", symbol)
        current_price = multi_data.get_timeframe_price("15m", symbol, "close")
        
        if not all(v is not None for v in [
            daily_momentum, daily_roc, hourly_rsi, hourly_bb_width,
            sma_5, sma_20, volume_ratio, current_price
        ]):
            return "hold"
        
        current_positions = self.get_current_positions()
        is_holding = symbol in current_positions
        
        # 강한 매수 신호: 모든 타임프레임에서 긍정적
        if (not is_holding and
            daily_momentum > 0 and daily_roc > self.config["roc_threshold"] and  # 일간 강한 상승 모멘텀
            hourly_rsi < 70 and hourly_bb_width > 0.02 and  # 1시간 과매수 아님 + 변동성 충분
            sma_5 > sma_20 and current_price > sma_5 and  # 15분 상승 추세
            volume_ratio > 1.5):  # 거래량 증가
            
            return "strong_buy"
        
        # 매도 신호
        if is_holding:
            if (daily_momentum < 0 or daily_roc < -self.config["roc_threshold"] or  # 일간 모멘텀 반전
                hourly_rsi > 80 or  # 1시간 과매수
                sma_5 < sma_20 or current_price < sma_20):  # 15분 추세 반전
                
                return "sell"
        
        return "hold" 