#!/usr/bin/env python3
"""
QuantBT 고급 멀티타임프레임 백테스팅 예제

이 예제는 다음을 보여줍니다:
1. 4개 타임프레임 동시 분석 (1m, 15m, 1h, 4h)
2. 다중 지표 조합 (MACD, RSI, 볼린저밴드)
3. 고급 리스크 관리 (포지션 크기 조정, 손절매)
4. 동적 타이밍 최적화

전략 로직:
- 4시간봉: MACD 기반 주요 트렌드 
- 1시간봉: 볼린저밴드 기반 변동성 필터
- 15분봉: RSI + SMA 기반 진입 신호
- 1분봉: 정밀한 타이밍 조정
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import polars as pl
import numpy as np

# QuantBT 모듈 추가 (quantbt/examples에서 실행되는 경우)
sys.path.append(str(Path(__file__).parent.parent.parent))

from quantbt.core.interfaces.strategy import MultiTimeframeTradingStrategy
from quantbt.core.entities.order import Order, OrderSide, OrderType
from quantbt.infrastructure.engine.simple_engine import SimpleBacktestEngine
from quantbt.infrastructure.data.csv_provider import CSVDataProvider
from quantbt.infrastructure.brokers.simple_broker import SimpleBroker
from quantbt.core.value_objects.backtest_config import BacktestConfig


class AdvancedMultiTimeframeStrategy(MultiTimeframeTradingStrategy):
    """고급 멀티타임프레임 전략
    
    4H: MACD 기반 주요 트렌드 분석
    1H: 볼린저밴드 기반 변동성 및 과매매 분석
    15M: RSI + SMA 기반 진입 신호 
    1M: 정밀한 타이밍 및 리스크 관리
    """
    
    def __init__(self):
        super().__init__(
            name="AdvancedMultiTimeframe",
            timeframes=["1m", "15m", "1h", "4h"],  # 4개 타임프레임
            position_size_pct=0.6,  # 보수적인 60% 포지션
            max_positions=3         # 최대 3개 동시 포지션
        )
        
        # MACD 파라미터 (4시간봉)
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 볼린저밴드 파라미터 (1시간봉)
        self.bb_period = 20
        self.bb_std = 2.0
        
        # RSI 파라미터 (15분봉)
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # SMA 파라미터 (15분봉)
        self.signal_short_sma = 10
        self.signal_long_sma = 21
        
        # 리스크 관리
        self.stop_loss_pct = 0.02      # 2% 손절매
        self.take_profit_pct = 0.06    # 6% 익절매
        self.trailing_stop_pct = 0.015 # 1.5% 트레일링
        
        print(f"🎯 {self.name} 전략 초기화 완료")
        print(f"   • 타임프레임: {self.timeframes}")
        print(f"   • MACD: {self.macd_fast}/{self.macd_slow}/{self.macd_signal}")
        print(f"   • 볼린저밴드: {self.bb_period}기간, {self.bb_std}σ")
        print(f"   • RSI: {self.rsi_period}기간 ({self.rsi_oversold}/{self.rsi_overbought})")
        print(f"   • 리스크 관리: 손절 {self.stop_loss_pct*100}%, 익절 {self.take_profit_pct*100}%")
    
    def precompute_indicators_multi_timeframe(self, data_dict):
        """각 타임프레임별 고급 지표 계산"""
        print("\n📊 고급 멀티타임프레임 지표 계산 중...")
        
        result = {}
        
        for timeframe, df in data_dict.items():
            print(f"   • {timeframe}: {len(df)} 레코드 처리 중...")
            
            # 심볼별로 그룹화하여 지표 계산
            enriched_data = df.sort(["symbol", "timestamp"]).group_by("symbol").map_groups(
                lambda group: self._compute_indicators_for_timeframe(group, timeframe)
            )
            
            result[timeframe] = enriched_data
            
        print("✅ 모든 고급 지표 계산 완료")
        return result
    
    def _compute_indicators_for_timeframe(self, symbol_data, timeframe):
        """타임프레임별 특화 지표 계산"""
        
        if timeframe == "4h":
            # 4시간봉: MACD + 장기 SMA
            return symbol_data.with_columns([
                self._calculate_macd(pl.col("close")).alias("macd"),
                self._calculate_macd_signal(pl.col("close")).alias("macd_signal"),
                self._calculate_macd_histogram(pl.col("close")).alias("macd_hist"),
                pl.col("close").rolling_mean(50).alias("sma_50"),
                pl.col("close").rolling_mean(100).alias("sma_100"),
                self.calculate_rsi(pl.col("close"), 14).alias("rsi")
            ])
            
        elif timeframe == "1h":
            # 1시간봉: 볼린저밴드 + 변동성 지표
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(self.bb_period).alias("bb_middle"),
                self._calculate_bb_upper(pl.col("close")).alias("bb_upper"),
                self._calculate_bb_lower(pl.col("close")).alias("bb_lower"),
                self._calculate_bb_width(pl.col("close")).alias("bb_width"),
                pl.col("close").rolling_mean(20).alias("sma_20"),
                self.calculate_rsi(pl.col("close"), 14).alias("rsi"),
                self._calculate_volatility(pl.col("close")).alias("volatility")
            ])
            
        elif timeframe == "15m":
            # 15분봉: RSI + SMA 교차 + 모멘텀
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(self.signal_short_sma).alias("sma_10"),
                pl.col("close").rolling_mean(self.signal_long_sma).alias("sma_21"),
                self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi"),
                self._calculate_momentum(pl.col("close"), 10).alias("momentum"),
                self._calculate_rate_of_change(pl.col("close"), 5).alias("roc")
            ])
            
        else:  # 1분봉
            # 1분봉: 단기 지표 + 노이즈 필터
            return symbol_data.with_columns([
                pl.col("close").rolling_mean(5).alias("sma_5"),
                pl.col("close").rolling_mean(10).alias("sma_10"),
                self.calculate_rsi(pl.col("close"), 10).alias("rsi_fast"),
                self._calculate_price_velocity(pl.col("close")).alias("velocity")
            ])
    
    # 고급 지표 계산 메서드들
    def _calculate_macd(self, close_col):
        """MACD 라인 계산"""
        ema_fast = close_col.ewm_mean(span=self.macd_fast)
        ema_slow = close_col.ewm_mean(span=self.macd_slow)
        return ema_fast - ema_slow
    
    def _calculate_macd_signal(self, close_col):
        """MACD 시그널 라인 계산"""
        macd = self._calculate_macd(close_col)
        return macd.ewm_mean(span=self.macd_signal)
    
    def _calculate_macd_histogram(self, close_col):
        """MACD 히스토그램 계산"""
        macd = self._calculate_macd(close_col)
        signal = self._calculate_macd_signal(close_col)
        return macd - signal
    
    def _calculate_bb_upper(self, close_col):
        """볼린저밴드 상단"""
        sma = close_col.rolling_mean(self.bb_period)
        std = close_col.rolling_std(self.bb_period)
        return sma + (std * self.bb_std)
    
    def _calculate_bb_lower(self, close_col):
        """볼린저밴드 하단"""
        sma = close_col.rolling_mean(self.bb_period)
        std = close_col.rolling_std(self.bb_period)
        return sma - (std * self.bb_std)
    
    def _calculate_bb_width(self, close_col):
        """볼린저밴드 폭 (변동성 지표)"""
        upper = self._calculate_bb_upper(close_col)
        lower = self._calculate_bb_lower(close_col)
        middle = close_col.rolling_mean(self.bb_period)
        return (upper - lower) / middle
    
    def _calculate_volatility(self, close_col):
        """변동성 계산 (20기간 표준편차)"""
        return close_col.rolling_std(20) / close_col.rolling_mean(20)
    
    def _calculate_momentum(self, close_col, period):
        """모멘텀 계산"""
        return close_col / close_col.shift(period) - 1
    
    def _calculate_rate_of_change(self, close_col, period):
        """변화율 계산"""
        return (close_col - close_col.shift(period)) / close_col.shift(period) * 100
    
    def _calculate_price_velocity(self, close_col):
        """가격 변화 속도 (3기간 평균 변화율)"""
        change = close_col.pct_change()
        return change.rolling_mean(3)
    
    def generate_signals_multi_timeframe(self, multi_data):
        """고급 멀티타임프레임 신호 생성"""
        orders = []
        
        for symbol in multi_data.symbols:
            # 각 타임프레임별 신호 분석
            trend_signal = self._analyze_4h_trend(multi_data, symbol)          # 주요 트렌드
            volatility_signal = self._analyze_1h_volatility(multi_data, symbol) # 변동성 필터
            entry_signal = self._analyze_15m_entry(multi_data, symbol)         # 진입 신호
            timing_signal = self._analyze_1m_timing(multi_data, symbol)        # 타이밍 조정
            
            # 리스크 관리 신호
            risk_signal = self._check_risk_management(multi_data, symbol)
            
            # 현재 포지션 상태
            current_positions = self.get_current_positions()
            position_count = len(current_positions)
            
            # 종합 신호 점수 계산
            signal_score = self._calculate_signal_score(
                trend_signal, volatility_signal, entry_signal, timing_signal
            )
            
            print(f"🔍 {symbol} 신호 분석:")
            print(f"   • 4H 트렌드: {trend_signal}")
            print(f"   • 1H 변동성: {volatility_signal}")  
            print(f"   • 15M 진입: {entry_signal}")
            print(f"   • 1M 타이밍: {timing_signal}")
            print(f"   • 종합 점수: {signal_score:.2f}")
            
            # 매수 조건: 종합 점수 > 0.7
            if (signal_score > 0.7 and 
                symbol not in current_positions and
                position_count < self.max_positions and
                risk_signal == "safe"):
                
                current_price = multi_data.get_timeframe_price("1m", symbol, "close")
                if current_price:
                    # 동적 포지션 크기 조정 (신호 강도에 따라)
                    base_quantity = self.calculate_position_size(
                        symbol, current_price, self.get_portfolio_value()
                    )
                    adjusted_quantity = base_quantity * min(signal_score, 1.0)
                    
                    if adjusted_quantity > 0:
                        orders.append(Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=adjusted_quantity,
                            order_type=OrderType.MARKET,
                            metadata={
                                "signal_score": signal_score,
                                "trend_signal": trend_signal,
                                "volatility_signal": volatility_signal,
                                "strategy": "advanced_multi_timeframe"
                            }
                        ))
                        
        
            
            # 매도 조건: 리스크 또는 신호 약화
            elif symbol in current_positions:
                exit_signal = self._check_exit_conditions(
                    multi_data, symbol, signal_score, risk_signal
                )
                
                if exit_signal != "hold":
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=current_positions[symbol],
                        order_type=OrderType.MARKET,
                        metadata={
                            "exit_reason": exit_signal,
                            "signal_score": signal_score
                        }
                    ))
                    
    
        
        return orders
    
    def _analyze_4h_trend(self, multi_data, symbol):
        """4시간봉 MACD 기반 주요 트렌드 분석"""
        macd = multi_data.get_timeframe_indicator("4h", "macd", symbol)
        macd_signal = multi_data.get_timeframe_indicator("4h", "macd_signal", symbol)
        macd_hist = multi_data.get_timeframe_indicator("4h", "macd_hist", symbol)
        sma_50 = multi_data.get_timeframe_indicator("4h", "sma_50", symbol)
        sma_100 = multi_data.get_timeframe_indicator("4h", "sma_100", symbol)
        current_price = multi_data.get_timeframe_price("4h", symbol, "close")
        
        if not all([macd, macd_signal, macd_hist, sma_50, sma_100, current_price]):
            return "neutral"
        
        score = 0
        
        # MACD 신호
        if macd > macd_signal and macd_hist > 0:
            score += 3  # 강한 상승
        elif macd > macd_signal:
            score += 2  # 상승
        elif macd < macd_signal and macd_hist < 0:
            score -= 3  # 강한 하락
        elif macd < macd_signal:
            score -= 2  # 하락
        
        # SMA 트렌드
        if current_price > sma_50 > sma_100:
            score += 2  # 정배열
        elif current_price < sma_50 < sma_100:
            score -= 2  # 역배열
        
        # 점수별 트렌드 분류
        if score >= 4:
            return "very_bullish"
        elif score >= 2:
            return "bullish"
        elif score <= -4:
            return "very_bearish"
        elif score <= -2:
            return "bearish"
        else:
            return "neutral"
    
    def _analyze_1h_volatility(self, multi_data, symbol):
        """1시간봉 볼린저밴드 기반 변동성 분석"""
        current_price = multi_data.get_timeframe_price("1h", symbol, "close")
        bb_upper = multi_data.get_timeframe_indicator("1h", "bb_upper", symbol)
        bb_lower = multi_data.get_timeframe_indicator("1h", "bb_lower", symbol)
        bb_middle = multi_data.get_timeframe_indicator("1h", "bb_middle", symbol)
        bb_width = multi_data.get_timeframe_indicator("1h", "bb_width", symbol)
        
        if not all([current_price, bb_upper, bb_lower, bb_middle, bb_width]):
            return "neutral"
        
        # 볼린저밴드 위치 분석
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # 변동성 상태 분석
        if bb_width < 0.05:  # 저변동성 (밴드 수축)
            if 0.3 <= bb_position <= 0.7:
                return "low_vol_opportunity"  # 좋은 진입 기회
            else:
                return "low_vol_wait"
        elif bb_width > 0.15:  # 고변동성 (밴드 확장)
            if bb_position > 0.8:
                return "high_vol_overbought"  # 과매수
            elif bb_position < 0.2:
                return "high_vol_oversold"   # 과매도
            else:
                return "high_vol_normal"
        else:  # 보통 변동성
            if bb_position > 0.7:
                return "normal_vol_upper"
            elif bb_position < 0.3:
                return "normal_vol_lower"
            else:
                return "normal_vol_middle"
    
    def _analyze_15m_entry(self, multi_data, symbol):
        """15분봉 RSI + SMA 기반 진입 신호"""
        current_price = multi_data.get_timeframe_price("15m", symbol, "close")
        sma_10 = multi_data.get_timeframe_indicator("15m", "sma_10", symbol)
        sma_21 = multi_data.get_timeframe_indicator("15m", "sma_21", symbol)
        rsi = multi_data.get_timeframe_indicator("15m", "rsi", symbol)
        momentum = multi_data.get_timeframe_indicator("15m", "momentum", symbol)
        
        if not all([current_price, sma_10, sma_21, rsi, momentum]):
            return "neutral"
        
        score = 0
        
        # SMA 교차 신호
        if sma_10 > sma_21 and current_price > sma_10:
            score += 2  # 골든크로스
        elif sma_10 < sma_21 and current_price < sma_10:
            score -= 2  # 데드크로스
        
        # RSI 신호
        if 30 < rsi < 50:  # 과매도에서 회복
            score += 2
        elif 50 < rsi < 70:  # 건전한 상승
            score += 1
        elif rsi > 75:  # 과매수
            score -= 2
        elif rsi < 25:  # 극도 과매도
            score -= 1
        
        # 모멘텀 신호
        if momentum > 0.02:  # 강한 상승 모멘텀
            score += 1
        elif momentum < -0.02:  # 강한 하락 모멘텀
            score -= 1
        
        # 점수별 신호 분류
        if score >= 4:
            return "strong_buy"
        elif score >= 2:
            return "buy"
        elif score <= -4:
            return "strong_sell"
        elif score <= -2:
            return "sell"
        else:
            return "neutral"
    
    def _analyze_1m_timing(self, multi_data, symbol):
        """1분봉 정밀 타이밍 분석"""
        current_price = multi_data.get_timeframe_price("1m", symbol, "close")
        sma_5 = multi_data.get_timeframe_indicator("1m", "sma_5", symbol)
        sma_10 = multi_data.get_timeframe_indicator("1m", "sma_10", symbol)
        velocity = multi_data.get_timeframe_indicator("1m", "velocity", symbol)
        
        if not all([current_price, sma_5, sma_10, velocity]):
            return "neutral"
        
        # 단기 트렌드 체크
        if current_price > sma_5 > sma_10 and velocity > 0:
            return "good_timing"
        elif current_price < sma_5 < sma_10 and velocity < 0:
            return "bad_timing"
        else:
            return "neutral_timing"
    
    def _check_risk_management(self, multi_data, symbol):
        """리스크 관리 신호 확인"""
        # 1시간봉 변동성으로 리스크 체크
        volatility = multi_data.get_timeframe_indicator("1h", "volatility", symbol)
        
        if volatility and volatility > 0.05:  # 고변동성
            return "high_risk"
        elif volatility and volatility < 0.02:  # 저변동성
            return "low_risk"
        else:
            return "safe"
    
    def _calculate_signal_score(self, trend, volatility, entry, timing):
        """종합 신호 점수 계산 (0-1 범위)"""
        score = 0.0
        
        # 4시간봉 트렌드 (40% 가중치)
        trend_weights = {
            "very_bullish": 1.0, "bullish": 0.7, "neutral": 0.0,
            "bearish": -0.7, "very_bearish": -1.0
        }
        score += trend_weights.get(trend, 0.0) * 0.4
        
        # 1시간봉 변동성 (20% 가중치)
        vol_weights = {
            "low_vol_opportunity": 0.8, "normal_vol_middle": 0.5,
            "high_vol_oversold": 0.3, "high_vol_normal": 0.0,
            "high_vol_overbought": -0.5
        }
        score += vol_weights.get(volatility, 0.0) * 0.2
        
        # 15분봉 진입 (30% 가중치)
        entry_weights = {
            "strong_buy": 1.0, "buy": 0.7, "neutral": 0.0,
            "sell": -0.7, "strong_sell": -1.0
        }
        score += entry_weights.get(entry, 0.0) * 0.3
        
        # 1분봉 타이밍 (10% 가중치)
        timing_weights = {
            "good_timing": 1.0, "neutral_timing": 0.0, "bad_timing": -1.0
        }
        score += timing_weights.get(timing, 0.0) * 0.1
        
        return max(0.0, score)  # 0 이상으로 정규화
    
    def _check_exit_conditions(self, multi_data, symbol, signal_score, risk_signal):
        """종합적 청산 조건 확인"""
        # 1. 신호 약화
        if signal_score < 0.3:
            return "signal_weak"
        
        # 2. 리스크 증가
        if risk_signal == "high_risk":
            return "high_risk"
        
        # 3. 트렌드 전환
        trend = self._analyze_4h_trend(multi_data, symbol)
        if trend in ["bearish", "very_bearish"]:
            return "trend_reversal"
        
        # 4. 기술적 과매수
        entry_signal = self._analyze_15m_entry(multi_data, symbol)
        if entry_signal in ["sell", "strong_sell"]:
            return "technical_sell"
        
        return "hold"
    
    def _compute_indicators_for_symbol(self, symbol_data: pl.DataFrame) -> pl.DataFrame:
        """단일 타임프레임 호환성을 위한 구현 (주 타임프레임 사용)"""
        # 15분봉 지표만 계산하여 반환
        return symbol_data.with_columns([
            pl.col("close").rolling_mean(self.signal_short_sma).alias("sma_10"),
            pl.col("close").rolling_mean(self.signal_long_sma).alias("sma_21"),
            self.calculate_rsi(pl.col("close"), self.rsi_period).alias("rsi"),
            self._calculate_momentum(pl.col("close"), 10).alias("momentum"),
            self._calculate_rate_of_change(pl.col("close"), 5).alias("roc")
        ])
    
    def generate_signals(self, data) -> List[Order]:
        """단일 타임프레임 호환성을 위한 구현"""
        # MultiTimeframeDataBatch가 아닌 경우 단순 전략으로 대체
        if not hasattr(data, 'timeframes'):
            # 기본 SMA 교차 전략으로 작동
            orders = []
            
            for symbol in data.symbols:
                current_price_data = data.get_latest(symbol)
                if not current_price_data:
                    continue
                
                latest_data = data.get_latest_with_indicators(symbol)
                if not latest_data:
                    continue
                
                sma_10 = latest_data.get("sma_10")
                sma_21 = latest_data.get("sma_21")
                
                if sma_10 and sma_21:
                    current_positions = self.get_current_positions()
                    
                    # 매수 신호
                    if (sma_10 > sma_21 and 
                        symbol not in current_positions and 
                        len(current_positions) < self.max_positions):
                        
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
                    elif sma_10 < sma_21 and symbol in current_positions:
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


async def generate_advanced_sample_data():
    """고급 예제용 샘플 데이터 생성 (더 현실적인 패턴)"""
    print("📊 고급 샘플 데이터 생성 중...")
    
    symbols = ["BTC", "ETH", "SOL"]  # 3개 종목
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 1)  # 5개월 데이터
    
    data_rows = []
    
    for symbol in symbols:
        print(f"   • {symbol} 고급 데이터 생성...")
        
        # 종목별 특성
        if symbol == "BTC":
            base_price = 45000
            volatility = 0.02
        elif symbol == "ETH":
            base_price = 2800
            volatility = 0.025
        else:  # SOL
            base_price = 100
            volatility = 0.03
        
        current_price = base_price
        current_time = start_date
        
        # 트렌드 사이클 (며칠마다 트렌드 변경)
        trend_cycle = 0
        trend_direction = 1
        
        while current_time < end_date:
            import random
            import math
            
            # 트렌드 사이클 관리 (7일마다 변경 가능성)
            if trend_cycle % (7 * 24 * 60) == 0:  # 7일 = 7*24*60분
                if random.random() < 0.3:  # 30% 확률로 트렌드 변경
                    trend_direction *= -1
            
            # 시간대별 변동성 (아시아/유럽/미국 시간)
            hour = current_time.hour
            if 9 <= hour <= 17:  # 아시아 시간 (높은 변동성)
                time_volatility = volatility * 1.2
            elif 15 <= hour <= 23:  # 유럽+미국 시간 (매우 높은 변동성)
                time_volatility = volatility * 1.5
            else:  # 낮은 변동성 시간
                time_volatility = volatility * 0.7
            
            # 기본 트렌드 + 노이즈 + 시간대 효과
            trend_change = trend_direction * 0.0001  # 약한 트렌드
            noise = random.uniform(-time_volatility, time_volatility)
            
            # 주기적 패턴 (일일 사이클)
            daily_cycle = math.sin(2 * math.pi * (current_time.hour + current_time.minute/60) / 24) * 0.0005
            
            total_change = trend_change + noise + daily_cycle
            current_price *= (1 + total_change)
            
            # OHLCV 생성 (더 현실적)
            open_price = current_price
            minute_volatility = time_volatility * 0.1  # 1분내 변동
            
            close_change = random.uniform(-minute_volatility, minute_volatility)
            close_price = current_price * (1 + close_change)
            
            high_price = max(open_price, close_price) * (1 + random.uniform(0, minute_volatility/2))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, minute_volatility/2))
            
            # 볼륨은 변동성과 상관관계
            base_volume = 500
            volatility_volume = abs(close_change) * 10000
            time_volume = 1.5 if 15 <= hour <= 23 else 0.8
            volume = base_volume * time_volume + volatility_volume + random.uniform(50, 200)
            
            data_rows.append({
                "timestamp": current_time,
                "symbol": symbol,
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "volume": round(volume, 2)
            })
            
            current_price = close_price
            current_time += timedelta(minutes=1)
            trend_cycle += 1
    
    print(f"✅ 총 {len(data_rows)} 개 고급 레코드 생성")
    return pl.DataFrame(data_rows)


async def main():
    """고급 멀티타임프레임 백테스팅 실행"""
    print("🚀 QuantBT 고급 멀티타임프레임 백테스팅 시작\n")
    
    try:
        # 1. 고급 샘플 데이터 생성
        sample_data = await generate_advanced_sample_data()
        
        # 2. 임시 CSV 파일로 저장 (프로젝트 루트에)
        data_dir = Path(__file__).parent.parent.parent / "temp_advanced_data"
        data_dir.mkdir(exist_ok=True)
        
        for symbol in sample_data["symbol"].unique():
            symbol_data = sample_data.filter(pl.col("symbol") == symbol)
            csv_path = data_dir / f"{symbol}.csv"
            symbol_data.write_csv(csv_path)
            print(f"💾 {symbol} 고급 데이터 저장: {csv_path}")
        
        # 3. 백테스팅 컴포넌트 설정
        data_provider = CSVDataProvider(str(data_dir))
        broker = SimpleBroker(
            initial_cash=200000,  # 더 큰 초기 자본
            commission_rate=0.001,
            slippage_rate=0.0001
        )
        strategy = AdvancedMultiTimeframeStrategy()
        
        # 4. 백테스팅 설정
        config = BacktestConfig(
            symbols=["BTC", "ETH", "SOL"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            initial_cash=200000,
            timeframe="1m",
            commission_rate=0.001
        )
        
        print(f"\n⚙️ 고급 백테스팅 설정:")
        print(f"   • 종목: {config.symbols}")
        print(f"   • 기간: {config.start_date.date()} ~ {config.end_date.date()}")
        print(f"   • 초기 자본: ${config.initial_cash:,}")
        print(f"   • 타임프레임: 4개 (1m, 15m, 1h, 4h)")
        
        # 5. 백테스팅 실행
        engine = SimpleBacktestEngine()
        engine.set_strategy(strategy)
        engine.set_data_provider(data_provider)
        engine.set_broker(broker)
        
        print(f"\n🔄 고급 멀티타임프레임 백테스팅 실행 중...")
        result = await engine.run(config)
        
        # 6. 상세 결과 출력
        print(f"\n" + "="*70)
        print(f"📊 고급 멀티타임프레임 백테스팅 결과")
        print(f"="*70)
        print(f"총 수익률:        {result.total_return_pct:>10.2f}%")
        print(f"연간 수익률:      {result.annual_return_pct:>10.2f}%")
        print(f"변동성:          {result.volatility_pct:>10.2f}%")
        print(f"샤프 비율:       {result.sharpe_ratio:>10.2f}")
        print(f"소르티노 비율:    {getattr(result, 'sortino_ratio', 0):>10.2f}")
        print(f"최대 낙폭:       {result.max_drawdown_pct:>10.2f}%")
        print(f"칼마 비율:       {getattr(result, 'calmar_ratio', 0):>10.2f}")
        print(f"-" * 70)
        print(f"총 거래 수:       {result.total_trades:>10}")
        print(f"승률:            {result.win_rate_pct:>10.2f}%")
        print(f"평균 수익:       ${getattr(result, 'avg_win', 0):>10.2f}")
        print(f"평균 손실:       ${getattr(result, 'avg_loss', 0):>10.2f}")
        print(f"수익/손실비:      {getattr(result, 'profit_loss_ratio', 0):>10.2f}")
        print(f"평균 거래 기간:   {getattr(result, 'avg_trade_duration', 'N/A'):>10}")
        print(f"="*70)
        
        # 7. 고급 분석
        if hasattr(result, 'monthly_returns'):
            print(f"\n📅 월별 수익률:")
            for month, return_pct in result.monthly_returns.items():
                print(f"   {month}: {return_pct:>8.2f}%")
        
        # 8. 포트폴리오 통계
        if hasattr(result, 'portfolio_values') and result.portfolio_values:
            values = result.portfolio_values
            print(f"\n📈 포트폴리오 통계:")
            print(f"   시작 가치:    ${values[0]:>12,.2f}")
            print(f"   최고 가치:    ${max(values):>12,.2f}")
            print(f"   최저 가치:    ${min(values):>12,.2f}")
            print(f"   최종 가치:    ${values[-1]:>12,.2f}")
            print(f"   최고점 대비:   {((values[-1] - max(values)) / max(values) * 100):>8.2f}%")
        
        # 9. 청소
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"\n🧹 임시 데이터 정리 완료")
        
        print(f"\n✅ 고급 백테스팅 완료!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 비동기 실행
    result = asyncio.run(main())
    
    if result:
        print(f"\n💡 이 고급 예제는 다음을 보여줍니다:")
        print(f"   • 4개 타임프레임 동시 분석 (1m/15m/1h/4h)")
        print(f"   • 다중 지표 조합 (MACD/RSI/볼린저밴드)")
        print(f"   • 동적 신호 점수 계산 및 포지션 크기 조정")
        print(f"   • 고급 리스크 관리 (변동성 기반 필터링)")
        print(f"   • 종합적 진입/청산 조건")
    else:
        print(f"\n🔧 문제가 발생했습니다. 로그를 확인해주세요.") 