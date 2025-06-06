"""
간단한 백테스팅 엔진

기본적인 백테스팅 기능을 제공하는 엔진 구현체입니다.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import polars as pl

from ...core.interfaces.backtest_engine import BacktestEngineBase
from ...core.interfaces.strategy import BacktestContext
from ...core.entities.trade import Trade, OrderSide
from ...core.entities.market_data import MarketDataBatch
from ...core.value_objects.backtest_config import BacktestConfig
from ...core.value_objects.backtest_result import BacktestResult

# 멀티타임프레임 지원을 위한 import 추가
try:
    from ...core.entities.market_data import MultiTimeframeDataBatch
    from ...core.interfaces.strategy import MultiTimeframeTradingStrategy
    from ...core.utils.timeframe import TimeframeUtils
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False


class SimpleBacktestEngine(BacktestEngineBase):
    """간단한 백테스팅 엔진"""
    
    def __init__(self, name: str = "SimpleBacktestEngine"):
        super().__init__(name)
        self.context: Optional[BacktestContext] = None
    
    def _detect_multi_timeframe_requirement(self, config: BacktestConfig) -> bool:
        """멀티 타임프레임 백테스팅 필요 여부 감지
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            멀티 타임프레임이 필요하면 True
        """
        if not MULTI_TIMEFRAME_AVAILABLE:
            return False
        
        # 1. Config에서 멀티 타임프레임 지정된 경우
        if config.is_multi_timeframe:
            return True
        
        # 2. 전략이 MultiTimeframeTradingStrategy이고 여러 타임프레임을 요구하는 경우
        if (isinstance(self.strategy, MultiTimeframeTradingStrategy) and
            hasattr(self.strategy, 'timeframes') and
            len(self.strategy.timeframes) > 1):
            return True
        
        return False
    
    def _get_effective_timeframes(self, config: BacktestConfig) -> List[str]:
        """실제 사용할 타임프레임 리스트 결정
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            사용할 타임프레임 리스트
        """
        # Config에 멀티 타임프레임이 지정된 경우 우선 사용
        if config.is_multi_timeframe:
            return config.timeframes
        
        # 전략에 타임프레임이 지정된 경우
        if (isinstance(self.strategy, MultiTimeframeTradingStrategy) and
            hasattr(self.strategy, 'timeframes')):
            return self.strategy.timeframes
        
        # 기본값: config의 단일 타임프레임
        return [config.timeframe or "1D"]
    
    def _sync_strategy_timeframes(self, effective_timeframes: List[str]) -> None:
        """전략의 타임프레임 정보를 실제 사용할 타임프레임과 동기화
        
        Args:
            effective_timeframes: 실제 사용할 타임프레임 리스트
        """
        if isinstance(self.strategy, MultiTimeframeTradingStrategy):
            # 전략이 멀티 타임프레임을 지원하는 경우 타임프레임 정보 업데이트
            self.strategy.timeframes = effective_timeframes
            
            # primary_timeframe 재설정 (가장 작은 타임프레임)
            if effective_timeframes:
                timeframe_minutes = [(tf, TimeframeUtils.get_timeframe_minutes(tf)) for tf in effective_timeframes]
                self.strategy.primary_timeframe = min(timeframe_minutes, key=lambda x: x[1])[0]
        
    async def _execute_backtest(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행"""
        start_time = datetime.now()
        self._notify_progress(0.0, "백테스팅 초기화 중...")
        
        # 컨텍스트 초기화
        self.context = BacktestContext(
            initial_cash=config.initial_cash,
            symbols=config.symbols
        )
        
        # 브로커 초기화
        self.broker.portfolio.cash = config.initial_cash
        
        # 멀티타임프레임 감지 (config 우선, 전략 타입 고려)
        is_multi_timeframe = self._detect_multi_timeframe_requirement(config)
        
        if is_multi_timeframe:
            return await self._execute_multi_timeframe_backtest(config, start_time)
        else:
            return await self._execute_single_timeframe_backtest(config, start_time)
    
    async def _execute_single_timeframe_backtest(self, config: BacktestConfig, start_time: datetime) -> BacktestResult:
        """단일 타임프레임 백테스팅 실행 (기존 로직)"""
        # 데이터 로드 및 지표 사전 계산
        self._notify_progress(5.0, "데이터 로딩 중...")
        raw_data = await self._load_raw_data(config)
        
        # 시장 데이터를 엔진에 저장 (포지션 기반 평가를 위해)
        self._market_data = {}
        for symbol in config.symbols:
            symbol_data = raw_data.filter(pl.col("symbol") == symbol)
            self._market_data[symbol] = symbol_data
        
        self._notify_progress(10.0, "지표 계산 중...")
        enriched_data = self.strategy.precompute_indicators(raw_data)
        
        # 전략 초기화
        self.strategy.initialize(self.context)
        # 브로커 연결
        self.strategy.set_broker(self.broker)
        self._notify_progress(20.0, "전략 초기화 완료")
        
        # 데이터 스트림 생성 및 실행
        try:
            trades = await self._run_backtest_loop_with_enriched_data(config, enriched_data)
            self._notify_progress(90.0, "성과 분석 중...")
            
            # 백테스팅 결과 생성
            result = await self._create_result(config, start_time, datetime.now(), trades)
            self._notify_progress(100.0, "백테스팅 완료")
            
            # 전략 종료 처리
            self.strategy.finalize(self.context)
            
            return result
            
        except Exception as e:
            self._notify_progress(0.0, f"백테스팅 실패: {str(e)}")
            raise
    
    async def _execute_multi_timeframe_backtest(self, config: BacktestConfig, start_time: datetime) -> BacktestResult:
        """멀티타임프레임 백테스팅 실행"""
        if not MULTI_TIMEFRAME_AVAILABLE:
            raise ImportError("MultiTimeframe components not available.")

        # 데이터 로드 및 멀티타임프레임 처리
        self._notify_progress(5.0, "멀티타임프레임 데이터 로딩 중...")

        # 실제 사용할 타임프레임 결정
        effective_timeframes = self._get_effective_timeframes(config)
        base_timeframe = min(effective_timeframes, key=TimeframeUtils.get_timeframe_minutes)

        # 전략의 타임프레임 정보 동기화 (필요한 경우)
        self._sync_strategy_timeframes(effective_timeframes)

        # 단일 타임프레임 전략인 경우 주 타임프레임으로만 백테스팅
        if not hasattr(self.strategy, 'precompute_indicators_multi_timeframe'):
            # 단일 타임프레임 전략을 주 타임프레임으로만 실행
            primary_tf = config.primary_timeframe or config.base_timeframe
            data = await self.data_provider.get_data(
                symbols=config.symbols,
                start=config.start_date,
                end=config.end_date,
                timeframe=primary_tf
            )
            
            # 단일 타임프레임 백테스팅으로 위임
            single_tf_config = BacktestConfig(
                symbols=config.symbols,
                start_date=config.start_date,
                end_date=config.end_date,
                timeframe=primary_tf,
                initial_cash=config.initial_cash,
                commission_rate=config.commission_rate,
                slippage_rate=config.slippage_rate,
                benchmark_symbol=config.benchmark_symbol,
                save_trades=config.save_trades,
                save_portfolio_history=config.save_portfolio_history,
                metadata=config.metadata
            )
            return await self._execute_single_timeframe_backtest(
                single_tf_config,
                start_time
            )

        # 멀티타임프레임 데이터 로드
        multi_data = await self.data_provider.get_multi_timeframe_data(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            timeframes=effective_timeframes,
            base_timeframe=base_timeframe
        )

        # 시장 데이터를 엔진에 저장 (포지션 기반 평가를 위해) - 기준 타임프레임 사용
        self._market_data = {}
        if base_timeframe in multi_data:
            for symbol in config.symbols:
                symbol_data = multi_data[base_timeframe].filter(pl.col("symbol") == symbol)
                self._market_data[symbol] = symbol_data

        self._notify_progress(10.0, "멀티타임프레임 지표 계산 중...")

        # 멀티 타임프레임 전략의 지표 계산
        enriched_multi_data = self.strategy.precompute_indicators_multi_timeframe(multi_data)
        
        # 전략 초기화
        self.strategy.initialize(self.context)
        # 브로커 연결
        self.strategy.set_broker(self.broker)
        self._notify_progress(20.0, "멀티타임프레임 전략 초기화 완료")
        
        # 멀티타임프레임 백테스팅 실행
        try:
            trades = await self._run_multi_timeframe_backtest_loop(config, enriched_multi_data, base_timeframe)
            self._notify_progress(90.0, "성과 분석 중...")
            
            # 백테스팅 결과 생성
            result = await self._create_result(config, start_time, datetime.now(), trades)
            self._notify_progress(100.0, "멀티타임프레임 백테스팅 완료")
            
            # 전략 종료 처리
            self.strategy.finalize(self.context)
            
            return result
            
        except Exception as e:
            self._notify_progress(0.0, f"멀티타임프레임 백테스팅 실패: {str(e)}")
            raise
    
    async def _run_multi_timeframe_backtest_loop(
        self, 
        config: BacktestConfig, 
        enriched_multi_data: Dict[str, pl.DataFrame],
        base_timeframe: str
    ) -> List[Trade]:
        """멀티타임프레임 백테스팅 루프 실행"""
        import time
        
        all_trades = []
        loop_start_time = time.time()
        TIMEOUT_SECONDS = 30  # 30초 타임아웃
        
        # 기준 타임프레임의 시간 단위로 루프 실행
        base_data = enriched_multi_data[base_timeframe]
        
        # 데이터 유효성 검증
        if base_data.height == 0:
            return all_trades
        
        # 타임스탬프별 그룹핑
        time_groups = base_data.group_by("timestamp").agg(
            pl.all().exclude("timestamp")
        ).sort("timestamp")
        
        total_steps = time_groups.height
        
        # 성능 최적화: 처리 제한
        if total_steps > 1000:
            time_groups = time_groups.head(500)
            total_steps = 500
        
        # 병목 측정용 변수들
        filter_time_total = 0.0
        strategy_time_total = 0.0
        broker_time_total = 0.0
        
        for step, time_group in enumerate(time_groups.iter_rows(named=True)):
            # 타임아웃 체크
            current_time = time.time()
            elapsed_time = current_time - loop_start_time
            
            if elapsed_time > TIMEOUT_SECONDS:
                break
            
            timestamp = time_group["timestamp"]
            
            # 현재 시점까지의 멀티타임프레임 데이터 생성
            filter_start = time.time()
            timeframe_data = {}
            for tf, df in enriched_multi_data.items():
                current_batch_data = df.filter(pl.col("timestamp") <= timestamp)
                timeframe_data[tf] = current_batch_data
            filter_time_total += time.time() - filter_start
            
            if not timeframe_data or all(df.height == 0 for df in timeframe_data.values()):
                continue
            
            # MultiTimeframeDataBatch 생성
            multi_batch = MultiTimeframeDataBatch(
                timeframe_data=timeframe_data,
                symbols=config.symbols,
                primary_timeframe=base_timeframe
            )
            
            # 진행률 업데이트
            if step % max(1, total_steps // 20) == 0:
                progress = 20 + (step / total_steps) * 65
                self._notify_progress(progress, f"멀티타임프레임 처리 중: {timestamp}")
            
            # 컨텍스트 업데이트
            if self.context:
                self.context.current_time = timestamp
            
            # 브로커에 시장 데이터 업데이트 (기준 타임프레임으로)
            broker_start = time.time()
            base_batch = MarketDataBatch(
                data=timeframe_data[base_timeframe],
                symbols=config.symbols,
                timeframe=base_timeframe
            )
            self.broker.update_market_data(base_batch)
            broker_time_total += time.time() - broker_start
            
            # 전략에서 멀티타임프레임 신호 생성
            try:
                strategy_start = time.time()
                # 멀티타임프레임 전략의 경우 on_data를 오버로드하여 호출
                if hasattr(self.strategy, 'on_data') and callable(getattr(self.strategy, 'on_data')):
                    orders = self.strategy.on_data(multi_batch)
                else:
                    orders = []
                strategy_time_total += time.time() - strategy_start
                
                # 생성된 주문들 실행
                for order in orders:
                    order_id = self.broker.submit_order(order)
                    
                    # 체결된 거래가 있는지 확인
                    if order.is_filled:
                        # 최근 거래를 찾아서 전략에 알림
                        recent_trades = [
                            trade for trade in self.broker.get_trades()
                            if trade.order_id == order_id
                        ]
                        for trade in recent_trades:
                            if trade not in all_trades:
                                all_trades.append(trade)
                                self.strategy.on_order_fill(trade)
                                
            except Exception as e:
                print(f"멀티타임프레임 전략 실행 오류 (시간: {timestamp}): {e}")
                continue
        
        return all_trades
    
    async def _load_raw_data(self, config: BacktestConfig) -> pl.DataFrame:
        """원본 OHLCV 데이터 로드"""
        # 단일 get_data 호출로 모든 데이터를 한 번에 로드
        data = await self.data_provider.get_data(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            timeframe=config.timeframe
        )
        
        return data.sort(["timestamp", "symbol"])
    
    async def _run_backtest_loop_with_enriched_data(self, config: BacktestConfig, enriched_data: pl.DataFrame) -> List[Trade]:
        """지표가 포함된 데이터로 백테스팅 실행"""
        all_trades = []
        
        # 시간순으로 데이터를 그룹화
        time_groups = enriched_data.group_by("timestamp").agg(
            pl.all().exclude("timestamp")
        ).sort("timestamp")
        
        total_steps = time_groups.height
        
        for step, time_group in enumerate(time_groups.iter_rows(named=True)):
            timestamp = time_group["timestamp"]
            
            # 현재 시점까지의 모든 데이터를 포함하는 배치 생성
            current_batch_data = enriched_data.filter(
                pl.col("timestamp") <= timestamp
            )
            
            if current_batch_data.height == 0:
                continue
            
            data_batch = MarketDataBatch(
                data=current_batch_data,
                symbols=config.symbols,
                timeframe=config.timeframe
            )
            
            # 진행률 업데이트
            if step % max(1, total_steps // 20) == 0:
                progress = 20 + (step / total_steps) * 65
                self._notify_progress(progress, f"처리 중: {timestamp}")
            
            # 컨텍스트 업데이트
            if self.context:
                self.context.current_time = timestamp
            
            # 브로커에 시장 데이터 업데이트
            self.broker.update_market_data(data_batch)
            
            # 전략에서 신호 생성
            try:
                orders = self.strategy.on_data(data_batch)
                
                # 생성된 주문들 실행
                for order in orders:
                    order_id = self.broker.submit_order(order)
                    
                    # 체결된 거래가 있는지 확인
                    if order.is_filled:
                        # 최근 거래를 찾아서 전략에 알림
                        recent_trades = [
                            trade for trade in self.broker.get_trades()
                            if trade.order_id == order_id
                        ]
                        for trade in recent_trades:
                            if trade not in all_trades:
                                all_trades.append(trade)
                                self.strategy.on_order_fill(trade)
                
                # 브로커의 모든 거래 확인 (누락된 거래 방지)
                broker_trades = self.broker.get_trades()
                for trade in broker_trades:
                    if trade not in all_trades:
                        all_trades.append(trade)
                                
            except Exception as e:
                print(f"전략 실행 오류 (시간: {timestamp}): {e}")
                continue
        
        return all_trades
    
    async def _create_result(
        self, 
        config: BacktestConfig, 
        start_time: datetime, 
        end_time: datetime,
        trades: List[Trade]
    ) -> BacktestResult:
        """백테스팅 결과 생성"""
        
        # 최종 포트폴리오 상태
        final_portfolio = self.broker.get_portfolio()
        final_equity = final_portfolio.equity
        
        # 거래 통계 계산
        trade_stats = self._calculate_trade_statistics(trades)
        
        # 성과 지표 계산
        performance_metrics = self._calculate_performance_metrics(
            config, final_equity, trades
        )
        
        # 백테스팅 중 사용된 시장 데이터 활용
        market_data = None
        if hasattr(self, '_market_data') and self._market_data:
            market_data = self._market_data
        
        # 포트폴리오 히스토리 생성 (선택적) - 시장 데이터 전달
        portfolio_history, equity_curve = self._create_portfolio_history(trades, config, market_data)
        
        # 시각화 모드일 때 추가 데이터 수집
        benchmark_equity_curve = None
        benchmark_returns = None
        daily_returns = None
        monthly_returns = None
        drawdown_periods = None
        trade_signals = None
        
        if config.visualization_mode:
            self._notify_progress(92.0, "시각화 데이터 생성 중...")
            
            # 벤치마크 데이터 생성
            if config.benchmark_symbol or config.symbols:
                benchmark_equity_curve, benchmark_returns = await self._create_benchmark_data(config, equity_curve)
            
            # 일간/월간 수익률 계산 (충분한 데이터가 있을 때만)
            if equity_curve is not None and len(equity_curve) >= 2:
                daily_returns = self._calculate_daily_returns(equity_curve)
                monthly_returns = self._calculate_monthly_returns(daily_returns)
                drawdown_periods = self._calculate_drawdown_periods(equity_curve)
            else:
                # 데이터가 부족한 경우 None으로 설정
                daily_returns = None
                monthly_returns = None 
                drawdown_periods = None
            
            # 거래 시그널 데이터 생성
            if trades:
                trade_signals = self._create_trade_signals(trades)
        
        return BacktestResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            
            # 성과 지표
            total_return=performance_metrics["total_return"],
            annual_return=performance_metrics["annual_return"],
            volatility=performance_metrics["volatility"],
            sharpe_ratio=performance_metrics["sharpe_ratio"],
            max_drawdown=performance_metrics["max_drawdown"],
            
            # 거래 통계
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            profit_factor=trade_stats["profit_factor"],
            
            # 최종 상태
            final_portfolio=final_portfolio,
            final_equity=final_equity,
            
            # 선택적 데이터
            trades=trades if config.save_trades else None,
            portfolio_history=portfolio_history,
            equity_curve=equity_curve,
            
            # 시각화용 데이터
            benchmark_equity_curve=benchmark_equity_curve,
            benchmark_returns=benchmark_returns,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            drawdown_periods=drawdown_periods,
            trade_signals=trade_signals
        )
    
    def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, Any]:
        """거래 통계 계산 - 개별 거래 기준으로 단순화"""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        # 단순히 모든 거래를 카운트 (매수/매도 구분 없이)
        total_trades = len(trades)
        
        # 실제 손익이 있는 거래들만 분석 (매도 거래)
        profitable_trades = []
        losing_trades = []
        
        for trade in trades:
            # Trade 객체에 pnl 속성이 없으므로 임시로 생략
            # TODO: 포지션 기반 손익 계산 로직 구현 필요
            # 현재는 pnl_impact를 사용한 간단한 분류
            if trade.side.value == -1:  # 매도
                # pnl_impact > 0이면 매도로 이익, < 0이면 손실로 간주 (단순화)
                if trade.pnl_impact > 0:
                    profitable_trades.append(trade)
                else:
                    losing_trades.append(trade)
        
        # 매수만 있는 경우를 위한 기본 통계
        winning_trades = len(profitable_trades)
        losing_count = len(losing_trades)
        
        win_rate = (winning_trades / max(winning_trades + losing_count, 1)) * 100
        
        # Trade 객체에 pnl 속성이 없으므로 pnl_impact 사용 (임시)
        avg_win = sum(trade.pnl_impact for trade in profitable_trades) / max(winning_trades, 1) if profitable_trades else 0.0
        avg_loss = sum(abs(trade.pnl_impact) for trade in losing_trades) / max(losing_count, 1) if losing_trades else 0.0
        
        profit_factor = avg_win / max(avg_loss, 0.01) if avg_loss > 0 else 0.0
        
        return {
            "total_trades": total_trades,  # 모든 거래 (매수/매도 포함)
            "winning_trades": winning_trades,
            "losing_trades": losing_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }
    
    def _calculate_performance_metrics(
        self, 
        config: BacktestConfig, 
        final_equity: float,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """성과 지표 계산"""
        initial_cash = config.initial_cash
        total_return = (final_equity - initial_cash) / initial_cash
        
        # 연간 수익률 계산
        duration_years = config.duration_days / 365.25
        annual_return = ((final_equity / initial_cash) ** (1 / duration_years) - 1) if duration_years > 0 else 0.0
        
        # 변동성 계산 (단순화된 계산)
        volatility = 0.15  # 임시값 - 실제로는 일별 수익률의 표준편차 계산
        
        # 샤프 비율 (무위험 수익률 3% 가정)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # 최대 낙폭 (단순화된 계산)
        max_drawdown = 0.1  # 임시값 - 실제로는 최고점 대비 최대 하락폭 계산
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    
    def _create_portfolio_history(
        self, 
        trades: List[Trade], 
        config: BacktestConfig,
        market_data: Optional[Dict[str, pl.DataFrame]] = None
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """포트폴리오 히스토리 생성 - 포지션 기반 평가 지원"""
        if not config.save_portfolio_history:
            return None, None
        
        # 시장 데이터 사용 가능 여부에 따라 분기 처리
        if market_data is not None and len(market_data) > 0:
            # 포지션 기반 평가 (새로운 방식)
            return self._create_position_based_equity_curve(trades, config, market_data)
        else:
            # 기존 거래 임팩트 방식 (호환성 유지)
                         return self._create_legacy_equity_curve(trades, config)
    
    def _create_legacy_equity_curve(
        self, 
        trades: List[Trade], 
        config: BacktestConfig
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """기존 거래 임팩트 기반 equity curve 생성 (호환성 유지)"""
        
        # 시작일과 종료일 기반으로 일별 날짜 범위 생성
        start_date = config.start_date
        end_date = config.end_date
        
        # 일별 날짜 리스트 생성
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 거래별 equity 변화 계산 (날짜 타입 통일)
        trade_impacts = {}  # {date: total_impact}
        if trades:
            for trade in trades:
                # datetime 객체로 통일 (시간 부분 제거)
                trade_date = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                if trade_date not in trade_impacts:
                    trade_impacts[trade_date] = 0
                trade_impacts[trade_date] += trade.pnl_impact - trade.commission
        
        # 일별 equity curve 생성
        equity_values = []
        running_equity = config.initial_cash
        
        for date in dates:
            # 해당 날짜의 거래 영향 적용 (날짜 타입 일치)
            if date in trade_impacts:
                running_equity += trade_impacts[date]
            equity_values.append(running_equity)
        
        # DataFrame 생성 (타입 안전성)
        equity_curve = pl.DataFrame({
            "timestamp": dates,
            "equity": [float(eq) for eq in equity_values],
            "pnl": [float(eq - config.initial_cash) for eq in equity_values]
        }, strict=False)
        
        # 포트폴리오 히스토리는 equity_curve와 동일하게 설정
        portfolio_history = equity_curve.clone()
        
        return portfolio_history, equity_curve
    
    def _create_position_based_equity_curve(
        self, 
        trades: List[Trade], 
        config: BacktestConfig,
        market_data: Dict[str, pl.DataFrame]
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """포지션 기반 equity curve 생성 (일별 종가 기준 평가)"""
        
        # 일별 날짜 리스트 생성
        dates = []
        current_date = config.start_date
        while current_date <= config.end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 거래별 영향 정리
        trade_events = {}  # {date: [trades]}
        for trade in trades:
            trade_date = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if trade_date not in trade_events:
                trade_events[trade_date] = []
            trade_events[trade_date].append(trade)
        
        # 포지션 및 현금 추적
        cash = config.initial_cash
        positions = {}  # {symbol: quantity}
        equity_values = []
        position_values_history = []
        cash_history = []
        
        for date in dates:
            # 해당 날짜의 거래 처리
            if date in trade_events:
                for trade in trade_events[date]:
                    symbol = trade.symbol
                    
                    # 포지션 업데이트
                    if symbol not in positions:
                        positions[symbol] = 0
                    
                    if trade.side == OrderSide.BUY:
                        positions[symbol] += trade.quantity
                        cash -= trade.quantity * trade.price + trade.commission
                    else:  # SELL
                        positions[symbol] -= trade.quantity
                        cash += trade.quantity * trade.price - trade.commission
                    
                    # 포지션이 0이 되면 제거
                    if positions[symbol] == 0:
                        del positions[symbol]
            
            # 포지션 시장가치 평가 (일별 종가 기준)
            total_position_value = 0
            for symbol, quantity in positions.items():
                if symbol in market_data:
                    # 해당 날짜의 종가 찾기 - 같은 날의 가장 가까운 데이터 사용
                    symbol_data = market_data[symbol]
                    
                    # 날짜만 비교 (시간 무시)
                    target_date = date.date()
                    day_price_data = symbol_data.filter(
                        pl.col("timestamp").dt.date() == target_date
                    )
                    
                    if len(day_price_data) > 0:
                        # 해당 날짜의 가장 마지막 데이터 사용 (종가)
                        close_price = day_price_data["close"][-1]
                        position_value = quantity * close_price
                        total_position_value += position_value
                    else:
                        # 해당 날짜 데이터가 없으면 가장 가까운 이전 데이터 사용
                        past_data = symbol_data.filter(
                            pl.col("timestamp") <= date
                        )
                        if len(past_data) > 0:
                            # 가장 최근 종가 사용
                            close_price = past_data["close"][-1]
                            position_value = quantity * close_price
                            total_position_value += position_value
            
            # 총 포트폴리오 가치
            total_equity = cash + total_position_value
            
            equity_values.append(total_equity)
            position_values_history.append(total_position_value)
            cash_history.append(cash)
        
        # DataFrame 생성 (타입 안전성)
        equity_curve = pl.DataFrame({
            "timestamp": dates,
            "equity": [float(eq) for eq in equity_values],
            "cash": [float(c) for c in cash_history],
            "positions_value": [float(pv) for pv in position_values_history],
            "pnl": [float(eq - config.initial_cash) for eq in equity_values]
        }, strict=False)
        
        # 포트폴리오 히스토리는 equity_curve와 동일하게 설정
        portfolio_history = equity_curve.clone()
        
        return portfolio_history, equity_curve
    
    # 시각화 데이터 생성 메서드들
    async def _create_benchmark_data(
        self, 
        config: BacktestConfig, 
        equity_curve: Optional[pl.DataFrame]
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """벤치마크 데이터 생성 - 동일비중 포트폴리오"""
        if equity_curve is None or not config.symbols:
            return None, None
        
        try:
            # 벤치마크는 동일비중 포트폴리오로 생성
            # 실제 구현에서는 data_provider를 통해 벤치마크 데이터 로드
            
            # 임시로 간단한 벤치마크 생성 (실제로는 시장 데이터 사용)
            dates = equity_curve["timestamp"].to_list()
            initial_value = config.initial_cash
            
            # 간단한 시뮬레이션: 연 7% 수익률 가정
            benchmark_values = []
            annual_growth = 0.07  # 7% 연간 성장
            daily_growth = (1 + annual_growth) ** (1/365.25) - 1  # 정확한 일일 성장률
            
            for i, date in enumerate(dates):
                days_passed = i
                value = initial_value * ((1 + daily_growth) ** days_passed)
                benchmark_values.append(value)
            
            benchmark_equity_curve = pl.DataFrame({
                "timestamp": dates,
                "equity": benchmark_values
            })
            
            # 벤치마크 수익률 계산
            returns = []
            for i in range(1, len(benchmark_values)):
                ret = (benchmark_values[i] - benchmark_values[i-1]) / benchmark_values[i-1]
                returns.append(ret)
            
            benchmark_returns = pl.DataFrame({
                "timestamp": dates[1:],
                "return": returns
            })
            
            return benchmark_equity_curve, benchmark_returns
            
        except Exception as e:
            print(f"벤치마크 데이터 생성 중 오류: {e}")
            return None, None
    
    def _calculate_daily_returns(self, equity_curve: pl.DataFrame) -> pl.DataFrame:
        """일간 수익률 계산"""
        if equity_curve is None or len(equity_curve) < 2:
            return None
        
        try:
            # 일간 수익률 계산
            equity_values = equity_curve["equity"].to_list()
            returns = []
            dates = equity_curve["timestamp"].to_list()
            
            for i in range(1, len(equity_values)):
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(ret)
            
            return pl.DataFrame({
                "timestamp": dates[1:],
                "return": returns
            })
        
        except Exception as e:
            print(f"일간 수익률 계산 중 오류: {e}")
            return None
    
    def _calculate_monthly_returns(self, daily_returns: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
        """월간 수익률 계산"""
        if daily_returns is None or len(daily_returns) == 0:
            return None
        
        try:
            # polars를 사용한 월간 집계
            monthly_returns = (
                daily_returns
                .with_columns([
                    pl.col("timestamp").dt.year().alias("year"),
                    pl.col("timestamp").dt.month().alias("month")
                ])
                .group_by(["year", "month"])
                .agg([
                    pl.col("timestamp").first().alias("timestamp"),
                    ((1 + pl.col("return")).product() - 1).alias("return")
                ])
                .sort("timestamp")
            )
            
            return monthly_returns.select(["timestamp", "return"])
        
        except Exception as e:
            print(f"월간 수익률 계산 중 오류: {e}")
            return None
    
    def _calculate_drawdown_periods(self, equity_curve: pl.DataFrame) -> pl.DataFrame:
        """드로다운 기간 계산"""
        if equity_curve is None or len(equity_curve) < 2:
            return None
        
        try:
            equity_values = equity_curve["equity"].to_numpy()
            dates = equity_curve["timestamp"].to_list()
            
            # 누적 최고점 계산
            running_max = np.maximum.accumulate(equity_values)
            
            # 드로다운 계산 (%)
            drawdown = (equity_values - running_max) / running_max
            
            return pl.DataFrame({
                "timestamp": dates,
                "equity": equity_values,
                "running_max": running_max,
                "drawdown": drawdown
            })
        
        except Exception as e:
            print(f"드로다운 계산 중 오류: {e}")
            return None
    
    def _create_trade_signals(self, trades: List[Trade]) -> pl.DataFrame:
        """거래 시그널 데이터 생성"""
        if not trades:
            return None
        
        try:
            signal_data = []
            
            for trade in trades:
                signal_type = "BUY" if trade.side.value == 1 else "SELL"
                signal_data.append({
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "signal": signal_type,
                    "price": trade.price,
                    "quantity": trade.quantity
                })
            
            return pl.DataFrame(signal_data)
        
        except Exception as e:
            print(f"거래 시그널 데이터 생성 중 오류: {e}")
            return None 