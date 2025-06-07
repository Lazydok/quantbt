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
        
        # 백테스팅 중 사용된 시장 데이터 활용
        market_data = None
        if hasattr(self, '_market_data') and self._market_data:
            market_data = self._market_data
        
        # 기본 성과 지표 계산용 경량 equity curve (항상 생성)
        lightweight_equity_curve = self._create_lightweight_equity_curve(trades, config, market_data)
        
        # 성과 지표 계산 - 경량 equity curve 사용하여 실제 값 계산
        performance_metrics = self._calculate_performance_metrics(
            config, final_equity, trades, lightweight_equity_curve
        )
        
        # 상세 포트폴리오 히스토리 생성 (선택적) - 시각화/분석용
        portfolio_history, equity_curve = self._create_portfolio_history(trades, config, market_data)
        
        # 경량 데이터 메모리 정리 (성과 지표 계산 완료 후)
        del lightweight_equity_curve
        
        # 상세 데이터 수집 (save_portfolio_history=True일 때)
        benchmark_equity_curve = None
        benchmark_returns = None
        daily_returns = None
        monthly_returns = None
        drawdown_periods = None
        trade_signals = None
        
        if config.save_portfolio_history:
            self._notify_progress(92.0, "시각화 데이터 생성 중...")
            
            # equity_curve가 없는 경우 기본 생성
            if equity_curve is None:
                portfolio_history, equity_curve = self._create_portfolio_history(trades, config, market_data)
            
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
            trades=trades if config.save_portfolio_history else None,
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
        """거래 통계 계산 - 실현 손익 기준 라운드 트립 분석"""
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
        
        # 포지션 청산 거래들만 분석 (실제 손익이 확정된 거래)
        position_closing_trades = [
            trade for trade in trades 
            if trade.is_position_closing and trade.realized_pnl != 0
        ]
        
        if not position_closing_trades:
            # 청산 거래가 없으면 전체 거래 수만 반환
            return {
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        # 수익/손실 거래 분류
        profitable_trades = [trade for trade in position_closing_trades if trade.realized_pnl > 0]
        losing_trades = [trade for trade in position_closing_trades if trade.realized_pnl < 0]
        
        # 통계 계산
        total_closing_trades = len(position_closing_trades)
        winning_trades = len(profitable_trades)
        losing_count = len(losing_trades)
        
        # 승률 계산 (0-1 범위로 반환, print_summary에서 백분율로 변환)
        win_rate = winning_trades / total_closing_trades if total_closing_trades > 0 else 0.0
        
        # 평균 수익/손실 계산
        total_profit = sum(trade.realized_pnl for trade in profitable_trades)
        total_loss = sum(abs(trade.realized_pnl) for trade in losing_trades)
        
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = total_loss / losing_count if losing_count > 0 else 0.0
        
        # Profit Factor 계산
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        return {
            "total_trades": len(trades),  # 전체 거래 수 (매수/매도 포함)
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
        trades: List[Trade],
        equity_curve: Optional[pl.DataFrame] = None
    ) -> Dict[str, float]:
        """성과 지표 계산 - 실제 equity_curve 기반"""
        initial_cash = config.initial_cash
        total_return = (final_equity - initial_cash) / initial_cash
        
        # 연간 수익률 계산
        duration_years = config.duration_days / 365.25
        annual_return = ((final_equity / initial_cash) ** (1 / duration_years) - 1) if duration_years > 0 else 0.0
        
        # 경량 equity_curve로 실제 값 계산 (항상 실제 값 사용)
        if equity_curve is not None and len(equity_curve) >= 2:
            # 실제 일별 수익률 기반 변동성 계산
            volatility, max_drawdown = self._calculate_real_volatility_and_mdd(equity_curve)
        else:
            # 시장 데이터가 없는 극단적인 경우에만 기본값 사용
            volatility = 0.15
            max_drawdown = 0.05
            print("⚠️ 시장 데이터 부족으로 성과 지표를 기본값으로 설정합니다.")
            print("정확한 계산을 위해 시장 데이터를 제공해주세요.")
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        risk_free_rate = 0.00
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    
    def _calculate_real_volatility_and_mdd(self, equity_curve: pl.DataFrame) -> tuple[float, float]:
        """실제 equity_curve를 사용한 변동성과 MDD 계산"""
        equity_values = equity_curve["equity"].to_numpy()
        
        # 일별 수익률 계산
        if len(equity_values) < 2:
            return 0.15, 0.05  # 기본값 반환
        
        daily_returns = np.diff(equity_values) / equity_values[:-1]
        
        # 변동성 계산 (연간화)
        volatility = np.std(daily_returns) * np.sqrt(365.25) if len(daily_returns) > 0 else 0.15
        
        # 최대 드로다운 계산
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.05
        
        return volatility, max_drawdown
    

    
    def _create_lightweight_equity_curve(
        self, 
        trades: List[Trade], 
        config: BacktestConfig,
        market_data: Optional[Dict[str, pl.DataFrame]] = None
    ) -> Optional[pl.DataFrame]:
        """기본 성과 지표 계산용 경량 equity curve 생성 (항상 실행)"""
        
        # 시장 데이터가 없으면 None 반환 (추정 로직은 _calculate_performance_metrics에서 처리)
        if market_data is None or len(market_data) == 0:
            print("⚠️ 시장 데이터가 없어 기본 성과 지표를 추정값으로 계산합니다.")
            return None
        
        # 경량 버전: 날짜와 equity만 저장 (벡터 연산 최적화)
        dates = []
        current_date = config.start_date
        while current_date <= config.end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 거래별 영향 정리 (딕셔너리로 빠른 접근)
        trade_events = {}
        for trade in trades:
            trade_date = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if trade_date not in trade_events:
                trade_events[trade_date] = []
            trade_events[trade_date].append(trade)
        
        # 벡터화된 계산을 위한 준비
        equity_values = np.full(len(dates), config.initial_cash, dtype=np.float64)
        cash = config.initial_cash
        positions = {}
        
        # 각 날짜별 처리 (벡터 연산 최적화)
        for i, date in enumerate(dates):
            # 거래 처리
            if date in trade_events:
                for trade in trade_events[date]:
                    symbol = trade.symbol
                    if symbol not in positions:
                        positions[symbol] = 0
                    
                    if trade.side == OrderSide.BUY:
                        positions[symbol] += trade.quantity
                        cash -= trade.quantity * trade.price + trade.commission
                    else:  # SELL
                        positions[symbol] -= trade.quantity
                        cash += trade.quantity * trade.price - trade.commission
                    
                    if positions[symbol] == 0:
                        del positions[symbol]
            
            # 포지션 평가 (벡터 연산)
            total_position_value = 0
            target_date = date.date()
            
            for symbol, quantity in positions.items():
                if symbol in market_data:
                    symbol_data = market_data[symbol]
                    
                    # 날짜 기준 데이터 필터링 (polars 벡터 연산)
                    day_price_data = symbol_data.filter(
                        pl.col("timestamp").dt.date() == target_date
                    )
                    
                    if len(day_price_data) > 0:
                        close_price = day_price_data["close"][-1]
                    else:
                        # 이전 데이터 사용
                        past_data = symbol_data.filter(pl.col("timestamp") <= date)
                        if len(past_data) > 0:
                            close_price = past_data["close"][-1]
                        else:
                            close_price = 0
                    
                    total_position_value += quantity * close_price
            
            equity_values[i] = cash + total_position_value
        
        # 경량 DataFrame 생성 (날짜와 equity만)
        lightweight_curve = pl.DataFrame({
            "timestamp": dates,
            "equity": equity_values.tolist()
        }, strict=False)
        
        return lightweight_curve
    
    def _create_portfolio_history(
        self, 
        trades: List[Trade], 
        config: BacktestConfig,
        market_data: Optional[Dict[str, pl.DataFrame]] = None
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """포트폴리오 히스토리 생성 - 포지션 기반 평가로 일원화"""
        if not config.save_portfolio_history:
            return None, None
        
        # 시장 데이터가 없으면 포지션 기반 평가 불가
        if market_data is None or len(market_data) == 0:
            print("⚠️ 포트폴리오 히스토리 생성 실패: 시장 데이터가 없습니다.")
            print("포지션 기반 평가를 위해서는 시장 데이터가 필요합니다.")
            return None, None
        
        # 포지션 기반 평가로 equity curve 생성
        return self._create_position_based_equity_curve(trades, config, market_data)
    
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
    
    def _sync_equity_curve_with_final_state(
        self, 
        equity_curve: pl.DataFrame, 
        final_portfolio
    ) -> pl.DataFrame:
        """equity_curve의 마지막 값을 브로커의 최종 상태와 동기화"""
        
        try:
            # 브로커의 최종 equity 값 
            final_equity_value = final_portfolio.equity
            
            # equity_curve의 마지막 행 업데이트
            equity_data = equity_curve.to_dict(as_series=False)
            
            # 마지막 인덱스
            last_idx = len(equity_data['equity']) - 1
            
            # 마지막 값들 업데이트
            equity_data['equity'][last_idx] = final_equity_value
            
            # cash와 positions_value가 있는 경우 업데이트
            if 'cash' in equity_data:
                equity_data['cash'][last_idx] = final_portfolio.cash
            
            if 'positions_value' in equity_data:
                equity_data['positions_value'][last_idx] = final_portfolio.total_market_value
            
            # pnl 다시 계산 (초기 자본 기준)
            if 'pnl' in equity_data:
                initial_cash = equity_data['equity'][0]  # 첫 번째 값이 초기 자본
                equity_data['pnl'][last_idx] = final_equity_value - initial_cash
            
            # 새로운 DataFrame 생성
            synchronized_equity_curve = pl.DataFrame(equity_data, strict=False)
            
            return synchronized_equity_curve
            
        except Exception as e:
            print(f"⚠️ equity_curve 동기화 중 오류: {e}")
            # 오류 발생 시 원본 반환
            return equity_curve
    
    # 시각화 데이터 생성 메서드들
    async def _create_benchmark_data(
        self, 
        config: BacktestConfig, 
        equity_curve: Optional[pl.DataFrame]
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """벤치마크 데이터 생성 - 균등분산 바이앤홀드 포트폴리오 (polars 벡터 연산)"""
        if equity_curve is None or not config.symbols:
            return None, None
        
        try:
            # 시장 데이터 준비
            if not hasattr(self, '_market_data') or not self._market_data:
                await self._load_market_data_for_benchmark(config)
            
            market_data = getattr(self, '_market_data', {})
            if not market_data:
                print("시장 데이터가 없어 벤치마크를 생성할 수 없습니다.")
                return None, None
            
            # 모든 심볼 데이터를 하나의 DataFrame으로 합치기
            all_symbol_data = []
            for symbol in config.symbols:
                if symbol in market_data:
                    symbol_df = market_data[symbol].select([
                        "timestamp", 
                        "close"
                    ]).with_columns(pl.lit(symbol).alias("symbol"))
                    all_symbol_data.append(symbol_df)
            
            if not all_symbol_data:
                print("유효한 심볼 데이터가 없습니다.")
                return None, None
            
            # 모든 데이터 결합
            combined_data = pl.concat(all_symbol_data)
            
            # 날짜별 평균 종가 계산 (polars 벡터 연산) - 날짜만 사용
            benchmark_avg_data = (
                combined_data
                .with_columns(pl.col("timestamp").dt.date().alias("date"))
                .group_by("date")
                .agg(pl.col("close").mean().alias("avg_close"))
                .sort("date")
            )
            
            # equity_curve 날짜 범위에 맞춘 데이터 필터링 - 날짜만 사용
            equity_dates = equity_curve.with_columns(
                pl.col("timestamp").dt.date().alias("date")
            ).select(["timestamp", "date"])
            
            benchmark_data = equity_dates.join(
                benchmark_avg_data, 
                on="date", 
                how="left"
            ).fill_null(strategy="forward")  # 누락된 날짜는 forward fill
            
            # 첫 번째 값으로 정규화 (벡터 연산) - 시작점 = 1.0
            first_avg_close = benchmark_data["avg_close"][0]
            if first_avg_close is None or first_avg_close <= 0:
                print("벤치마크 첫 번째 가격 데이터가 유효하지 않습니다.")
                return None, None
            
            benchmark_equity_curve = benchmark_data.with_columns(
                (pl.col("avg_close") / first_avg_close).alias("equity")
            ).select(["timestamp", "equity"])
            
            # 수익률 계산 (polars 벡터 연산)
            benchmark_returns = benchmark_equity_curve.with_columns(
                pl.col("equity").pct_change().alias("return")
            ).drop_nulls().select(["timestamp", "return"])
            
            return benchmark_equity_curve, benchmark_returns
            
        except Exception as e:
            print(f"벤치마크 데이터 생성 중 오류: {e}")
            return None, None
    
    async def _load_market_data_for_benchmark(self, config: BacktestConfig) -> None:
        """벤치마크용 시장 데이터 로드"""
        timeframe = config.timeframe or config.base_timeframe
        market_data = {}
        
        for symbol in config.symbols:
            try:
                symbol_data = await self.data_provider.get_data(
                    symbols=[symbol],
                    start=config.start_date,
                    end=config.end_date,
                    timeframe=timeframe
                )
                if len(symbol_data) > 0:
                    market_data[symbol] = symbol_data.filter(pl.col("symbol") == symbol)
            except Exception as e:
                print(f"심볼 {symbol} 데이터 로드 실패: {e}")
                continue
        
        self._market_data = market_data
    
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