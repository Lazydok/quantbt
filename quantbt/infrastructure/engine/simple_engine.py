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
from ...core.entities.trade import Trade
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
        
        # 멀티타임프레임 전략인지 확인
        is_multi_timeframe = (MULTI_TIMEFRAME_AVAILABLE and 
                             isinstance(self.strategy, MultiTimeframeTradingStrategy) and
                             hasattr(self.strategy, 'timeframes') and
                             len(self.strategy.timeframes) > 1)
        
        if is_multi_timeframe:
            return await self._execute_multi_timeframe_backtest(config, start_time)
        else:
            return await self._execute_single_timeframe_backtest(config, start_time)
    
    async def _execute_single_timeframe_backtest(self, config: BacktestConfig, start_time: datetime) -> BacktestResult:
        """단일 타임프레임 백테스팅 실행 (기존 로직)"""
        # 데이터 로드 및 지표 사전 계산
        self._notify_progress(5.0, "데이터 로딩 중...")
        raw_data = await self._load_raw_data(config)
        
        self._notify_progress(10.0, "지표 계산 중...")
        enriched_data = self.strategy.precompute_indicators(raw_data)
        
        # 전략 초기화
        self.strategy.initialize(self.context)
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
        
        # 전략에서 타임프레임 정보 가져오기
        timeframes = self.strategy.timeframes
        base_timeframe = min(timeframes, key=TimeframeUtils.get_timeframe_minutes)
        
        # 멀티타임프레임 데이터 로드
        multi_data = await self.data_provider.get_multi_timeframe_data(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            timeframes=timeframes,
            base_timeframe=base_timeframe
        )
        
        self._notify_progress(10.0, "멀티타임프레임 지표 계산 중...")
        
        # 각 타임프레임별로 지표 계산
        enriched_multi_data = self.strategy.precompute_indicators_multi_timeframe(multi_data)
        
        # 전략 초기화
        self.strategy.initialize(self.context)
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
        all_data = []
        
        # 모든 데이터를 먼저 로드
        async for data_batch in self.data_provider.get_data_stream(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            timeframe=config.timeframe
        ):
            all_data.append(data_batch.data)
        
        # 모든 데이터를 하나의 DataFrame으로 합치기
        if all_data:
            combined_data = pl.concat(all_data)
            return combined_data.sort(["timestamp", "symbol"])
        else:
            # 빈 DataFrame 반환
            return pl.DataFrame({
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            })
    
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
        
        # 포트폴리오 히스토리 생성 (선택적)
        portfolio_history = None
        equity_curve = None
        if config.save_portfolio_history:
            portfolio_history, equity_curve = self._create_portfolio_history(trades, config)
        
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
            equity_curve=equity_curve
        )
    
    def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, Any]:
        """거래 통계 계산"""
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
        
        # 거래별 손익 계산 (단순화된 계산)
        trade_pnls = []
        for trade in trades:
            # 임시 계산 - 실제로는 포지션 단위로 계산해야 함
            pnl = trade.pnl_impact
            trade_pnls.append(pnl)
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = winning_count / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        total_wins = sum(winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_count,
            "losing_trades": losing_count,
            "win_rate": win_rate,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor)
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
        config: BacktestConfig
    ) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """포트폴리오 히스토리 생성"""
        if not trades:
            return None, None
        
        # 간단한 자본 곡선 생성
        timestamps = [trade.timestamp for trade in trades]
        equity_values = []
        
        running_equity = config.initial_cash
        for trade in trades:
            # 단순화된 자본 변화 계산
            running_equity += trade.pnl_impact - trade.commission
            equity_values.append(running_equity)
        
        equity_curve = pl.DataFrame({
            "timestamp": timestamps,
            "equity": equity_values,
            "pnl": [eq - config.initial_cash for eq in equity_values]
        })
        
        # 포트폴리오 히스토리는 equity_curve와 동일하게 설정
        portfolio_history = equity_curve.clone()
        
        return portfolio_history, equity_curve 