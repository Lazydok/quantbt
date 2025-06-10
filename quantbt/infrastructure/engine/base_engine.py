"""
Dict Native 백테스팅 엔진

List[Dict] → Dict 직접 전달 방식으로 최고 성능을 달성하는 엔진

핵심 설계 원리:
1. Zero Conversion: 중간 변환 완전 제거
2. Direct Access: List[i] 직접 전달
3. Simple Interface: Dict 기반 단순 인터페이스
4. Maximum Performance: 순수 Python 성능 극대화
"""

import asyncio, nest_asyncio
import time
import random
import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from tqdm import tqdm
import traceback
from IPython import get_ipython

def _is_jupyter_environment() -> bool:
    """현재 실행 환경이 Jupyter notebook인지 확인"""
    try:
        ipython = get_ipython()
        return ipython is not None and ipython.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False

from ...core.interfaces.backtest_engine import BacktestEngineBase
from ...core.value_objects.backtest_config import BacktestConfig
from ...core.value_objects.backtest_result import BacktestResult
from ...core.strategies.dict_based import DictTradingStrategy
from ...core.interfaces.strategy import TradingStrategy  # Phase 7 하이브리드 전략 지원
from ...core.entities.order import Order, OrderSide, OrderType, OrderStatus
from ...core.entities.trade import Trade
from ...infrastructure.brokers.simple_broker import SimpleBroker
from ...infrastructure.data.upbit_provider import UpbitDataProvider
from ...core.entities.position import Position

class BacktestEngine(BacktestEngineBase):
    """백테스팅 엔진
    """
    
    def __init__(self):
        """백테스팅 엔진 초기화"""
        super().__init__(name="BacktestEngine")
        self.strategy: Optional[Union[DictTradingStrategy, TradingStrategy]] = None
        self.broker: Optional[SimpleBroker] = None
        self.data_provider: Optional[UpbitDataProvider] = None
        self.pending_orders: List[Dict[str, Any]] = []
        self.filled_orders: List[Dict[str, Any]] = []
        self.failed_orders: List[Dict[str, Any]] = []
        self.execution_mode: str = "open"  # "open" 또는 "close"
        
        # 데이터 캐시 (중복 로딩 방지)
        self._cached_market_data: Optional[pl.DataFrame] = None
        self._cached_daily_market_data: Optional[pl.DataFrame] = None
        
    def set_strategy(self, strategy: Union[DictTradingStrategy, TradingStrategy]):
        """Dict 기반 전략 또는 Phase 7 하이브리드 전략 설정"""
        self.strategy = strategy
        
    def set_broker(self, broker: SimpleBroker):
        """브로커 설정"""
        self.broker = broker
        if self.strategy:
            self.strategy.set_broker(broker)
            
    def set_data_provider(self, data_provider: UpbitDataProvider):
        """데이터 제공자 설정"""
        self.data_provider = data_provider
    
    def _update_broker_market_data(self, current_candle: Dict[str, Any]):
        """Dict Native 방식: 간단한 가격 정보만 브로커에 업데이트"""
        # Dict Native에서는 복잡한 MarketDataBatch 대신 간단한 가격 딕셔너리만 사용
        symbol = current_candle['symbol']
        price_dict = {symbol: current_candle['close']}
        
        # 브로커 포트폴리오의 시장 가격만 업데이트 (Dict Native는 독립적 처리)
        self.broker.portfolio.update_market_prices(price_dict)
        
    def _add_order_to_queue(self, order: Order, candle_index: int, signal_price: float):
        """주문을 대기 큐에 추가 (미래 참조 방지)
        
        Args:
            order: 주문 객체
            candle_index: 현재 캔들 인덱스
            signal_price: 신호 생성 시점의 가격
        """
        pending_order = {
            'order': order,
            'execute_at_candle': candle_index + 1,  # 다음 캔들에서 체결
            'signal_price': signal_price,
            'status': 'PENDING'
        }
        self.pending_orders.append(pending_order)

    
    def _get_ready_orders(self, current_candle_index: int) -> List[Dict[str, Any]]:
        """현재 캔들에서 실행할 주문들 조회
        
        Args:
            current_candle_index: 현재 캔들 인덱스
            
        Returns:
            실행 가능한 주문 리스트
        """
        ready_orders = []
        for pending_order in self.pending_orders:
            if (pending_order['status'] == 'PENDING' and 
                pending_order['execute_at_candle'] <= current_candle_index):
                ready_orders.append(pending_order)
        
        return ready_orders

    def _execute_pending_order(self, pending_order: Dict[str, Any], 
                             current_candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """대기 중인 주문 체결 처리
        
        Args:
            pending_order: 대기 중인 주문 정보
            current_candle: 현재 캔들 데이터
            
        Returns:
            체결 정보 Dict (실패 시 None)
        """
        order = pending_order['order']
        signal_price = pending_order['signal_price']
        
        # 체결 가격 결정 (현재 캔들 시가 기준)
        if self.execution_mode == "close":
            execution_price = current_candle['close']
        else:  # "open" (default)
            execution_price = current_candle['open']
        
        # 슬리피지 적용 (간단한 랜덤 슬리피지)
        slippage_rate = random.uniform(-0.001, 0.001)  # ±0.1%
        actual_slippage = slippage_rate
        if order.side == OrderSide.BUY:
            final_price = execution_price * (1 + abs(slippage_rate))  # 매수는 불리하게
        else:
            final_price = execution_price * (1 - abs(slippage_rate))  # 매도는 불리하게
        

        
        # Dict Native 방식: 브로커 우회하여 직접 주문 체결
        try:
            # 브로커 포트폴리오 가져오기
            portfolio = self.broker.get_portfolio()
            
            # Dict Native에서 포트폴리오 직접 업데이트
            if order.side == OrderSide.BUY:
                # 매수: 현금 차감, 포지션 증가
                cost = order.quantity * final_price
                if portfolio.cash >= cost:
                    portfolio.cash -= cost
                    current_position = portfolio.get_position(order.symbol)
                    
                    # 포지션 업데이트
                    total_quantity = current_position.quantity + order.quantity
                    if current_position.quantity > 0:
                        # 기존 포지션이 있는 경우 평균 단가 계산
                        total_cost = (current_position.quantity * current_position.average_price) + cost
                        avg_price = total_cost / total_quantity
                    else:
                        # 신규 포지션
                        avg_price = final_price
                    
                    # 포지션 정보 업데이트 - 브로커의 메서드 사용
                    new_position = Position(
                        symbol=order.symbol,
                        quantity=total_quantity,
                        avg_price=avg_price,
                        market_price=final_price
                    )
                    portfolio.positions[order.symbol] = new_position
                    

                    
                else:
                    self.pending_orders.remove(pending_order)
                    pending_order['status'] = 'FAILED'
                    self.failed_orders.append(pending_order)

                    return None
                    
            else:  # SELL
                # 매도: 포지션 감소, 현금 증가
                current_position = portfolio.get_position(order.symbol)
                if current_position.quantity >= order.quantity:
                    proceeds = order.quantity * final_price
                    portfolio.cash += proceeds
                    
                    # 포지션 업데이트
                    new_quantity = current_position.quantity - order.quantity
                    new_position = Position(
                        symbol=order.symbol,
                        quantity=new_quantity,
                        avg_price=current_position.avg_price,  # 평단은 유지
                        market_price=final_price
                    )
                    portfolio.positions[order.symbol] = new_position
                    

                    
                else:
                    self.pending_orders.remove(pending_order)
                    pending_order['status'] = 'FAILED'
                    self.failed_orders.append(pending_order)

                    return None
            
            # 체결 정보 생성
            trade_info = {
                'timestamp': current_candle['timestamp'],
                'symbol': order.symbol,
                'side': order.side.name,
                'quantity': order.quantity,
                'signal_price': signal_price,
                'execution_price': final_price,
                'slippage': actual_slippage,
                'slippage_amount': abs(final_price - execution_price),
                'order_id': f"dict_native_{int(time.time() * 1000)}"
            }
            
            # Trade 객체 생성하여 브로커에 추가
            trade_obj = Trade(
                order_id=trade_info['order_id'],
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=final_price,
                timestamp=current_candle['timestamp'],
                commission=0.0,  # Dict Native에서는 단순화
                slippage=actual_slippage
            )
            self.broker.trades.append(trade_obj)
            
            # 체결 완료 후 대기열에서 제거
            self.pending_orders.remove(pending_order)
            
            # 체결 완료 리스트에 추가
            pending_order['status'] = 'FILLED'
            self.filled_orders.append(pending_order)
            
            
            
            
            return trade_info
                
        except Exception as e:
            # pending_order['status'] = 'FAILED'
            # Pending 주문 상태 유지
            return None
    
    async def _load_raw_data_as_polars_async(self, config: BacktestConfig) -> pl.DataFrame:
        """원본 데이터를 Polars DataFrame 형태로 로딩 (비동기 버전)
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            캔들 데이터 Polars DataFrame (시간순 정렬됨)
        """
        if not self.data_provider:
            raise ValueError("데이터 제공자가 설정되지 않았습니다")
        
        # 데이터 조회 - 비동기 방식
        symbol_data = await self.data_provider.get_data(
            symbols=config.symbols,
            start=config.start_date,
            end=config.end_date,
            timeframe=config.timeframe
        )
        
        # 시간순 정렬하여 반환 및 캐시 저장
        sorted_data = symbol_data.sort("timestamp")
        self._cached_market_data = sorted_data  # 캐시 저장 (중복 로딩 방지)
        return sorted_data

    def _load_raw_data_as_polars(self, config: BacktestConfig) -> pl.DataFrame:
        """원본 데이터를 Polars DataFrame 형태로 로딩
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            캔들 데이터 Polars DataFrame (시간순 정렬됨)
        """
        if not self.data_provider:
            raise ValueError("데이터 제공자가 설정되지 않았습니다")
        
        # 실행 환경에 따라 다른 방식으로 비동기 함수 호출
        if _is_jupyter_environment():
            # Jupyter notebook 환경에서는 await 사용
            # 이 경우 호출하는 메서드도 async로 만들어야 함
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 이벤트 루프가 있는 경우
                nest_asyncio.apply()
                symbol_data = asyncio.run(self.data_provider.get_data(
                    symbols=config.symbols,
                    start=config.start_date,
                    end=config.end_date,
                    timeframe=config.timeframe
                ))
            else:
                # 새로운 이벤트 루프 시작
                symbol_data = loop.run_until_complete(self.data_provider.get_data(
                    symbols=config.symbols,
                    start=config.start_date,
                    end=config.end_date,
                    timeframe=config.timeframe
                ))
        else:
            # 일반 Python 스크립트에서는 asyncio.run 사용
            symbol_data = asyncio.run(self.data_provider.get_data(
                symbols=config.symbols,
                start=config.start_date,
                end=config.end_date,
                timeframe=config.timeframe
            ))
        
        # 시간순 정렬하여 반환 및 캐시 저장
        sorted_data = symbol_data.sort("timestamp")
        self._cached_market_data = sorted_data  # 캐시 저장 (중복 로딩 방지)
        return sorted_data
    
    def _run_dict_native_backtest_loop(self, config: BacktestConfig, 
                                           enriched_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dict Native 백테스팅 루프 - 1 캔들 = 1 스텝
        
        Args:
            config: 백테스팅 설정
            enriched_data: 지표가 계산된 Dict 형태 시장 데이터
            
        Returns:
            거래 정보 리스트 (Dict 형태)
        """
        trades = []
        self.pending_orders = []  # 주문 대기열 초기화
        
        # 실시간 포트폴리오 평가금 추적 초기화
        self._portfolio_equity_history = {}  # {timestamp: equity}
        
        # tqdm 프로그레스바 생성 (save_portfolio_history=True일 때만)
        pbar = None
        if config.save_portfolio_history:
            pbar = self.create_progress_bar(len(enriched_data), "백테스팅 진행")
        
        try:
            # Dict Native 루프: 극고속 처리
            for i, current_candle in enumerate(enriched_data):
                
                # 0단계: 브로커에게 현재 시장 데이터 업데이트 (Dict 형태로 변환)
                try:
                    # Dict 형태의 시장 데이터를 MarketDataBatch 형태로 변환
                    self._update_broker_market_data(current_candle)
                except Exception as e:
                    # 브로커 시장 데이터 업데이트 실패 시 계속 진행
                    pass
                
                # 1단계: 이전 신호로 생성된 주문들 체결
                ready_orders = self._get_ready_orders(i)
                
                for pending_order in ready_orders:
                    trade_info = self._execute_pending_order(pending_order, current_candle)
                    if trade_info:
                        trades.append(trade_info)
                    
                
                # 2단계: 현재 캔들에서 신호 생성
                try:
                    signals = self.strategy.generate_signals_dict(current_candle)
                    
                    # 3단계: 신호를 주문 대기열에 추가 (다음 캔들에서 체결)
                    for order in signals:
                        signal_price = current_candle['close']  # 신호 생성 시점 가격
                        self._add_order_to_queue(order, i, signal_price)
                        
                except Exception as e:
                    # 신호 생성 오류 시 해당 캔들 건너뛰고 계속 진행
                    traceback.print_exc()
                    continue
                
                # 4단계: 현재 시점 포트폴리오 평가금 계산 및 저장
                self._calculate_and_store_portfolio_equity(current_candle, config)
                
                # 프로그레스바 업데이트 (생성된 경우에만)
                if pbar is not None:
                    timestamp = current_candle.get('timestamp', 'N/A')
                    self.update_progress_bar(pbar, f"처리중... {i+1}/{len(enriched_data)} ({timestamp})")
        
        finally:
            # 프로그레스바 정리 (생성된 경우에만)
            if pbar is not None:
                pbar.close()
        
        # 마지막 캔들에서 남은 주문들 처리
        if enriched_data:
            last_candle = enriched_data[-1]
            final_ready_orders = self._get_ready_orders(len(enriched_data))
            for pending_order in final_ready_orders:
                trade_info = self._execute_pending_order(pending_order, last_candle)
                if trade_info:
                    trades.append(trade_info)

        return trades
    
    def _calculate_and_store_portfolio_equity(self, current_candle: Dict[str, Any], config: BacktestConfig) -> None:
        """현재 시점에서 포트폴리오 평가금 계산 및 저장"""
        try:
            # 현재 포트폴리오 상태 가져오기
            portfolio = self.broker.get_portfolio()
            current_cash = portfolio.cash
            
            # 포지션 평가 (현재 캔들의 종가 기준)
            total_position_value = 0.0
            current_price = current_candle.get('close', 0.0)
            
            for symbol, position in portfolio.positions.items():
                if position.quantity > 0:
                    # 모든 포지션을 현재 캔들의 종가로 평가
                    # (현재는 단일 심볼이므로 current_candle의 종가 사용)
                    if symbol == current_candle.get('symbol', ''):
                        total_position_value += position.quantity * current_price
                    else:
                        # 다른 심볼의 경우 평균 매수가 사용 (fallback)
                        total_position_value += position.quantity * position.avg_price
            
            # 총 포트폴리오 평가금 = 현금 + 포지션 평가금
            total_equity = current_cash + total_position_value
            
            # 타임스탬프로 저장
            timestamp = current_candle['timestamp']
            self._portfolio_equity_history[timestamp] = total_equity
            
        except Exception as e:
            # 오류 발생시 무시하고 계속 진행
            pass
    
    def _convert_dict_trades_to_objects(self, trades_dict_list: List[Dict[str, Any]]) -> List[Trade]:
        """Dict 거래 정보를 Trade 객체로 변환"""
        trade_objects = []
        
        for i, trade_dict in enumerate(trades_dict_list):
            trade_obj = Trade(
                trade_id=trade_dict.get('order_id', f"dict_native_trade_{i}"),
                order_id=trade_dict.get('order_id', f"dict_native_order_{i}"),
                timestamp=trade_dict['timestamp'],
                symbol=trade_dict['symbol'],
                side=OrderSide.BUY if trade_dict['side'] == 'BUY' else OrderSide.SELL,
                quantity=trade_dict['quantity'],
                price=trade_dict['execution_price'],
                commission=0.0,  # Dict Native에서는 브로커 수수료 별도 처리
                slippage=trade_dict.get('slippage', 0.0),  # 슬리피지 정보 추가
                
                # Dict Native 추가 정보 메타데이터로 보존
                metadata={
                    'signal_price': trade_dict.get('signal_price'),
                    'slippage': trade_dict.get('slippage'),
                    'slippage_amount': trade_dict.get('slippage_amount'),
                    'dict_native_execution': True
                }
            )
            trade_objects.append(trade_obj)
        
        return trade_objects
    
    def _calculate_trade_statistics_dict(self, trades: List[Trade]) -> Dict[str, Any]:
        """Dict Native 기반 거래 통계 계산"""
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
        
        # 매수/매도 쌍으로 수익/손실 계산
        positions = {}  # {symbol: {'quantity': float, 'total_cost': float}}
        realized_trades = []  # 실현 손익이 있는 거래들
        
        for trade in trades:
            symbol = trade.symbol
            if symbol not in positions:
                positions[symbol] = {'quantity': 0.0, 'total_cost': 0.0}
            
            pos = positions[symbol]
            
            if trade.side == OrderSide.BUY:
                # 매수: 포지션 증가
                pos['total_cost'] += trade.quantity * trade.price
                pos['quantity'] += trade.quantity
            else:  # SELL
                # 매도: 포지션 감소 및 손익 실현
                if pos['quantity'] > 0:
                    # 평균 매수가 계산
                    avg_buy_price = pos['total_cost'] / pos['quantity'] if pos['quantity'] > 0 else 0
                    
                    # 실현 손익 계산
                    sell_quantity = min(trade.quantity, pos['quantity'])
                    realized_pnl = sell_quantity * (trade.price - avg_buy_price)
                    realized_trades.append(realized_pnl)
                    
                    # 포지션 업데이트
                    cost_reduction = (sell_quantity / pos['quantity']) * pos['total_cost']
                    pos['total_cost'] -= cost_reduction
                    pos['quantity'] -= sell_quantity
        
        # 통계 계산
        total_trades = len(trades)
        profitable_trades = [pnl for pnl in realized_trades if pnl > 0]
        losing_trades = [pnl for pnl in realized_trades if pnl < 0]
        
        winning_trades = len(profitable_trades)
        losing_count = len(losing_trades)
        
        # 승률 (0-1 범위)
        win_rate = winning_trades / len(realized_trades) if realized_trades else 0.0
        
        # 평균 수익/손실
        avg_win = sum(profitable_trades) / winning_trades if profitable_trades else 0.0
        avg_loss = abs(sum(losing_trades)) / losing_count if losing_trades else 0.0
        
        # Profit Factor
        total_profit = sum(profitable_trades)
        total_loss = abs(sum(losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }
    
    def _get_portfolio_equity_history(self) -> List[float]:
        """실시간 포트폴리오 평가금 데이터 반환 (최적화)"""
        
        # 실시간 포트폴리오 평가금 데이터가 있는지 확인
        if hasattr(self, '_portfolio_equity_history') and self._portfolio_equity_history:
            return list(self._portfolio_equity_history.values())
        
        # 없으면 오류
        raise ValueError("실시간 포트폴리오 평가금 데이터가 없습니다. save_portfolio_history=True로 설정하거나 백테스팅을 다시 실행하세요.")
        
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """최대 드로다운 계산"""
        if len(equity_curve) < 2:
            return 0.05  # 기본값
        
        # numpy를 사용한 효율적인 계산
        equity_array = np.array(equity_curve)
        
        # 누적 최고점 계산
        running_max = np.maximum.accumulate(equity_array)
        
        # 드로다운 계산 (각 시점에서의 손실률)
        drawdown = (equity_array - running_max) / running_max
        
        # 최대 드로다운 (가장 큰 손실)
        max_drawdown = np.min(drawdown)  # 이미 음수이므로 min 사용
        
        # 절댓값으로 반환 (양수로 표시)
        return abs(max_drawdown) if len(drawdown) > 0 else 0.05
    
    # === Phase 8 추가: 완전한 시각화 데이터 생성 메서드들 ===
    
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
    
    def _calculate_drawdown_periods(self, equity_curve: pl.DataFrame) -> Optional[pl.DataFrame]:
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
    
    def _create_trade_signals(self, trades: List[Trade]) -> Optional[pl.DataFrame]:
        """거래 신호 데이터 생성"""
        if not trades:
            return None
        
        try:
            signal_data = []
            for trade in trades:
                signal_data.append({
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "signal": trade.side.name,  # "BUY" or "SELL"
                    "price": trade.price,
                    "quantity": trade.quantity
                })
            
            return pl.DataFrame(signal_data)
        except Exception as e:
            print(f"거래 신호 데이터 생성 중 오류: {e}")
            return None
    
    def _create_benchmark_data(self, config: BacktestConfig, 
                                   equity_curve: pl.DataFrame) -> Optional[pl.DataFrame]:
        """벤치마크 데이터 생성 (Buy & Hold) - 일자별 종가 기준"""
        try:
            # 첫 번째 심볼을 벤치마크로 사용 (Buy & Hold)
            benchmark_symbol = config.symbols[0]
            
            # 캐시된 시장 데이터 재사용 (중복 로딩 방지!)
            if self._cached_market_data is not None:
                market_data = self._cached_market_data.filter(pl.col("symbol") == benchmark_symbol)
            else:
                # 캐시가 없는 경우에만 새로 로딩
                import asyncio
                market_data = asyncio.run(self.data_provider.get_data(
                    symbols=[benchmark_symbol],
                    start=config.start_date,
                    end=config.end_date,
                    timeframe=config.timeframe
                ))
            
            if market_data is None or len(market_data) == 0:
                return None
            
            # 일자별 종가 데이터 추출
            daily_prices = market_data.group_by(
                pl.col("timestamp").dt.date().alias("date")
            ).agg(
                pl.col("close").last().alias("close"),
                pl.col("timestamp").last().alias("timestamp")
            ).sort("timestamp")
            
            # Buy & Hold 전략으로 벤치마크 계산
            initial_price = daily_prices["close"][0]  # 첫 번째 종가
            initial_cash = config.initial_cash
            shares = initial_cash / initial_price
            
            benchmark_data = []
            for row in daily_prices.iter_rows(named=True):
                equity = shares * row["close"]
                benchmark_data.append({
                    "timestamp": row["timestamp"].replace(hour=0, minute=0, second=0, microsecond=0),
                    "equity": equity,
                    "price": row["close"]
                })
            
            benchmark_df = pl.DataFrame(benchmark_data)
            
            return benchmark_df
            
        except Exception as e:
            traceback.print_exc()
            return None
    
    def _create_equity_curve_polars(self, config: BacktestConfig, trade_objects: List[Trade]) -> Optional[pl.DataFrame]:
        """실제 포트폴리오 평가금 계산 (unrealized_pnl + 포지션진입금 + 현금)"""
        if not config.save_portfolio_history:
            return None
        
        # 시장 데이터 로드 (정확한 종가 평가를 위해)
        market_data = None
        try:
            # 캐시된 일봉 데이터 재사용 (중복 로딩 방지!)
            if self._cached_daily_market_data is not None:
                market_data = self._cached_daily_market_data
            elif hasattr(self, 'data_provider') and self.data_provider:
                # 캐시가 없는 경우에만 새로 로딩
                import asyncio
                market_data = asyncio.run(self.data_provider.get_data(
                    symbols=config.symbols,
                    start=config.start_date,
                    end=config.end_date,
                    timeframe="1d"  # 일봉 데이터 사용 (종가 평가용)
                ))
                # 일봉 데이터 캐시 저장
                self._cached_daily_market_data = market_data
    
        except Exception as e:
            print(f"⚠️ 시장 데이터 로드 실패, 거래 데이터로 대체: {e}")
            market_data = None
        
        # 일자별 포인트 생성 (종가 기준)
        dates = []
        current_date = config.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = config.end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 거래 이벤트 매핑 (일별)
        trade_events = {}
        for i, trade in enumerate(trade_objects):
            trade_date = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if trade_date not in trade_events:
                trade_events[trade_date] = []
            trade_events[trade_date].append(trade)
        
        # 일별 종가 데이터 매핑 생성 (심볼별)
        daily_close_prices = {}  # {date: {symbol: close_price}}
        latest_prices = {}  # {symbol: latest_price} - 마지막 알려진 가격
        
        if market_data is not None and hasattr(market_data, 'iter_rows'):
            # 시장 데이터에서 일별 종가 추출
            for row in market_data.iter_rows(named=True):
                # 일봉 데이터는 보통 09:00 등으로 되어 있으므로 날짜만 추출
                date = row['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
                symbol = row.get('symbol', config.symbols[0] if config.symbols else 'DEFAULT')
                close_price = row['close']
                
                if date not in daily_close_prices:
                    daily_close_prices[date] = {}
                daily_close_prices[date][symbol] = close_price
                latest_prices[symbol] = close_price  # 마지막 가격 추적
        else:
            # 시장 데이터가 없으면 거래 데이터에서 종가 추정
            for trade in trade_objects:
                date = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                symbol = trade.symbol
                
                if date not in daily_close_prices:
                    daily_close_prices[date] = {}
                    
                daily_close_prices[date][symbol] = trade.price
                latest_prices[symbol] = trade.price  # 마지막 가격 추적
        
        # 시계열 데이터 생성
        timestamps = []
        equity_values = []
        cash_values = []
        position_costs = []
        unrealized_pnls = []
        pnl_values = []
        
        # 포트폴리오 시계열 구성
        cash = config.initial_cash
        positions = {}  # {symbol: {'quantity': float, 'total_cost': float}}
        
        for i, date in enumerate(dates):
            # 해당 날짜의 거래 처리
            # 날짜 매칭 시 timezone 정보 제거하여 정확한 매칭 보장
            date_key = None
            for trade_date in trade_events.keys():
                if trade_date.date() == date.date():
                    date_key = trade_date
                    break
            
            if date_key is not None and date_key in trade_events:
                day_trades = trade_events[date_key]
                
                for trade in day_trades:
                    symbol = trade.symbol
                    if symbol not in positions:
                        positions[symbol] = {'quantity': 0.0, 'total_cost': 0.0}
                    
                    if trade.side == OrderSide.BUY:
                        trade_cost = trade.quantity * trade.price
                        cash -= trade_cost
                        
                        # 포지션 업데이트
                        positions[symbol]['quantity'] += trade.quantity
                        positions[symbol]['total_cost'] += trade_cost
                        
                    else:  # SELL
                        trade_proceeds = trade.quantity * trade.price
                        cash += trade_proceeds
                        
                        # 포지션 업데이트 (비례 배분으로 평균 단가 유지)
                        if positions[symbol]['quantity'] > 0:
                            cost_ratio = trade.quantity / positions[symbol]['quantity']
                            cost_reduction = positions[symbol]['total_cost'] * cost_ratio
                            
                            positions[symbol]['quantity'] -= trade.quantity
                            positions[symbol]['total_cost'] -= cost_reduction
                            
                            # 0이 된 포지션 제거
                            if positions[symbol]['quantity'] <= 1e-8:  # floating point 오차 고려
                                del positions[symbol]
            
            # 현재 날짜의 종가로 포지션 평가 (심볼별 처리)
            total_position_value = 0.0
            total_position_cost = 0.0  # 진입금
            total_unrealized_pnl = 0.0
            
            for symbol, pos_data in positions.items():
                qty = pos_data['quantity']
                cost = pos_data['total_cost']
                total_position_cost += cost
                
                if qty > 0:
                    # 현재 종가 찾기 (우선순위: 당일 > 이전일 > 마지막 알려진 가격 > 평균단가)
                    current_close_price = None
                    price_source = "unknown"
                    
                    # 1. 당일 종가 찾기
                    date_found = False
                    for price_date, price_data in daily_close_prices.items():
                        if price_date.date() == date.date() and symbol in price_data:
                            current_close_price = price_data[symbol]
                            price_source = "current_day"
                            date_found = True

                            break
                    
                    if not date_found:
                        # 2. 이전 일자의 종가 찾기 (최대 7일까지)
                        for j in range(1, min(8, len(dates))):
                            prev_date = date - timedelta(days=j)
                            if prev_date in daily_close_prices and symbol in daily_close_prices[prev_date]:
                                current_close_price = daily_close_prices[prev_date][symbol]
                                price_source = f"prev_day_{j}"
                                break
                        
                        # 3. 마지막 알려진 가격 사용
                        if current_close_price is None and symbol in latest_prices:
                            current_close_price = latest_prices[symbol]
                            price_source = "latest_known"
                    
                    if current_close_price:
                        # 현재가 기준 포지션 가치
                        current_position_value = qty * current_close_price
                        total_position_value += current_position_value
                        
                        # Unrealized PnL 계산
                        unrealized_pnl = current_position_value - cost
                        total_unrealized_pnl += unrealized_pnl
                        

                    else:
                        # 종가를 찾을 수 없으면 평균 단가 사용 (unrealized PnL = 0)
                        avg_price = cost / qty if qty > 0 else 0
                        position_value = qty * avg_price
                        total_position_value += position_value
                        

            
            # 포트폴리오 평가금 계산 = 현금 + 포지션 현재가치 
            # = 현금 + 포지션진입금 + Unrealized PnL
            total_equity = cash + total_position_value
            realized_pnl = total_equity - config.initial_cash - total_unrealized_pnl
            total_pnl = total_equity - config.initial_cash
            
            # 시계열 데이터 추가
            timestamps.append(date)
            equity_values.append(total_equity)
            cash_values.append(cash)
            position_costs.append(total_position_cost)
            unrealized_pnls.append(total_unrealized_pnl)
            pnl_values.append(total_pnl)
        
        # Polars DataFrame 생성 (데이터 타입 안정성을 위해 strict=False 설정)
        try:
            # 모든 숫자 값을 float로 통일하여 타입 충돌 방지
            equity_values_float = [float(v) for v in equity_values]
            cash_values_float = [float(v) for v in cash_values]
            position_costs_float = [float(v) for v in position_costs]
            unrealized_pnls_float = [float(v) for v in unrealized_pnls]
            pnl_values_float = [float(v) for v in pnl_values]
            
            equity_df = pl.DataFrame({
                "timestamp": timestamps,
                "equity": equity_values_float,              # 총 평가금액 (현금 + 포지션 현재가치)
                "cash": cash_values_float,                  # 현금
                "position_cost": position_costs_float,      # 포지션 진입금
                "unrealized_pnl": unrealized_pnls_float,    # 미실현 손익
                "total_pnl": pnl_values_float              # 총 손익
            }, strict=False)
            
            return equity_df
            
        except Exception as e:
            traceback.print_exc()
            return None
    
    def _create_daily_returns_polars(self, equity_curve: pl.DataFrame) -> Optional[pl.DataFrame]:
        """일간 수익률 계산 (일자별 종가 기준) - 이미 계산된 equity curve 활용"""
        if equity_curve is None or len(equity_curve) < 2:
            return None
        
        try:
            # Polars를 사용한 수익률 계산 (최적화된 벡터 연산)
            daily_returns = equity_curve.with_columns(
                pl.col("equity").pct_change().alias("return")
            ).drop_nulls().select(["timestamp", "return"])
            
            return daily_returns
        except Exception as e:
            print(f"일간 수익률 계산 오류: {e}")
            return None
    
    def _calculate_performance_metrics_dict_optimized(self, config: BacktestConfig, 
                                                    trade_objects: List[Trade],
                                                    portfolio_equity_history: Dict[datetime, float]) -> Dict[str, float]:
        """실시간 포트폴리오 평가금 데이터를 활용한 최적화된 성과 지표 계산"""
        
        # 기본 수익률 계산
        initial_cash = config.initial_cash
        final_equity = self.broker.get_portfolio().equity
        total_return = (final_equity - initial_cash) / initial_cash
        
        # 연간 수익률 (정확한 기간 계산)
        duration_seconds = (config.end_date - config.start_date).total_seconds()
        duration_days = duration_seconds / (24 * 3600)
        duration_years = duration_days / 365.25
        
        if duration_years > 0 and total_return > -1:
            annual_return = ((final_equity / initial_cash) ** (1 / duration_years) - 1)
        else:
            annual_return = 0.0
        
        # 이미 계산된 실시간 포트폴리오 평가금으로 변동성 및 MDD 계산
        if portfolio_equity_history and len(portfolio_equity_history) > 1:
            equity_values = list(portfolio_equity_history.values())
            
            # 변동성 계산 (시계열 기반)
            returns = []
            for i in range(1, len(equity_values)):
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(ret)
            
            if returns:
                volatility = np.std(returns) * np.sqrt(365.25)
            else:
                volatility = 0.15
            
            # 최대 드로다운 계산 (최적화된 numpy 연산)
            equity_array = np.array(equity_values)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.05
        else:
            # fallback: 기본값 사용
            volatility = 0.15
            max_drawdown = 0.05
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    
    def _create_result_from_dict(self, config: BacktestConfig, start_time: datetime, 
                                     end_time: datetime, trades: List[Dict[str, Any]]) -> BacktestResult:
        """Dict 기반 백테스팅 결과 생성 - 표준 BacktestResult 사용
        
        Args:
            config: 백테스팅 설정
            start_time: 시작 시간
            
            
            end_time: 종료 시간
            trades: 체결된 거래 리스트 (Dict 형태)
            
        Returns:
            표준 BacktestResult 객체
        """
        # 1. Dict → Trade 객체 변환
        trade_objects = self._convert_dict_trades_to_objects(trades)
        
        # 2. 거래 통계 계산
        trade_stats = self._calculate_trade_statistics_dict(trade_objects)
        
        # 3. 성과 지표 계산
        performance_metrics = self._calculate_performance_metrics_dict_optimized(config, trade_objects, self._portfolio_equity_history)
        
        # 4. 시각화 데이터 생성 (선택적) - Phase 8 완전한 시각화 데이터 지원
        equity_curve = None
        daily_returns = None
        monthly_returns = None
        drawdown_periods = None
        trade_signals = None
        benchmark_equity_curve = None
        benchmark_returns = None
        
        if config.save_portfolio_history:
            # 기본 시각화 데이터
            equity_curve = self._create_equity_curve_polars(config, trade_objects)
            if equity_curve is not None:
                daily_returns = self._create_daily_returns_polars(equity_curve)
                
                # === Phase 8 추가: 완전한 시각화 데이터 생성 ===
                
                # 1. 월별 수익률 계산
                monthly_returns = self._calculate_monthly_returns(daily_returns)
                
                # 2. 드로다운 기간 계산
                drawdown_periods = self._calculate_drawdown_periods(equity_curve)
                
                # 3. 거래 신호 데이터 생성
                trade_signals = self._create_trade_signals(trade_objects)
                
                # 4. 벤치마크 데이터 생성 (Buy & Hold)
                benchmark_equity_curve = self._create_benchmark_data(config, equity_curve)
                if benchmark_equity_curve is not None:
                    benchmark_returns = self._create_daily_returns_polars(benchmark_equity_curve)
        
        # 5. 최종 포트폴리오 정보
        final_portfolio = self.broker.get_portfolio()
        
        # 6. 간단한 결과 출력
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t['side'] == 'BUY'])
        sell_trades = len([t for t in trades if t['side'] == 'SELL'])
        
        # 슬리피지 통계
        if trades:
            avg_slippage = sum(abs(t.get('slippage', 0)) for t in trades) / len(trades)
            total_slippage_cost = sum(t.get('slippage_amount', 0) for t in trades)
        else:
            avg_slippage = 0.0
            total_slippage_cost = 0.0
        
        initial_equity = config.initial_cash
        final_equity = final_portfolio.equity
        total_return_amount = final_equity - initial_equity
        
        # 7. 표준 BacktestResult 반환
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
            
            # 거래 및 시각화 데이터
            trades=trade_objects if config.save_portfolio_history else None,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            
            # Phase 8 추가: 완전한 시각화 데이터
            monthly_returns=monthly_returns,
            drawdown_periods=drawdown_periods,
            trade_signals=trade_signals,
            benchmark_equity_curve=benchmark_equity_curve,
            benchmark_returns=benchmark_returns,
            
            # 추가 메타데이터
            metadata={
                'dict_native_engine': True,
                'avg_slippage': avg_slippage,
                'total_slippage_cost': total_slippage_cost,
                'processing_speed': len(trades) / (end_time - start_time).total_seconds() if (end_time - start_time).total_seconds() > 0 else 0
            }
        )
    
    def _execute_backtest(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            백테스팅 결과
        """
        if not self.strategy:
            raise ValueError("전략이 설정되지 않았습니다")
        if not self.broker:
            raise ValueError("브로커가 설정되지 않았습니다")
        if not self.data_provider:
            raise ValueError("데이터 제공자가 설정되지 않았습니다")
        
        total_start_time = datetime.now()

        # 0단계: 브로커 초기화 (중요!)
        self.broker.portfolio.cash = config.initial_cash
        self.broker.portfolio.positions = {}
        self.broker.orders = {}
        self.broker.trades = []


        # 1단계: 원본 데이터 로딩 (시간 측정 별도)
        data_load_start = time.time()
        raw_data_df = self._load_raw_data_as_polars(config)
        data_load_time = time.time() - data_load_start
        
        # 2단계: 지표 계산 (시간 측정)
        indicator_start = time.time()
        if isinstance(self.strategy, TradingStrategy):
            # Polars DataFrame으로 직접 지표 계산 (변환 과정 제거!)
            enriched_df = self.strategy.precompute_indicators(raw_data_df)
            
            # 백테스팅 루프용으로만 List[Dict] 변환
            enriched_data = enriched_df.to_dicts()
        else:
            # 기존 DictTradingStrategy의 경우만 Dict 변환
            raw_data_dict = raw_data_df.to_dicts()
            enriched_data = self.strategy.precompute_indicators_dict(raw_data_dict)
        indicator_time = time.time() - indicator_start
        
        # 3단계: 백테스팅 루프 실행 (시간 측정)
        backtest_start = time.time()
        trades = self._run_dict_native_backtest_loop(config, enriched_data)
        backtest_time = time.time() - backtest_start
        
        # 4단계: 결과 생성
        result_start = time.time()
        end_time = datetime.now()
        result = self._create_result_from_dict(config, total_start_time, end_time, trades)
        result_time = time.time() - result_start
        
        # 총 백테스팅 로직 시간 (데이터 로딩 제외)
        pure_backtest_time = indicator_time + backtest_time + result_time
        total_time = (end_time - total_start_time).total_seconds()
        
        # 결과에 시간 정보 추가
        if hasattr(result, 'metadata') and result.metadata:
            result.metadata.update({
                'data_load_time': data_load_time,
                'indicator_time': indicator_time, 
                'backtest_time': backtest_time,
                'result_time': result_time,
                'pure_backtest_time': pure_backtest_time,
                'total_time': total_time,
                'processing_speed': len(enriched_data)/pure_backtest_time if pure_backtest_time > 0 else 0
            })
        
        return result
    
    
    def cleanup(self):
        """백테스트 엔진 정리 - aiohttp 세션 등 리소스 정리"""
        if self.data_provider and hasattr(self.data_provider, '_session'):
            if self.data_provider._session and not self.data_provider._session.closed:
                asyncio.run(self.data_provider._session.close())
                self.data_provider._session = None
        
        # 데이터 캐시 정리 (메모리 절약)
        self._cached_market_data = None
        self._cached_daily_market_data = None
    
    def run(self, config: BacktestConfig) -> BacktestResult:
        """백테스팅 실행 - 공개 인터페이스
        
        Args:
            config: 백테스팅 설정
            
        Returns:
            백테스팅 결과
        """
        try:
            return self._execute_backtest(config)
        finally:
            # 백테스팅 완료 후 리소스 정리
            self.cleanup()