"""
병렬 백테스팅 엔진

그리드 서치를 위한 병렬 백테스팅 엔진입니다.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import traceback
import pickle
import os

from ...core.value_objects.grid_search_config import GridSearchConfig
from ...core.value_objects.grid_search_result import GridSearchResult, GridSearchSummary
from ...core.value_objects.backtest_result import BacktestResult
from ...core.interfaces.data_provider import IDataProvider
from ...core.interfaces.broker import IBroker
from ...core.interfaces.strategy import TradingStrategy
from .simple_engine import SimpleBacktestEngine


class ParallelBacktestEngine:
    """병렬 백테스팅 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._data_provider: Optional[IDataProvider] = None
        self._broker_class: Optional[type] = None
        self._broker_config: Optional[Dict[str, Any]] = None
    
    def set_data_provider(self, data_provider: IDataProvider) -> None:
        """데이터 프로바이더 설정"""
        self._data_provider = data_provider
    
    def set_broker(self, broker: IBroker) -> None:
        """브로커 설정 (클래스와 설정을 분리하여 저장)"""
        self._broker_class = type(broker)
        self._broker_config = {
            'initial_cash': broker.portfolio.cash,
            'commission_rate': getattr(broker, 'commission_rate', 0.001),
            'slippage_rate': getattr(broker, 'slippage_rate', 0.001)
        }
    
    async def run_grid_search_threaded(self, config: GridSearchConfig) -> GridSearchResult:
        """스레드 기반 그리드 서치 실행"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"스레드 기반 그리드 서치 시작: {config.valid_combinations}개 조합")
            
            # 파라미터 조합들 가져오기
            all_combinations = config.parameter_combinations
            
            # 스레드 수 결정
            max_workers = config.max_workers or min(mp.cpu_count(), len(all_combinations))
            
            # 결과 수집
            all_summaries = []
            total_executed = 0
            successful_runs = 0
            failed_runs = 0
            
            # ThreadPoolExecutor로 병렬 실행
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 작업 제출
                future_to_params = {
                    executor.submit(
                        _run_single_backtest_threaded,
                        params,
                        config
                    ): params 
                    for params in all_combinations
                }
                
                # 결과 수집
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    total_executed += 1
                    
                    try:
                        summary = future.result()
                        if summary:
                            all_summaries.append(summary)
                            successful_runs += 1
                            self.logger.info(f"백테스트 완료: {params}")
                        else:
                            failed_runs += 1
                    except Exception as e:
                        failed_runs += 1
                        self.logger.error(f"백테스트 실패 - {params}: {e}")
            
            # 최적 결과 찾기
            if not all_summaries:
                raise ValueError("성공한 백테스트가 없습니다.")
            
            best_summary = max(all_summaries, key=lambda x: x.calmar_ratio)
            best_params = best_summary.params
            
            # 상위 결과들 선별 (필요시 상세 결과 포함)
            top_results = None
            if config.save_detailed_results:
                # TODO: 상위 N개에 대해 상세 백테스트 재실행
                pass
            
            end_time = datetime.now()
            
            result = GridSearchResult(
                config=config,
                start_time=start_time,
                end_time=end_time,
                summaries=all_summaries,
                best_params=best_params,
                best_summary=best_summary,
                top_results=top_results,
                total_executed=total_executed,
                successful_runs=successful_runs,
                failed_runs=failed_runs
            )
            
            self.logger.info(f"스레드 기반 그리드 서치 완료: {result.duration:.1f}초, "
                           f"성공률: {result.success_rate:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"스레드 기반 그리드 서치 실행 오류: {e}")
            self.logger.error(traceback.format_exc())
            raise

    async def run_grid_search(self, config: GridSearchConfig) -> GridSearchResult:
        """그리드 서치 실행"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"그리드 서치 시작: {config.valid_combinations}개 조합")
            
            # 파라미터 조합들 가져오기
            all_combinations = config.parameter_combinations
            
            # CPU 코어 수 결정
            max_workers = config.max_workers or min(mp.cpu_count(), len(all_combinations))
            
            # 배치별로 실행
            all_summaries = []
            total_executed = 0
            successful_runs = 0
            failed_runs = 0
            
            # 데이터를 미리 로드하여 공유
            shared_data = await self._prepare_shared_data(config.base_config)
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 배치별로 작업 제출
                futures = []
                for batch_idx in range(config.total_batches):
                    batch_combinations = config.get_batch_combinations(batch_idx)
                    if not batch_combinations:
                        continue
                    
                    future = executor.submit(
                        _run_batch_backtest,
                        batch_combinations,
                        config,
                        shared_data,
                        self._broker_class,
                        self._broker_config
                    )
                    futures.append(future)
                
                # 결과 수집
                for future in as_completed(futures):
                    try:
                        batch_summaries, batch_stats = future.result()
                        all_summaries.extend(batch_summaries)
                        total_executed += batch_stats['total']
                        successful_runs += batch_stats['success']
                        failed_runs += batch_stats['failed']
                        
                        self.logger.info(f"배치 완료: {len(batch_summaries)}개 결과")
                        
                    except Exception as e:
                        self.logger.error(f"배치 실행 오류: {e}")
                        failed_runs += 1
            
            # 최적 결과 찾기
            if not all_summaries:
                raise ValueError("성공한 백테스트가 없습니다.")
            
            best_summary = max(all_summaries, key=lambda x: x.calmar_ratio)
            best_params = best_summary.params
            
            # 상위 결과들 선별 (필요시 상세 결과 포함)
            top_results = None
            if config.save_detailed_results:
                # TODO: 상위 N개에 대해 상세 백테스트 재실행
                pass
            
            end_time = datetime.now()
            
            result = GridSearchResult(
                config=config,
                start_time=start_time,
                end_time=end_time,
                summaries=all_summaries,
                best_params=best_params,
                best_summary=best_summary,
                top_results=top_results,
                total_executed=total_executed,
                successful_runs=successful_runs,
                failed_runs=failed_runs
            )
            
            self.logger.info(f"그리드 서치 완료: {result.duration:.1f}초, "
                           f"성공률: {result.success_rate:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"그리드 서치 실행 오류: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def _prepare_shared_data(self, config) -> Dict[str, Any]:
        """공유 데이터 준비 (시장 데이터 등)"""
        try:
            # 시장 데이터 로드
            market_data = await self._data_provider.get_data(
                symbols=config.symbols,
                start=config.start_date,
                end=config.end_date,
                timeframe=config.timeframe
            )
            
            # 직렬화 가능한 형태로 변환
            shared_data = {
                'market_data': market_data,
                'config': config
            }
            
            return shared_data
            
        except Exception as e:
            self.logger.error(f"공유 데이터 준비 오류: {e}")
            raise


def _run_batch_backtest(
    combinations: List[Dict[str, Any]],
    config: GridSearchConfig,
    shared_data: Dict[str, Any],
    broker_class: type,
    broker_config: Dict[str, Any]
) -> Tuple[List[GridSearchSummary], Dict[str, int]]:
    """배치 백테스트 실행 (워커 함수)"""
    
    summaries = []
    stats = {'total': 0, 'success': 0, 'failed': 0}
    
    for params in combinations:
        stats['total'] += 1
        
        try:
            # 전략 인스턴스 생성
            strategy = config.strategy_class(**params)
            
            # 브로커 인스턴스 생성
            broker = broker_class(**broker_config)
            
            # 시장 데이터 설정 (공유 데이터 사용)
            market_data = shared_data['market_data']
            base_config = shared_data['config']
            
            # 백테스트 엔진 생성 및 실행
            engine = SimpleBacktestEngine()
            engine.set_strategy(strategy)
            engine.set_broker(broker)
            
            # 워커 프로세스에서는 간단한 동기 실행 사용
            result = _run_simple_sync_backtest(strategy, broker, base_config, market_data)
            
            # 최소 거래 횟수 필터링
            if result.total_trades < config.min_trades:
                continue
            
            # 요약 결과 생성
            summary = GridSearchSummary(
                params=params,
                calmar_ratio=result._calculate_calmar_ratio(),
                annual_return=result.annual_return,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_return=result.total_return,
                volatility=result.volatility,
                final_equity=result.final_equity
            )
            
            summaries.append(summary)
            stats['success'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            # 디버깅을 위해 오류 정보 저장
            print(f"백테스트 실패 - 파라미터: {params}, 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return summaries, stats


def _run_single_backtest_with_preloaded_data(engine, config, market_data) -> BacktestResult:
    """미리 로드된 데이터를 사용한 단일 백테스트 실행"""
    import asyncio
    from ...core.value_objects.backtest_config import BacktestConfig
    from ...core.interfaces.strategy import BacktestContext
    from datetime import datetime
    
    try:
        # 새로운 이벤트 루프 생성 (워커 프로세스용)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # BacktestConfig 객체 생성
        backtest_config = BacktestConfig(
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            initial_cash=config.initial_cash,
            save_portfolio_history=False  # 그리드 서치에서는 성능 최적화를 위해 비활성화
        )
        
        # 미리 로드된 데이터를 엔진에 설정
        engine._market_data = {}
        for symbol in backtest_config.symbols:
            import polars as pl
            symbol_data = market_data.filter(pl.col("symbol") == symbol)
            engine._market_data[symbol] = symbol_data
        
        # 직접 백테스트 실행 (data_provider 우회)
        result = loop.run_until_complete(
            _execute_backtest_with_preloaded_data(engine, backtest_config, market_data)
        )
        
        return result
        
    finally:
        loop.close()


async def _execute_backtest_with_preloaded_data(engine, config, market_data):
    """미리 로드된 데이터로 백테스트 실행 (data_provider 우회)"""
    start_time = datetime.now()
    
    # 컨텍스트 초기화
    engine.context = BacktestContext(
        initial_cash=config.initial_cash,
        symbols=config.symbols
    )
    
    # 브로커 초기화
    engine.broker.portfolio.cash = config.initial_cash
    
    # 지표 사전 계산 (전략 메서드 사용)
    enriched_data = engine.strategy.precompute_indicators(market_data)
    
    # 전략 초기화
    engine.strategy.initialize(engine.context)
    # 브로커 연결
    engine.strategy.set_broker(engine.broker)
    
    # 백테스트 실행
    trades = await engine._run_backtest_loop_with_enriched_data(config, enriched_data)
    
    # 결과 생성
    result = await engine._create_result(config, start_time, datetime.now(), trades)
    
    # 전략 종료 처리
    engine.strategy.finalize(engine.context)
    
    return result


def _run_single_backtest_threaded(params, config):
    """스레드 기반 단일 백테스트 실행"""
    try:
        # 새로운 이벤트 루프 생성 (스레드용)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 전략 생성
            strategy = config.strategy_class(**params)
            
            # 브로커 생성
            from ...infrastructure.brokers.simple_broker import SimpleBroker
            broker = SimpleBroker(
                initial_cash=config.base_config.initial_cash,
                commission_rate=0.001,
                slippage_rate=0.001
            )
            
            # 데이터 프로바이더 생성
            from ...infrastructure.data.upbit_provider import UpbitDataProvider
            data_provider = UpbitDataProvider(cache_dir="./data/upbit_cache")
            
            # 엔진 생성 및 설정
            from ...infrastructure.engine.simple_engine import SimpleBacktestEngine
            engine = SimpleBacktestEngine()
            engine.set_data_provider(data_provider)
            engine.set_strategy(strategy)
            engine.set_broker(broker)
            
            # 백테스트 설정
            from ...core.value_objects.backtest_config import BacktestConfig
            backtest_config = BacktestConfig(
                symbols=config.base_config.symbols,
                start_date=config.base_config.start_date,
                end_date=config.base_config.end_date,
                timeframe=config.base_config.timeframe,
                initial_cash=config.base_config.initial_cash,
                save_portfolio_history=False
            )
            
            # 백테스트 실행
            result = loop.run_until_complete(engine.run(backtest_config))
            
            # 최소 거래 횟수 필터링
            if result.total_trades < config.min_trades:
                return None
            
            # 요약 결과 생성
            from ...core.value_objects.grid_search_result import GridSearchSummary
            summary = GridSearchSummary(
                params=params,
                calmar_ratio=result._calculate_calmar_ratio(),
                annual_return=result.annual_return,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_return=result.total_return,
                volatility=result.volatility,
                final_equity=result.final_equity
            )
            
            return summary
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"스레드 백테스트 실패 - 파라미터: {params}, 오류: {str(e)}")
        return None


def _run_simple_sync_backtest(strategy, broker, config, market_data):
    """간단한 동기 백테스트 실행 (워커 프로세스용)"""
    import asyncio
    from ...infrastructure.data.upbit_provider import UpbitDataProvider
    from ...infrastructure.engine.simple_engine import SimpleBacktestEngine
    from ...core.value_objects.backtest_config import BacktestConfig
    
    try:
        # 새로운 이벤트 루프 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 데이터 프로바이더 생성 (워커 프로세스용)
        data_provider = UpbitDataProvider(cache_dir="./data/upbit_cache")
        
        # 엔진 생성 및 설정
        engine = SimpleBacktestEngine()
        engine.set_data_provider(data_provider)
        engine.set_strategy(strategy)
        engine.set_broker(broker)
        
        # 백테스트 설정
        backtest_config = BacktestConfig(
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            initial_cash=config.initial_cash,
            save_portfolio_history=False
        )
        
        # 백테스트 실행
        result = loop.run_until_complete(engine.run(backtest_config))
        
        return result
        
    finally:
        loop.close()


def _run_single_backtest_sync(engine, config, market_data) -> BacktestResult:
    """단일 백테스트 동기 실행 (레거시)"""
    return _run_single_backtest_with_preloaded_data(engine, config, market_data)


class SharedDataManager:
    """공유 데이터 관리자"""
    
    def __init__(self):
        self._shared_data = None
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """공유 데이터 설정"""
        self._shared_data = data
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """공유 데이터 반환"""
        return self._shared_data


# 전역 공유 데이터 관리자
_shared_manager = SharedDataManager() 