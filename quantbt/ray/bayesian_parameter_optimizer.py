from typing import Type, List, Dict, Any
import asyncio
import ray
import threading
import time
from unittest.mock import AsyncMock
import skopt
import logging
import numpy as np

from quantbt.core.interfaces.strategy import TradingStrategy
from quantbt.core.value_objects.backtest_config import BacktestConfig
from quantbt.ray.optimization.parameter_space import ParameterSpace
from quantbt.ray.optimization.bayesian_optimizer import BayesianOptimizer
from quantbt.ray.optimization.early_stopping import EarlyStopping
from quantbt.ray.backtest_actor import BacktestActor
from quantbt.ray.data_manager import RayDataManager
from quantbt.ray.monitoring.progress_tracker import ProgressTracker
from quantbt.ray.monitoring.simple_monitor import SimpleMonitor

BAR_WIDTH = 40
logger = logging.getLogger(__name__)

class BayesianParameterOptimizer:
    """
    베이지안 최적화를 사용하여 트레이딩 전략의 파라미터를 최적화하는 메인 클래스.

    이 클래스는 최적화 프로세스의 모든 구성 요소를 관리합니다:
    - 최적화할 전략 (TradingStrategy)
    - 탐색할 파라미터 공간 (ParameterSpace)
    - 백테스트 설정 (BacktestConfig)
    - Ray 액터 관리 및 분산 처리
    """

    def __init__(
        self,
        strategy_class: Type[TradingStrategy],
        param_space: ParameterSpace,
        config: BacktestConfig,
        num_actors: int = 4,
        n_initial_points: int = 10,
    ):
        """
        BayesianParameterOptimizer를 초기화합니다.

        Args:
            strategy_class (Type[TradingStrategy]): 최적화할 트레이딩 전략 클래스.
            param_space (ParameterSpace): 파라미터 탐색 공간.
            config (BacktestConfig): 백테스트 설정.
            num_actors (int): 병렬 처리에 사용할 Ray 액터의 수.
            n_initial_points (int): 베이지안 최적화의 초기 랜덤 탐색 횟수.
        """
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.config = config
        self.num_actors = num_actors
        self.n_initial_points = n_initial_points
        self.actors: List[BacktestActor] = []
        self.early_stopper: EarlyStopping = None
        self.stop_display_event = threading.Event()

        self.bayesian_optimizer = BayesianOptimizer(
            space=param_space,
            n_initial_points=n_initial_points,
        )

    def _display_progress(self, progress_tracker: ProgressTracker, monitor: SimpleMonitor, interval: int = 5):
        """별도의 스레드에서 진행 상황을 주기적으로 표시합니다."""
        while not self.stop_display_event.is_set():
            # 화면을 지우고 커서를 맨 위로 이동
            print("\033[2J\033[H", end="")

            print("=============== Bayesian Optimization Progress ===============")
            print(progress_tracker.format_progress(show_bar=True, bar_width=50))
            print("--------------------------------------------------------------")
            print(monitor.format_summary())
            print("==============================================================")

            time.sleep(interval)

    async def optimize(
        self,
        objective_metric: str = 'sharpe_ratio',
        n_iter: int = 100,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        베이지안 최적화를 실행합니다.
        """
        logger.info("🚀 베이지안 최적화 프로세스를 시작합니다.")
        logger.info(f" - 목표 지표: {objective_metric}")
        logger.info(f" - 총 반복 횟수: {n_iter}")
        logger.info(f" - 배치 크기: {self.num_actors}")

        if early_stopping_patience > 0:
            patience_in_batches = max(1, early_stopping_patience // self.num_actors)
            self.early_stopper = EarlyStopping(
                patience=patience_in_batches,
                min_delta=early_stopping_min_delta
            )
            logger.info(f" - 조기 종료 활성화: patience={patience_in_batches} 배치 ({early_stopping_patience} 시도), min_delta={early_stopping_min_delta}")

        progress_tracker = ProgressTracker(total_tasks=n_iter)
        monitor = SimpleMonitor()

        self.stop_display_event.clear()
        display_thread = threading.Thread(
            target=self._display_progress,
            args=(progress_tracker, monitor),
            daemon=True
        )
        display_thread.start()

        if not self.actors:
            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.num_actors,
                    ignore_reinit_error=True,
                    logging_level=logging.INFO,
                    log_to_driver=True
                )
            
            data_manager = RayDataManager.remote()
            shared_data_ref = data_manager.load_real_data.remote(
                symbols=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                timeframe=self.config.timeframe,
            )
            self._initialize_actors(shared_data_ref)

        all_results = []
        progress_tracker.start()
        
        try:
            while progress_tracker.completed_tasks < n_iter:
                batch_size = min(self.num_actors, n_iter - progress_tracker.completed_tasks)
                if batch_size <= 0:
                    break

                params_list = self.bayesian_optimizer.ask(n_points=batch_size)
                logger.info(f"배치 시작 | {batch_size}개 파라미터 생성 (총 {progress_tracker.completed_tasks}/{n_iter} 진행)")

                tasks = []
                for i, params in enumerate(params_list):
                    actor = self.actors[i]
                    if isinstance(actor, AsyncMock):
                        future = actor.execute_backtest(params, self.strategy_class)
                    else:
                        future = actor.execute_backtest.remote(params, self.strategy_class)
                    tasks.append(future)

                batch_results_raw = await asyncio.gather(*tasks)

                scores, scores_for_check = [], []
                for params, result in zip(params_list, batch_results_raw):
                    score = result.get(objective_metric, 0.0)
                    scores.append(-score)
                    scores_for_check.append(score)

                    monitor_payload = {'params': params, **result}
                    monitor.record_result(monitor_payload)

                    # 주요 지표들을 최상위 레벨에 추가하여 접근성 향상
                    trial_result = {
                        'params': params,
                        'result': result,
                        # 주요 성과 지표들을 최상위 레벨에 추가
                        'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                        'calmar_ratio': result.get('calmar_ratio', 0.0),
                        'total_return': result.get('total_return', 0.0),
                        'annual_return': result.get('annual_return', 0.0),
                        'max_drawdown': result.get('max_drawdown', 0.0),
                        'volatility': result.get('volatility', 0.0),
                        'win_rate': result.get('win_rate', 0.0),
                        'profit_factor': result.get('profit_factor', 0.0),
                        'total_trades': result.get('total_trades', 0),
                        'final_equity': result.get('final_equity', 0.0),
                        'success': result.get('success', True)
                    }
                    all_results.append(trial_result)

                progress_tracker.update(completed=len(batch_results_raw))
                
                best_batch_score = max(scores_for_check) if scores_for_check else 0
                avg_batch_score = np.mean(scores_for_check) if scores_for_check else 0
                logger.info(f"배치 완료 | 최고 점수: {best_batch_score:.4f}, 평균 점수: {avg_batch_score:.4f}")

                self.bayesian_optimizer.tell(params_list, scores)

                if self.early_stopper:
                    if self.early_stopper.check(best_batch_score):
                        logger.info(f"조기 종료 조건 충족: {self.early_stopper.patience}번의 배치 동안 성능 개선 없음.")
                        break
        finally:
            self.stop_display_event.set()
            display_thread.join(timeout=2)

            logger.info("🏁 베이지안 최적화 프로세스 종료.")
            print("\033[2J\033[H", end="")
            print("=============== Bayesian Optimization Final Result ===============")
            print(monitor.format_summary())
            print("==================================================================")
        return all_results

    def _initialize_actors(self, shared_data_ref: ray.ObjectRef):
        """
        Ray 백테스트 액터를 초기화합니다.
        """
        logger.info(f"초기화 시작: {self.num_actors}개의 백테스트 액터를 생성합니다.")
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_actors)
            
        self.actors = [BacktestActor.remote(f"actor_{i}", shared_data_ref) for i in range(self.num_actors)]
        
        config_dict = self.config.to_dict()
        init_tasks = [actor.initialize_engine.remote(config_dict) for actor in self.actors]
        results = ray.get(init_tasks)
        
        if all(results):
            logger.info("✅ 모든 액터가 성공적으로 초기화되었습니다.")
        else:
            failed_actors = [i for i, r in enumerate(results) if not r]
            logger.error(f"❌ 다음 액터 초기화에 실패했습니다: {failed_actors}")
            raise RuntimeError("일부 액터를 초기화하지 못했습니다. 자세한 내용은 로그를 확인하세요.") 