"""
QuantBT Engine Adapter

QuantBT BacktestEngine과 Ray Actor를 연동하는 어댑터
기존 QuantBT 시스템을 Ray 기반 분산 환경에서 실행할 수 있도록 지원
"""

import logging
from typing import Dict, Any, Type, Optional, List
from unittest.mock import Mock
import ray

logger = logging.getLogger(__name__)


class QuantBTEngineAdapter:
    """QuantBT BacktestEngine과 Ray Actor 연동 어댑터
    
    기존 QuantBT 백테스트 엔진을 Ray 환경에서 실행할 수 있도록
    인터페이스를 제공하는 어댑터 클래스입니다.
    """
    
    def __init__(self, base_config: Any, data_manager_ref: Optional[ray.ObjectRef] = None, shared_data_ref: Optional[ray.ObjectRef] = None):
        """어댑터 초기화
        
        Args:
            base_config: 백테스트 기본 설정 (BacktestConfig)
            data_manager_ref: RayDataManager 참조 (선택사항)
            shared_data_ref: 공유 데이터 참조 (선택사항)
        """
        self.base_config = base_config
        self.data_manager_ref = data_manager_ref
        self.shared_data_ref = shared_data_ref
        
        # 데이터 소스 결정
        if self.shared_data_ref is not None:
            logger.info("공유 데이터 참조 모드로 초기화")
        elif self.data_manager_ref is not None:
            logger.info("RayDataManager 통합 모드로 초기화")
        else:
            # 기존 로컬 캐싱 모드
            self._data_cache = {}
            logger.info("로컬 캐싱 모드로 초기화")
        
        logger.info("QuantBTEngineAdapter 초기화 완료")
    
    def create_backtest_task(self, params: Dict, strategy_class: Type) -> Dict:
        """백테스트 작업 생성
        
        Args:
            params: 전략 파라메터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 작업 정의
        """
        return {
            'strategy_class': strategy_class,
            'params': params,
            'config': self.base_config,
            'task_id': f"{strategy_class.__name__}_{hash(str(params))}"
        }
    
    def execute_backtest(self, params: Dict, strategy_class: Type) -> Dict:
        """백테스트 실행 (Ray Actor에서 호출)
        
        Args:
            params: 전략 파라메터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 결과
        """
        try:
            logger.info(f"백테스트 실행 시작: {strategy_class.__name__} with {params}")
            
            # 1. 실제 데이터 로딩
            logger.info("데이터 로딩 중...")
            data = self._load_real_data()
            logger.info(f"데이터 로딩 완료: {len(data) if hasattr(data, '__len__') else 'Unknown'} 행")
            
            # 2. 실제 백테스트 엔진 생성
            logger.info("백테스트 엔진 생성 중...")
            engine = self._create_real_backtest_engine(data)
            
            # 3. 전략 인스턴스 생성 및 설정
            logger.info("전략 인스턴스 생성 중...")
            strategy = strategy_class(**params)
            engine.set_strategy(strategy)
            
            # 4. 실제 백테스트 실행
            logger.info("백테스트 실행 중...")
            result = engine.run(self.base_config)
            logger.info(f"백테스트 완료: {type(result)}")
            
            # 백테스트 결과 상세 정보 로그
            if hasattr(result, 'total_trades'):
                logger.info(f"총 거래 수: {result.total_trades}")
                logger.info(f"총 수익률: {result.total_return}")
                logger.info(f"최종 자산: {result.final_equity}")
            
            # 5. 실제 결과 추출
            logger.info("결과 추출 중...")
            extracted_result = self._extract_real_results(result)
            extracted_result['success'] = True
            extracted_result['params'] = params
            
            logger.info(f"백테스트 성공: {extracted_result}")
            return extracted_result
            
        except Exception as e:
            logger.error(f"실제 백테스트 실행 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            return {
                'success': False,
                'params': params,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': -1.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'final_portfolio_value': self.base_config.initial_cash,
                'error': str(e)
            }
    
    def _generate_mock_result(self, params: Dict, strategy: Any) -> Mock:
        """모킹된 백테스트 결과 생성 (테스트용)
        
        Args:
            params: 전략 파라메터
            strategy: 전략 인스턴스
            
        Returns:
            Mock: 모킹된 백테스트 결과
        """
        import random
        
        # 파라메터에 따라 다른 성과 생성 (SMA 전략 특성 반영)
        buy_sma = params.get('buy_sma', 15)
        sell_sma = params.get('sell_sma', 30)
        
        # SMA 차이가 클수록 더 보수적인 전략 (낮은 수익률, 낮은 변동성)
        sma_diff = sell_sma - buy_sma
        
        # 기본 성과 (SMA 차이에 따라 조정)
        base_return = random.uniform(0.05, 0.25) * (1 - sma_diff * 0.01)
        volatility = random.uniform(0.15, 0.35) * (1 - sma_diff * 0.005)
        
        # 샤프 비율 계산 (수익률 / 변동성)
        sharpe_ratio = base_return / volatility if volatility > 0 else 0
        
        # 최대 낙폭 (변동성에 비례)
        max_drawdown = -random.uniform(0.05, 0.20) * volatility
        
        # 승률 (보수적인 전략일수록 높은 승률)
        win_rate = random.uniform(0.45, 0.75) + (sma_diff * 0.01)
        win_rate = min(win_rate, 0.85)  # 최대 85%
        
        # 거래 횟수 (SMA 차이가 클수록 적은 거래)
        total_trades = max(5, int(random.uniform(20, 80) * (1 - sma_diff * 0.02)))
        
        # 최종 포트폴리오 가치
        final_value = self.base_config.initial_cash * (1 + base_return)
        
        # Mock 결과 객체 생성
        mock_result = Mock()
        mock_result.get_metrics.return_value = {
            'total_return': base_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': final_value
        }
        
        return mock_result
    
    def _load_real_data(self) -> Any:
        """실제 업비트 데이터 로딩 (공유 데이터, RayDataManager 또는 로컬)
        
        Returns:
            Any: 로딩된 시장 데이터
        """
        if self.shared_data_ref is not None:
            # 공유 데이터 참조 사용 (가장 효율적)
            logger.info("공유 데이터 참조에서 데이터 로딩")
            return ray.get(self.shared_data_ref)
        elif self.data_manager_ref is not None:
            # RayDataManager 사용
            return self._get_data_from_manager()
        else:
            # 기존 로컬 캐싱 방식
            return self._load_data_locally()
    
    def _get_data_from_manager(self) -> Any:
        """RayDataManager에서 데이터 ObjectRef 가져오기 (개선된 제로카피 방식)"""
        try:
            # 캐시된 데이터 참조 먼저 확인 (제로카피)
            cached_ref = ray.get(self.data_manager_ref.get_cached_data_reference.remote(
                symbols=self.base_config.symbols,
                start_date=self.base_config.start_date,
                end_date=self.base_config.end_date,
                timeframe=self.base_config.timeframe
            ))
            
            if cached_ref is not None:
                # 캐시 히트: 제로카피로 데이터 가져오기
                logger.info(f"RayDataManager 캐시 히트: 제로카피 데이터 접근")
                data = ray.get(cached_ref)
                logger.info(f"RayDataManager에서 캐시된 데이터 로딩 완료: {len(data):,}행")
                return data
            else:
                # 캐시 미스: 새로 로딩
                logger.info(f"RayDataManager 캐시 미스: 새로 데이터 로딩")
                data_ref = ray.get(self.data_manager_ref.load_real_data.remote(
                    symbols=self.base_config.symbols,
                    start_date=self.base_config.start_date,
                    end_date=self.base_config.end_date,
                    timeframe=self.base_config.timeframe
                ))
                
                # ObjectRef에서 실제 데이터 가져오기
                data = ray.get(data_ref)
                logger.info(f"RayDataManager에서 새로 데이터 로딩 완료: {len(data):,}행")
                return data
            
        except Exception as e:
            logger.error(f"RayDataManager 데이터 로딩 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            raise
    
    def _load_data_locally(self) -> Any:
        """로컬 캐싱 방식으로 데이터 로딩 (기존 방식)"""
        cache_key = f"{self.base_config.symbols}_{self.base_config.start_date}_{self.base_config.end_date}"
        
        if cache_key in self._data_cache:
            logger.debug(f"로컬 캐시에서 데이터 반환: {cache_key}")
            return self._data_cache[cache_key]
        
        # 실제 데이터 프로바이더 사용
        from quantbt.infrastructure.data.upbit_provider import UpbitDataProvider
        
        data_provider = UpbitDataProvider()
        import asyncio
        
        # 이벤트 루프가 이미 실행 중인지 확인
        try:
            loop = asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 create_task 사용
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    data_provider.get_data(
                        symbols=self.base_config.symbols,
                        start=self.base_config.start_date,
                        end=self.base_config.end_date,
                        timeframe=self.base_config.timeframe
                    )
                )
                data = future.result()
        except RuntimeError:
            # 실행 중인 루프가 없으면 asyncio.run 사용
            data = asyncio.run(data_provider.get_data(
                symbols=self.base_config.symbols,
                start=self.base_config.start_date,
                end=self.base_config.end_date,
                timeframe=self.base_config.timeframe
            ))
        
        # 로컬 캐싱
        self._data_cache[cache_key] = data
        logger.info(f"데이터 로딩 완료: {len(data)} 행")
        
        return data
    
    def _create_real_backtest_engine(self, data: Any):
        """실제 QuantBT 백테스트 엔진 생성
        
        Args:
            data: 로딩된 시장 데이터
            
        Returns:
            BacktestEngine: 실제 QuantBT 백테스트 엔진
        """
        try:
            from quantbt import BacktestEngine, SimpleBroker
            
            # 백테스트 엔진 생성
            engine = BacktestEngine()
            
            # 미리 로딩된 데이터를 사용하는 커스텀 데이터 프로바이더 생성
            custom_data_provider = self._create_custom_data_provider(data)
            engine.set_data_provider(custom_data_provider)
            
            # 브로커 생성 및 설정
            broker = SimpleBroker(
                commission_rate=self.base_config.commission_rate,
                slippage_rate=self.base_config.slippage_rate
            )
            engine.set_broker(broker)
            
            logger.info("실제 QuantBT 백테스트 엔진 생성 완료")
            logger.info(f"미리 로딩된 데이터 크기: {len(data) if hasattr(data, '__len__') else 'Unknown'}")
            return engine
            
        except Exception as e:
            logger.error(f"실제 백테스트 엔진 생성 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            raise
    
    def _create_custom_data_provider(self, preloaded_data: Any):
        """미리 로딩된 데이터를 사용하는 커스텀 데이터 프로바이더 생성
        
        Args:
            preloaded_data: 미리 로딩된 데이터
            
        Returns:
            커스텀 데이터 프로바이더
        """
        class CustomDataProvider:
            """미리 로딩된 데이터를 반환하는 커스텀 데이터 프로바이더"""
            
            def __init__(self, data):
                self.data = data
                logger.info(f"CustomDataProvider 생성: {len(data) if hasattr(data, '__len__') else 'Unknown'} 행")
            
            async def get_data(self, symbols=None, start=None, end=None, timeframe=None):
                """미리 로딩된 데이터 반환"""
                logger.info("CustomDataProvider: 미리 로딩된 데이터 반환")
                return self.data
                
            def close(self):
                """리소스 정리 (호환성을 위해)"""
                pass
        
        return CustomDataProvider(preloaded_data)
    
    def _extract_real_results(self, result) -> Dict:
        """실제 백테스트 결과 추출 및 표준화
        
        Args:
            result: QuantBT BacktestResult 객체
            
        Returns:
            Dict: 표준화된 결과 딕셔너리
        """
        try:
            return {
                'total_return': float(result.total_return),
                'sharpe_ratio': float(result.sharpe_ratio),
                'max_drawdown': float(result.max_drawdown),
                'win_rate': float(result.win_rate),
                'total_trades': int(result.total_trades),
                'final_portfolio_value': float(result.final_equity),
                'annual_return': float(result.annual_return),
                'volatility': float(result.volatility),
                'profit_factor': float(result.profit_factor),
                'execution_time': float(result.duration)
            }
            
        except Exception as e:
            logger.error(f"실제 결과 추출 실패: {e}")
            # 에러 상세 정보를 로그에 추가
            logger.error(f"result 객체 타입: {type(result)}")
            logger.error(f"result 속성들: {dir(result) if hasattr(result, '__dict__') else 'N/A'}")
            raise
    
    def extract_results(self, backtest_result: Any) -> Dict:
        """백테스트 결과 추출 및 표준화
        
        Args:
            backtest_result: QuantBT 백테스트 결과 객체
            
        Returns:
            Dict: 표준화된 결과 딕셔너리
        """
        try:
            # QuantBT 결과에서 메트릭 추출
            if hasattr(backtest_result, 'get_metrics'):
                metrics = backtest_result.get_metrics()
            else:
                # 직접 딕셔너리인 경우
                metrics = backtest_result
            
            # 표준 형식으로 변환
            return {
                'total_return': metrics.get('total_return', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'total_trades': metrics.get('total_trades', 0),
                'final_portfolio_value': metrics.get('final_portfolio_value', self.base_config.initial_cash)
            }
            
        except Exception as e:
            logger.error(f"결과 추출 실패: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': -1.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'final_portfolio_value': self.base_config.initial_cash,
                'error': str(e)
            }
    
    def validate_strategy_class(self, strategy_class: Type) -> bool:
        """전략 클래스 유효성 검증
        
        Args:
            strategy_class: 검증할 전략 클래스
            
        Returns:
            bool: 유효성 여부
        """
        try:
            # 기본 속성 확인
            required_attrs = ['__init__']
            for attr in required_attrs:
                if not hasattr(strategy_class, attr):
                    logger.error(f"전략 클래스에 {attr} 속성이 없습니다")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"전략 클래스 검증 실패: {e}")
            return False
    
    def validate_parameters(self, params: Dict, strategy_class: Type) -> bool:
        """파라메터 유효성 검증
        
        Args:
            params: 검증할 파라메터
            strategy_class: 전략 클래스
            
        Returns:
            bool: 유효성 여부
        """
        try:
            # 전략 인스턴스 생성 테스트
            strategy_class(**params)
            return True
            
        except Exception as e:
            logger.error(f"파라메터 검증 실패: {e}")
            return False
    
    def get_supported_strategies(self) -> List[str]:
        """지원되는 전략 목록 반환
        
        Returns:
            List[str]: 지원되는 전략 이름 목록
        """
        return [
            'SimpleSMAStrategy',
            'TradingStrategy'  # 기본 전략 클래스
        ]
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """어댑터 정보 반환
        
        Returns:
            Dict: 어댑터 정보
        """
        return {
            'adapter_version': '1.0.0',
            'quantbt_version': 'mock',  # 실제 구현에서는 QuantBT 버전
            'supported_strategies': self.get_supported_strategies(),
            'base_config': {
                'symbols': getattr(self.base_config, 'symbols', []),
                'timeframe': getattr(self.base_config, 'timeframe', '1h'),
                'initial_cash': getattr(self.base_config, 'initial_cash', 10000000)
            }
        } 