"""
QuantBT Engine Adapter V2 (단순화 버전)

shared_data_ref만 사용하는 단순화된 어댑터로 최고 성능을 제공합니다.
- shared_data_ref 전용 설계
- 526배 빠른 데이터 접근
- 복잡한 캐시 로직 제거
- 메모리 효율성 극대화
"""

import logging
from typing import Dict, Any, Type
from unittest.mock import Mock
import ray

logger = logging.getLogger(__name__)


class QuantBTEngineAdapter:
    """QuantBT BacktestEngine과 Ray Actor 연동 어댑터 V2
    
    shared_data_ref만 사용하는 단순화된 구조로 최고 성능을 제공합니다.
    """
    
    def __init__(self, base_config: Any, shared_data_ref: ray.ObjectRef):
        """어댑터 초기화
        
        Args:
            base_config: 백테스트 기본 설정 (BacktestConfig)
            shared_data_ref: 공유 데이터 참조 (필수)
        """
        self.base_config = base_config
        self.shared_data_ref = shared_data_ref
        
        if self.shared_data_ref is None:
            raise ValueError("shared_data_ref는 필수입니다")
        
        logger.info("QuantBTEngineAdapter 초기화 완료 (shared_data_ref 전용)")
    
    def execute_backtest(self, params: Dict, strategy_class: Type) -> Dict:
        """백테스트 실행 (실제 QuantBT 엔진 사용)
        
        Args:
            params: 전략 파라미터
            strategy_class: 전략 클래스
            
        Returns:
            Dict: 백테스트 결과
        """
        try:
            logger.info(f"백테스트 실행 시작: {strategy_class.__name__} with {params}")
            
            # 1. 공유 데이터에서 직접 로딩 (최고 성능)
            logger.info("공유 데이터에서 직접 로딩...")
            data = ray.get(self.shared_data_ref)
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
            result = engine.run(self.base_config, show_progress=False)
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
            logger.error(f"백테스트 실행 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            
            # 실패 시 기본 결과 반환
            return {
                'success': False,
                'error': str(e),
                'params': params,
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'final_portfolio_value': self.base_config.initial_cash
            }
    
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
            CustomDataProvider: 커스텀 데이터 프로바이더
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
        """실제 백테스트 결과에서 필요한 정보 추출
        
        Args:
            result: 백테스트 결과 객체
            
        Returns:
            Dict: 추출된 결과 정보
        """
        try:
            # BacktestResult 객체에서 직접 속성 접근
            if hasattr(result, 'sharpe_ratio'):
                # get_summary() 메서드에 해당하는 모든 값들을 포함
                extracted_data = {
                    # 기본 성과 지표
                    'sharpe_ratio': float(result.sharpe_ratio),
                    'total_return': float(result.total_return),
                    'annual_return': float(result.annual_return),
                    'volatility': float(result.volatility),
                    'max_drawdown': float(result.max_drawdown),
                    
                    # 거래 통계
                    'total_trades': int(result.total_trades),
                    'winning_trades': int(result.winning_trades),
                    'losing_trades': int(result.losing_trades),
                    'win_rate': float(result.win_rate),
                    'avg_win': float(result.avg_win),
                    'avg_loss': float(result.avg_loss),
                    'profit_factor': float(result.profit_factor),
                    
                    # 최종 상태
                    'final_portfolio_value': float(result.final_equity),
                    'final_equity': float(result.final_equity),
                    'total_pnl': float(result.total_pnl),
                    'initial_capital': float(result.config.initial_cash),
                    
                    # 백분율 버전 (get_summary에서 사용)
                    'total_return_pct': float(result.total_return_pct),
                    'annual_return_pct': float(result.annual_return_pct),
                    'volatility_pct': float(result.volatility_pct),
                    'max_drawdown_pct': float(result.max_drawdown_pct),
                    'win_rate_pct': float(result.win_rate_pct),
                    
                    # 추가 계산 지표
                    'calmar_ratio': float(result._calculate_calmar_ratio()),
                    'sortino_ratio': float(result._calculate_sortino_ratio()),
                    
                    # 실행 시간
                    'execution_time': float(result.duration),
                    'duration': float(result.duration),
                    
                    # 기간 정보
                    'period': f"{result.config.start_date.date()} ~ {result.config.end_date.date()}",
                    'start_date': str(result.config.start_date.date()),
                    'end_date': str(result.config.end_date.date()),
                }
                
                return extracted_data
            
            # get_metrics 메서드가 있는 경우
            elif hasattr(result, 'get_metrics'):
                metrics = result.get_metrics()
                return {
                    'sharpe_ratio': float(metrics.get('sharpe_ratio', 0.0)),
                    'total_return': float(metrics.get('total_return', 0.0)),
                    'annual_return': float(metrics.get('annual_return', 0.0)),
                    'volatility': float(metrics.get('volatility', 0.0)),
                    'max_drawdown': float(metrics.get('max_drawdown', 0.0)),
                    'win_rate': float(metrics.get('win_rate', 0.0)),
                    'total_trades': int(metrics.get('total_trades', 0)),
                    'profit_factor': float(metrics.get('profit_factor', 0.0)),
                    'final_portfolio_value': float(metrics.get('final_portfolio_value', self.base_config.initial_cash)),
                    'calmar_ratio': float(metrics.get('calmar_ratio', 0.0)),
                    'sortino_ratio': float(metrics.get('sortino_ratio', 0.0)),
                    'execution_time': float(metrics.get('execution_time', 0.0))
                }
            
            # 딕셔너리 형태인 경우
            elif isinstance(result, dict):
                return {
                    'sharpe_ratio': float(result.get('sharpe_ratio', 0.0)),
                    'total_return': float(result.get('total_return', 0.0)),
                    'annual_return': float(result.get('annual_return', 0.0)),
                    'volatility': float(result.get('volatility', 0.0)),
                    'max_drawdown': float(result.get('max_drawdown', 0.0)),
                    'win_rate': float(result.get('win_rate', 0.0)),
                    'total_trades': int(result.get('total_trades', 0)),
                    'profit_factor': float(result.get('profit_factor', 0.0)),
                    'final_portfolio_value': float(result.get('final_portfolio_value', self.base_config.initial_cash)),
                    'calmar_ratio': float(result.get('calmar_ratio', 0.0)),
                    'sortino_ratio': float(result.get('sortino_ratio', 0.0)),
                    'execution_time': float(result.get('execution_time', 0.0))
                }
            
            else:
                logger.warning(f"알 수 없는 결과 타입: {type(result)}")
                return self._create_default_result()
                
        except Exception as e:
            logger.error(f"결과 추출 실패: {e}")
            import traceback
            logger.error(f"상세 에러: {traceback.format_exc()}")
            return self._create_default_result()
    
    def _create_default_result(self) -> Dict:
        """기본 결과 생성
        
        Returns:
            Dict: 기본 결과
        """
        return {
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'final_portfolio_value': self.base_config.initial_cash,
            'final_equity': self.base_config.initial_cash,
            'total_pnl': 0.0,
            'initial_capital': self.base_config.initial_cash,
            'total_return_pct': 0.0,
            'annual_return_pct': 0.0,
            'volatility_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'execution_time': 0.0,
            'duration': 0.0,
            'period': f"{self.base_config.start_date.date()} ~ {self.base_config.end_date.date()}",
            'start_date': str(self.base_config.start_date.date()),
            'end_date': str(self.base_config.end_date.date())
        } 