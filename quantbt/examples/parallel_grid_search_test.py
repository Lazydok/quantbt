# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path

# 현재 위치에서 프로젝트 루트 찾기
current_dir = Path.cwd()
if 'examples' in str(current_dir):
    # examples 폴더에서 실행하는 경우
    project_root = current_dir.parent.parent
else:
    # 프로젝트 루트에서 실행하는 경우
    project_root = current_dir

# 프로젝트 루트를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 필요한 라이브러리 import
import asyncio
from datetime import datetime
import logging

# quantbt 라이브러리 import
from quantbt import (
    UpbitDataProvider,
    SimpleBroker,
)
from quantbt.core.value_objects.grid_search_config import SMAGridSearchConfig
from quantbt.infrastructure.engine.parallel_backtest_engine import ParallelBacktestEngine
from quantbt.examples.strategies.sma_grid_strategy import SMAGridStrategy

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("✅ 라이브러리 import 완료")


async def main():
    """병렬 그리드 서치 테스트 메인 함수"""
    
    print("🚀 병렬 그리드 서치 백테스팅 시작")
    
    # 1. 그리드 서치 설정
    config = SMAGridSearchConfig.create_sma_config(
        strategy_class=SMAGridStrategy,
        symbols=["KRW-BTC"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),  # 3개월 (빠른 테스트)
        buy_sma_range=[5, 10, 15],      # 3개 값
        sell_sma_range=[10, 20, 30],    # 3개 값
        initial_cash=10_000_000
    )
    
    # 그리드 서치 전용 설정 업데이트
    config = SMAGridSearchConfig(
        base_config=config.base_config,
        strategy_class=config.strategy_class,
        strategy_params=config.strategy_params,
        fixed_params=config.fixed_params,
        max_workers=4,                  # 4개 워커
        batch_size=3,                   # 배치 크기 3
        min_trades=5                    # 최소 5번 거래
    )
    
    print(f"📊 그리드 서치 설정:")
    print(f"   - 총 조합 수: {config.total_combinations}")
    print(f"   - 유효 조합 수: {config.valid_combinations}")
    print(f"   - 배치 수: {config.total_batches}")
    print(f"   - 워커 수: {config.max_workers}")
    
    # 2. 데이터 프로바이더 설정
    upbit_provider = UpbitDataProvider()
    
    # 3. 브로커 설정
    broker = SimpleBroker(
        initial_cash=config.base_config.initial_cash,
        commission_rate=config.base_config.commission_rate,
        slippage_rate=config.base_config.slippage_rate
    )
    
    # 4. 병렬 백테스트 엔진 설정
    parallel_engine = ParallelBacktestEngine()
    parallel_engine.set_data_provider(upbit_provider)
    parallel_engine.set_broker(broker)
    
    try:
        # 5. 그리드 서치 실행
        print("⏳ 그리드 서치 실행 중...")
        result = await parallel_engine.run_grid_search_threaded(config)
        
        # 6. 결과 출력
        print("\n" + "="*60)
        print("                 그리드 서치 결과")
        print("="*60)
        
        result.print_summary(top_n=5)
        
        # 7. 시각화 (선택적)
        # try:
        #     print("\n📈 시각화 생성 중...")
            
        #     # Calmar Ratio 분포 히스토그램
        #     result.plot_distribution("calmar_ratio", bins=10)
            
        #     # 파라미터 히트맵 (buy_sma vs sell_sma)
        #     result.plot_heatmap("buy_sma", "sell_sma", "calmar_ratio")
            
        # except Exception as e:
        #     print(f"⚠️ 시각화 오류 (선택적): {e}")
            
        # 8. 결과 확인
        results_df = result.results_df
        print(f"\n📋 결과 데이터프레임 크기: {results_df.shape}")
        print(f"컬럼: {list(results_df.columns)}")
        
        return result
        
    except Exception as e:
        print(f"❌ 그리드 서치 실행 오류: {e}")
        raise


if __name__ == "__main__":
    # 비동기 실행
    result = asyncio.run(main())
    
    print("\n✅ 병렬 그리드 서치 백테스팅 완료!")
    print(f"🏆 최적 파라미터: {result.best_params}")
    print(f"📊 최적 Calmar Ratio: {result.best_summary.calmar_ratio:.3f}") 