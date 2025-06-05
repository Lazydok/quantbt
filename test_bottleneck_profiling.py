"""
실제 백테스팅 병목 프로파일링 테스트

실제 멀티타임프레임 백테스팅에서 병목이 어디서 발생하는지 측정합니다.
"""

import asyncio
import polars as pl
import numpy as np
import time
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from typing import Dict, List

from quantbt import (
    SimpleBacktestEngine,
    CSVDataProvider,
    SimpleBroker,
    BacktestConfig,
    MultiTimeframeSMAStrategy
)


class BottleneckProfiler:
    """병목 지점 프로파일링을 위한 클래스"""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.phase_times = {}
        self.current_phase = None
        self.timed_out = False
        
    def start_profiling(self):
        """프로파일링 시작"""
        self.start_time = time.time()
        print(f"🔍 병목 프로파일링 시작 (타임아웃: {self.timeout_seconds}초)")
        
        # 타임아웃 시그널 설정
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout_seconds)
        
    def _timeout_handler(self, signum, frame):
        """타임아웃 핸들러"""
        self.timed_out = True
        print(f"\n⏰ 타임아웃! {self.timeout_seconds}초 경과로 프로파일링 중단")
        self.print_bottleneck_report()
        sys.exit(0)
        
    def start_phase(self, phase_name: str):
        """새 단계 시작"""
        if self.current_phase:
            self.end_phase()
            
        self.current_phase = phase_name
        self.phase_times[phase_name] = {
            'start': time.time(),
            'end': None,
            'duration': None
        }
        elapsed = time.time() - self.start_time
        print(f"📋 [{elapsed:6.2f}s] {phase_name} 시작...")
        
    def end_phase(self):
        """현재 단계 종료"""
        if self.current_phase and self.current_phase in self.phase_times:
            self.phase_times[self.current_phase]['end'] = time.time()
            self.phase_times[self.current_phase]['duration'] = (
                self.phase_times[self.current_phase]['end'] - 
                self.phase_times[self.current_phase]['start']
            )
            elapsed = time.time() - self.start_time
            duration = self.phase_times[self.current_phase]['duration']
            print(f"✅ [{elapsed:6.2f}s] {self.current_phase} 완료 (소요: {duration:.2f}초)")
            
    def print_bottleneck_report(self):
        """병목 보고서 출력"""
        print("\n" + "="*80)
        print("🔍 백테스팅 병목 분석 보고서")
        print("="*80)
        
        if self.current_phase:
            self.end_phase()
            
        total_elapsed = time.time() - self.start_time
        
        # 단계별 시간 정렬
        sorted_phases = sorted(
            [(name, data['duration'] or 0) for name, data in self.phase_times.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n⏱️  총 실행 시간: {total_elapsed:.2f}초")
        print(f"📊 단계별 소요 시간:")
        print("-" * 60)
        
        for phase_name, duration in sorted_phases:
            percentage = (duration / total_elapsed) * 100 if total_elapsed > 0 else 0
            bar_length = int(percentage / 2)  # 50% = 25 chars
            bar = "█" * bar_length + "░" * (25 - bar_length)
            print(f"{phase_name:<30} {duration:6.2f}s ({percentage:5.1f}%) {bar}")
            
        # 주요 병목 지점 식별
        if sorted_phases:
            bottleneck = sorted_phases[0]
            print(f"\n🚨 주요 병목: {bottleneck[0]} ({bottleneck[1]:.2f}초, {(bottleneck[1]/total_elapsed)*100:.1f}%)")


def generate_sample_data(temp_dir: Path, num_days: int = 3) -> None:
    """샘플 데이터 생성"""
    symbols = ["BTC", "ETH", "ADA"]
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=num_days)
    
    for symbol in symbols:
        # 1분봉 데이터 생성
        minutes = int((end_date - start_date).total_seconds() / 60)
        timestamps = [start_date + timedelta(minutes=i) for i in range(minutes)]
        
        np.random.seed(42 + hash(symbol) % 1000)
        base_price = {"BTC": 45000, "ETH": 2500, "ADA": 0.5}[symbol]
        
        data = []
        current_price = base_price
        
        for timestamp in timestamps:
            change = np.random.normal(0, 0.001)
            current_price *= (1 + change)
            
            high = current_price * (1 + abs(np.random.normal(0, 0.0005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.0005)))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "open": round(current_price, 6),
                "high": round(high, 6),
                "low": round(low, 6),
                "close": round(current_price, 6),
                "volume": round(volume, 2)
            })
        
        # CSV 파일 저장
        symbol_df = pl.DataFrame(data)
        symbol_csv_path = temp_dir / f"{symbol}.csv"
        symbol_df.write_csv(symbol_csv_path)
    
    print(f"📊 샘플 데이터 생성 완료: {num_days}일간 3개 심볼, 총 {len(data)*3:,}개 레코드")


async def profile_multi_timeframe_backtest():
    """멀티타임프레임 백테스팅 프로파일링"""
    profiler = BottleneckProfiler(timeout_seconds=60)  # 60초 타임아웃
    
    try:
        profiler.start_profiling()
        
        # 1. 데이터 준비
        profiler.start_phase("1. 샘플 데이터 생성")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            generate_sample_data(temp_path, num_days=5)  # 5일 데이터
            profiler.end_phase()
            
            # 2. 컴포넌트 초기화
            profiler.start_phase("2. 컴포넌트 초기화")
            data_provider = CSVDataProvider(str(temp_path))
            broker = SimpleBroker(initial_cash=100000.0)
            
            strategy = MultiTimeframeSMAStrategy(
                name="ProfileTestStrategy",
                timeframes=["5m", "1h"],
                config={
                    "hourly_short_period": 10,
                    "hourly_long_period": 20,
                    "signal_short_period": 5,
                    "signal_long_period": 10,
                },
                position_size_pct=0.8
            )
            
            engine = SimpleBacktestEngine()
            engine.set_strategy(strategy)
            engine.set_data_provider(data_provider)
            engine.set_broker(broker)
            profiler.end_phase()
            
            # 3. 데이터 로딩
            profiler.start_phase("3. 원본 데이터 로딩")
            config = BacktestConfig(
                symbols=["BTC", "ETH", "ADA"],
                start_date=datetime(2024, 1, 2),
                end_date=datetime(2024, 1, 5),
                initial_cash=100000.0,
                timeframe="1m",
                save_portfolio_history=False
            )
            
            # 실제 데이터 로딩 시뮬레이션
            raw_data = await engine._load_raw_data(config)
            print(f"   로딩된 데이터: {raw_data.height:,}개 레코드")
            profiler.end_phase()
            
            # 4. 지표 계산
            profiler.start_phase("4. 지표 계산 및 데이터 풍부화")
            enriched_data = strategy.precompute_indicators(raw_data)
            print(f"   풍부화된 데이터: {enriched_data.height:,}개 레코드")
            profiler.end_phase()
            
            # 5. 멀티타임프레임 데이터 생성
            profiler.start_phase("5. 멀티타임프레임 데이터 생성")
            from quantbt.core.utils.timeframe import TimeframeUtils
            multi_data = TimeframeUtils.create_multi_timeframe_data(
                enriched_data, strategy.timeframes, "1m"
            )
            total_records = sum(df.height for df in multi_data.values())
            print(f"   멀티타임프레임 데이터: {total_records:,}개 레코드")
            profiler.end_phase()
            
            # 6. 지표 재계산 (멀티타임프레임)
            profiler.start_phase("6. 멀티타임프레임 지표 계산")
            enriched_multi_data = strategy.precompute_indicators_multi_timeframe(multi_data)
            profiler.end_phase()
            
            # 7. 백테스팅 루프 실행 (일부만)
            profiler.start_phase("7. 백테스팅 루프 실행")
            
            # 실제 백테스팅 루프의 일부만 실행
            base_timeframe = "5m"  # 실제 존재하는 타임프레임 사용
            base_data = enriched_multi_data[base_timeframe]
            time_groups = base_data.group_by("timestamp").agg(
                pl.all().exclude("timestamp")
            ).sort("timestamp")
            
            print(f"   처리할 타임스텝: {time_groups.height:,}개")
            
            # 루프 내부 병목 측정
            loop_start = time.time()
            processed_steps = 0
            max_steps = min(100, time_groups.height)  # 최대 100스텝만 테스트
            
            profiler.start_phase("7a. 루프 내부 - 타임스탬프 순회")
            timestamps = time_groups.get_column("timestamp").to_list()[:max_steps]
            profiler.end_phase()
            
            profiler.start_phase("7b. 루프 내부 - 데이터 필터링")
            filter_time = 0
            for timestamp in timestamps[:10]:  # 10개만 샘플
                filter_start = time.time()
                current_data = base_data.filter(pl.col("timestamp") <= timestamp)
                filter_time += time.time() - filter_start
                processed_steps += 1
                
                if processed_steps >= 10:
                    break
                    
            avg_filter_time = filter_time / processed_steps if processed_steps > 0 else 0
            estimated_total_filter_time = avg_filter_time * time_groups.height
            print(f"   평균 필터링 시간: {avg_filter_time:.4f}초/스텝")
            print(f"   전체 예상 필터링 시간: {estimated_total_filter_time:.1f}초")
            profiler.end_phase()
            
            profiler.start_phase("7c. 루프 내부 - 전략 실행")
            # 전략 실행 시뮬레이션 (몇 개만)
            strategy_time = 0
            for i in range(min(5, len(timestamps))):
                strategy_start = time.time()
                # 전략 실행 시뮬레이션
                time.sleep(0.001)  # 실제 전략 실행 시뮬레이션
                strategy_time += time.time() - strategy_start
            
            avg_strategy_time = strategy_time / 5
            estimated_total_strategy_time = avg_strategy_time * time_groups.height
            print(f"   평균 전략 실행 시간: {avg_strategy_time:.4f}초/스텝")
            print(f"   전체 예상 전략 시간: {estimated_total_strategy_time:.1f}초")
            profiler.end_phase()
            
            profiler.end_phase()  # 백테스팅 루프 종료
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        profiler.print_bottleneck_report()


if __name__ == "__main__":
    print("🔍 실제 백테스팅 병목 프로파일링 시작")
    print("="*60)
    
    asyncio.run(profile_multi_timeframe_backtest()) 