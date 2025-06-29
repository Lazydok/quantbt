#!/usr/bin/env python3
"""
바이낸스 데이터프로바이더 사용 예제

기본적인 사용법과 다양한 기능들을 보여주는 예제 스크립트
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantbt.infrastructure.data import BinanceDataProvider


async def basic_usage_example():
    """기본 사용법 예제"""
    print("=" * 60)
    print("📈 바이낸스 데이터프로바이더 기본 사용법")
    print("=" * 60)
    
    # 날짜 설정 (최근 7일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    async with BinanceDataProvider() as provider:
        # 1. 기본 데이터 조회
        print("\n1️⃣ 기본 데이터 조회 (BTCUSDT 1분봉)")
        data = await provider.get_data(
            symbols=["BTCUSDT"],
            start=start_date,
            end=end_date,
            timeframe="1m"
        )
        
        if not data.is_empty():
            print(f"✅ 조회 완료: {len(data)}개 캔들")
            print(f"📊 데이터 샘플:")
            print(data.head())
        else:
            print("⚠️ 데이터가 없습니다.")


async def multiple_symbols_example():
    """여러 심볼 조회 예제"""
    print("\n=" * 60)
    print("📊 여러 심볼 동시 조회 예제")
    print("=" * 60)
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    async with BinanceDataProvider() as provider:
        print(f"🔍 심볼: {', '.join(symbols)}")
        print(f"📅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        data = await provider.get_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe="5m"
        )
        
        if not data.is_empty():
            print(f"✅ 조회 완료: {len(data)}개 캔들")
            
            # 심볼별 통계
            for symbol in symbols:
                symbol_data = data.filter(data["symbol"] == symbol)
                if not symbol_data.is_empty():
                    print(f"📈 {symbol}: {len(symbol_data)}개 캔들")


async def csv_export_example():
    """CSV 내보내기 예제"""
    print("\n=" * 60)
    print("📄 CSV 내보내기 예제")
    print("=" * 60)
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    async with BinanceDataProvider() as provider:
        print("📁 CSV 파일로 내보내기...")
        
        exported_files = await provider.export_to_csv(
            symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe="1m",
            output_dir="./example_data"
        )
        
        if exported_files:
            print(f"✅ {len(exported_files)}개 파일 내보내기 완료:")
            for symbol, filepath in exported_files.items():
                print(f"  📄 {symbol}: {filepath}")
        else:
            print("⚠️ 내보낼 데이터가 없습니다.")


async def symbol_discovery_example():
    """심볼 조회 예제"""
    print("\n=" * 60)
    print("🔍 심볼 조회 예제")
    print("=" * 60)
    
    async with BinanceDataProvider() as provider:
        # 전체 심볼 조회
        print("📊 전체 심볼 조회 중...")
        all_symbols = await provider.get_all_symbols()
        print(f"✅ 총 {len(all_symbols)}개 심볼 발견")
        
        # USDT 페어만 필터링
        print("\n💰 USDT 페어 조회 중...")
        usdt_symbols = await provider.get_usdt_symbols()
        print(f"✅ 총 {len(usdt_symbols)}개 USDT 페어 발견")
        
        # 상위 10개 출력
        print(f"\n📈 상위 10개 USDT 페어:")
        for i, symbol in enumerate(usdt_symbols[:10], 1):
            print(f"  {i:2d}. {symbol}")


async def storage_info_example():
    """저장소 정보 조회 예제"""
    print("\n=" * 60)
    print("💾 저장소 정보 조회 예제")
    print("=" * 60)
    
    async with BinanceDataProvider() as provider:
        storage_info = provider.get_storage_info()
        
        print("📊 저장소 정보:")
        print(f"  프로바이더: {storage_info['provider']}")
        
        if 'db_info' in storage_info:
            db_info = storage_info['db_info']
            print(f"  DB 경로: {db_info.get('db_path', 'N/A')}")
        
        if 'cache_info' in storage_info:
            cache_info = storage_info['cache_info']
            print(f"  캐시 디렉토리: {cache_info.get('cache_dir', 'N/A')}")
            print(f"  캐시 파일 수: {cache_info.get('total_files', 'N/A')}")


async def timeframe_example():
    """다양한 타임프레임 예제"""
    print("\n=" * 60)
    print("⏰ 다양한 타임프레임 예제")
    print("=" * 60)
    
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    symbol = "BTCUSDT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    async with BinanceDataProvider() as provider:
        print(f"📈 심볼: {symbol}")
        print(f"📅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        for timeframe in timeframes:
            try:
                data = await provider.get_data(
                    symbols=[symbol],
                    start=start_date,
                    end=end_date,
                    timeframe=timeframe
                )
                
                if not data.is_empty():
                    print(f"✅ {timeframe:>3s}: {len(data):>5,}개 캔들")
                else:
                    print(f"⚠️ {timeframe:>3s}: 데이터 없음")
                    
            except Exception as e:
                print(f"❌ {timeframe:>3s}: 오류 - {e}")


async def main():
    """메인 실행 함수"""
    print("🚀 바이낸스 데이터프로바이더 예제 시작")
    
    try:
        # 기본 사용법
        await basic_usage_example()
        
        # 여러 심볼 조회
        await multiple_symbols_example()
        
        # 심볼 조회
        await symbol_discovery_example()
        
        # 다양한 타임프레임
        await timeframe_example()
        
        # CSV 내보내기
        await csv_export_example()
        
        # 저장소 정보
        await storage_info_example()
        
        print("\n" + "=" * 60)
        print("✅ 모든 예제 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 예제 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main()) 