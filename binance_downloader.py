#!/usr/bin/env python3
"""
바이낸스 데이터 다운로더

전체 USDT 페어 심볼의 1분봉 데이터를 다운로드하고 DB에 저장하는 독립 실행 스크립트
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quantbt.infrastructure.data import BinanceDataProvider


async def download_all_usdt_data(
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
    batch_size: int = 5,
    export_csv: bool = False,
    csv_dir: str = "./binance_data"
):
    """전체 USDT 페어 데이터 다운로드"""
    
    # 날짜 파싱
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        print("올바른 형식: YYYY-MM-DD (예: 2024-01-01)")
        return
    
    print("🚀 바이낸스 데이터 다운로더 시작")
    print(f"📅 다운로드 기간: {start_date} ~ {end_date}")
    print(f"⏰ 타임프레임: {timeframe}")
    print(f"🔢 배치 크기: {batch_size}")
    print(f"📁 CSV 내보내기: {'예' if export_csv else '아니오'}")
    
    if export_csv:
        print(f"📂 CSV 저장 경로: {csv_dir}")
    
    print("\n" + "="*60)
    
    # 바이낸스 데이터 프로바이더 초기화
    async with BinanceDataProvider() as provider:
        try:
            # 전체 USDT 페어 다운로드
            results = await provider.download_all_usdt_symbols(
                start=start,
                end=end,
                timeframe=timeframe,
                batch_size=batch_size
            )
            
            print("\n" + "="*60)
            print("📊 다운로드 결과 상세")
            print("="*60)
            
            # 성공한 심볼들
            successful_symbols = [symbol for symbol, count in results.items() if count > 0]
            failed_symbols = [symbol for symbol, count in results.items() if count == -1]
            empty_symbols = [symbol for symbol, count in results.items() if count == 0]
            
            print(f"✅ 성공: {len(successful_symbols)}개 심볼")
            print(f"❌ 실패: {len(failed_symbols)}개 심볼")
            print(f"⚠️ 데이터 없음: {len(empty_symbols)}개 심볼")
            
            if export_csv and successful_symbols:
                print(f"\n📄 CSV 내보내기 시작...")
                
                # 성공한 심볼들만 CSV로 내보내기
                exported_files = await provider.export_to_csv(
                    symbols=successful_symbols[:10],  # 처음 10개만 내보내기 (테스트용)
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    output_dir=csv_dir
                )
                
                print(f"📁 {len(exported_files)}개 파일이 {csv_dir}에 저장되었습니다.")
            
            # 저장소 정보 출력
            storage_info = provider.get_storage_info()
            print(f"\n💾 저장소 정보:")
            print(f"DB 경로: {storage_info['db_info'].get('db_path', 'N/A')}")
            print(f"캐시 디렉토리: {storage_info['cache_info'].get('cache_dir', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 다운로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()


async def download_specific_symbols(
    symbols: list,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
    export_csv: bool = False,
    csv_dir: str = "./binance_data"
):
    """특정 심볼들만 다운로드"""
    
    # 날짜 파싱
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        return
    
    print("🎯 특정 심볼 다운로드 시작")
    print(f"📈 심볼: {', '.join(symbols)}")
    print(f"📅 기간: {start_date} ~ {end_date}")
    print(f"⏰ 타임프레임: {timeframe}")
    
    async with BinanceDataProvider() as provider:
        try:
            results = {}
            
            for symbol in symbols:
                print(f"\n📈 {symbol} 다운로드 중...")
                
                # 다운로드 전용 메서드 사용 (캐시 저장 없음)
                symbol_counts = await provider._fetch_raw_data_download_only([symbol], start, end, timeframe)
                
                count = symbol_counts.get(symbol, 0)
                if count > 0:
                    results[symbol] = count
                    print(f"✅ {symbol}: {count}개 캔들 다운로드 완료")
                elif count == 0:
                    results[symbol] = 0
                    print(f"⚠️ {symbol}: 데이터가 없습니다.")
                else:
                    results[symbol] = -1
                    print(f"❌ {symbol}: 다운로드 실패")
            
            if export_csv:
                print(f"\n📄 CSV 내보내기...")
                exported_files = await provider.export_to_csv(
                    symbols=symbols,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    output_dir=csv_dir
                )
                print(f"📁 {len(exported_files)}개 파일이 저장되었습니다.")
            
            # 결과 요약
            total_candles = sum(count for count in results.values() if count > 0)
            print(f"\n📊 총 {total_candles:,}개 캔들 다운로드 완료")
            
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="바이낸스 데이터 다운로더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 USDT 페어 다운로드 (2024년 1월)
  python binance_downloader.py --start 2024-01-01 --end 2024-01-31

  # 특정 심볼만 다운로드
  python binance_downloader.py --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2024-01-07

  # CSV 파일로 내보내기
  python binance_downloader.py --symbols BTCUSDT --start 2024-01-01 --end 2024-01-02 --csv --csv-dir ./my_data

  # 5분봉 데이터 다운로드
  python binance_downloader.py --symbols BTCUSDT --start 2024-01-01 --end 2024-01-07 --timeframe 5m
        """
    )
    
    parser.add_argument(
        "--start", 
        required=True,
        help="시작 날짜 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end", 
        required=True,
        help="종료 날짜 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="다운로드할 심볼들 (예: BTCUSDT ETHUSDT). 미지정시 전체 USDT 페어 다운로드"
    )
    
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="타임프레임 (기본값: 1m)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="배치 크기 (전체 다운로드시만 사용, 기본값: 5)"
    )
    
    parser.add_argument(
        "--csv",
        action="store_true",
        help="CSV 파일로 내보내기"
    )
    
    parser.add_argument(
        "--csv-dir",
        default="./binance_data",
        help="CSV 파일 저장 디렉토리 (기본값: ./binance_data)"
    )
    
    args = parser.parse_args()
    
    # 비동기 실행
    if args.symbols:
        # 특정 심볼 다운로드
        asyncio.run(download_specific_symbols(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
            export_csv=args.csv,
            csv_dir=args.csv_dir
        ))
    else:
        # 전체 USDT 페어 다운로드
        asyncio.run(download_all_usdt_data(
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
            batch_size=args.batch_size,
            export_csv=args.csv,
            csv_dir=args.csv_dir
        ))


if __name__ == "__main__":
    main() 