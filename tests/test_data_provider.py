"""
데이터 제공자 테스트

CSV 데이터 제공자의 동작을 검증합니다.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from quantbt.infrastructure.data.csv_provider import CSVDataProvider


class TestCSVDataProvider:
    """CSV 데이터 제공자 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 샘플 CSV 데이터 생성
            self._create_sample_data(temp_dir)
            yield temp_dir
    
    def _create_sample_data(self, data_dir: str):
        """샘플 CSV 데이터 생성"""
        data_dir = Path(data_dir)
        
        # AAPL 데이터
        aapl_data = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5)
            ],
            "open": [150.0, 152.0, 151.0, 153.0, 154.0],
            "high": [155.0, 157.0, 156.0, 158.0, 159.0],
            "low": [149.0, 151.0, 150.0, 152.0, 153.0],
            "close": [154.0, 156.0, 155.0, 157.0, 158.0],
            "volume": [1000000, 1100000, 1050000, 1200000, 1150000]
        })
        aapl_data.write_csv(data_dir / "AAPL.csv")
        
        # MSFT 데이터
        msft_data = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5)
            ],
            "open": [300.0, 302.0, 301.0, 303.0, 304.0],
            "high": [305.0, 307.0, 306.0, 308.0, 309.0],
            "low": [299.0, 301.0, 300.0, 302.0, 303.0],
            "close": [304.0, 306.0, 305.0, 307.0, 308.0],
            "volume": [800000, 850000, 820000, 900000, 880000]
        })
        msft_data.write_csv(data_dir / "MSFT.csv")
        
        # 통합 데이터 파일 (여러 심볼이 하나의 파일에)
        combined_data = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1), datetime(2023, 1, 1),
                datetime(2023, 1, 2), datetime(2023, 1, 2),
                datetime(2023, 1, 3), datetime(2023, 1, 3)
            ],
            "symbol": ["GOOGL", "AMZN", "GOOGL", "AMZN", "GOOGL", "AMZN"],
            "open": [100.0, 200.0, 102.0, 202.0, 101.0, 201.0],
            "high": [105.0, 205.0, 107.0, 207.0, 106.0, 206.0],
            "low": [99.0, 199.0, 101.0, 201.0, 100.0, 200.0],
            "close": [104.0, 204.0, 106.0, 206.0, 105.0, 205.0],
            "volume": [500000, 600000, 520000, 620000, 510000, 610000]
        })
        combined_data.write_csv(data_dir / "data.csv")
    
    def test_provider_creation(self, temp_data_dir):
        """데이터 제공자 생성 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        assert provider.name == "CSVDataProvider"
        assert provider.data_dir == Path(temp_data_dir)
    
    def test_invalid_data_dir_raises_error(self):
        """존재하지 않는 디렉토리로 생성 시 에러 테스트"""
        with pytest.raises(ValueError, match="Data directory does not exist"):
            CSVDataProvider("/nonexistent/directory")
    
    def test_get_symbols(self, temp_data_dir):
        """심볼 목록 조회 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        symbols = provider.get_symbols()
        
        # 파일명에서 추출된 심볼들과 파일 내용의 심볼들이 모두 포함되어야 함
        expected_symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "DATA"}  # DATA.csv도 포함
        assert set(symbols).issuperset({"AAPL", "MSFT", "GOOGL", "AMZN"})
    
    def test_validate_symbols(self, temp_data_dir):
        """심볼 유효성 검증 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        assert provider.validate_symbols(["AAPL", "MSFT"])
        assert not provider.validate_symbols(["AAPL", "INVALID_SYMBOL"])
    
    def test_validate_timeframe(self, temp_data_dir):
        """시간 프레임 유효성 검증 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        assert provider.validate_timeframe("1d")
        assert provider.validate_timeframe("1D")
        assert provider.validate_timeframe("1h")
        assert not provider.validate_timeframe("invalid")
    
    def test_validate_date_range(self, temp_data_dir):
        """날짜 범위 유효성 검증 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        future_date = datetime(2030, 1, 1)
        
        assert provider.validate_date_range(start, end)
        assert not provider.validate_date_range(end, start)  # 시작일이 종료일보다 늦음
        assert not provider.validate_date_range(future_date, future_date + timedelta(days=1))  # 미래 날짜
    
    @pytest.mark.asyncio
    async def test_get_data_single_symbol(self, temp_data_dir):
        """단일 심볼 데이터 조회 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        data = await provider.get_data(["AAPL"], start, end)
        
        assert data.height == 5  # 5일간의 데이터
        assert "AAPL" in data.select("symbol").unique().to_series().to_list()
        
        # 타임스탬프 순으로 정렬된 데이터 확인
        sorted_data = data.sort("timestamp")
        close_prices = sorted_data.select("close").to_series().to_list()
        assert close_prices == [154.0, 156.0, 155.0, 157.0, 158.0]
    
    @pytest.mark.asyncio
    async def test_get_data_multiple_symbols(self, temp_data_dir):
        """다중 심볼 데이터 조회 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        
        data = await provider.get_data(["AAPL", "MSFT"], start, end)
        
        symbols = data.select("symbol").unique().to_series().to_list()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert data.height == 6  # 2개 심볼 * 3일
    
    @pytest.mark.asyncio
    async def test_get_data_date_filtering(self, temp_data_dir):
        """날짜 필터링 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 2)
        end = datetime(2023, 1, 3)
        
        data = await provider.get_data(["AAPL"], start, end)
        
        assert data.height == 2  # 2일간의 데이터만
        timestamps = data.select("timestamp").to_series().to_list()
        assert all(start <= ts <= end for ts in timestamps)
    
    @pytest.mark.asyncio
    async def test_get_data_invalid_symbols(self, temp_data_dir):
        """잘못된 심볼로 데이터 조회 시 에러 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        with pytest.raises(ValueError, match="Invalid symbols"):
            await provider.get_data(["INVALID_SYMBOL"], start, end)
    
    @pytest.mark.asyncio
    async def test_get_data_stream(self, temp_data_dir):
        """데이터 스트림 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 3)
        
        batches = []
        async for batch in provider.get_data_stream(["AAPL"], start, end):
            batches.append(batch)
        
        assert len(batches) == 3  # 3일간의 배치
        
        # 각 배치는 누적 데이터를 포함해야 함
        assert batches[0].data.height == 1  # 첫 번째 날
        assert batches[1].data.height == 2  # 첫 번째 + 두 번째 날
        assert batches[2].data.height == 3  # 모든 날
    
    def test_get_data_info(self, temp_data_dir):
        """데이터 정보 조회 테스트"""
        provider = CSVDataProvider(temp_data_dir)
        info = provider.get_data_info()
        
        assert info["provider"] == "CSVDataProvider"
        assert info["data_dir"] == temp_data_dir
        assert "available_symbols" in info
        assert "symbol_count" in info
        assert info["symbol_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__]) 