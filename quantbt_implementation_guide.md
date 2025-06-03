# QuantBT 백테스팅 엔진 구현 가이드

## 프로젝트 셋업

### 1. 프로젝트 초기화

```bash
# 프로젝트 생성
mkdir quantbt
cd quantbt

# Poetry로 의존성 관리
poetry init
poetry add polars numpy pydantic click dynaconf
poetry add --group dev pytest black isort mypy pre-commit
poetry add --group optional plotly matplotlib streamlit fastapi

# Git 초기화
git init
git add .
git commit -m "Initial commit"
```

### 2. pyproject.toml 설정

```toml
[tool.poetry]
name = "quantbt"
version = "0.1.0"
description = "High-performance backtesting engine for quantitative trading"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "quantbt"}]

[tool.poetry.dependencies]
python = "^3.9"
polars = "^0.20.0"
numpy = "^1.24.0"
pydantic = "^2.0.0"
click = "^8.0.0"
dynaconf = "^3.2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^23.0.0"
isort = "^5.12.0"
mypy = "^1.5.0"
pre-commit = "^3.0.0"

[tool.poetry.group.optional.dependencies]
plotly = "^5.17.0"
matplotlib = "^3.7.0"
streamlit = "^1.28.0"
fastapi = "^0.104.0"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## 핵심 구현

### 1. 도메인 엔티티

```python
# quantbt/core/entities/market_data.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any
import polars as pl

@dataclass(frozen=True)
class MarketData:
    """시장 데이터 엔티티"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "metadata": self.metadata or {}
        }

@dataclass
class MarketDataBatch:
    """배치 형태의 시장 데이터"""
    data: pl.DataFrame
    symbols: list[str]
    timeframe: str
    
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """특정 심볼의 최신 데이터 조회"""
        symbol_data = self.data.filter(pl.col("symbol") == symbol)
        if symbol_data.height == 0:
            return None
        
        latest = symbol_data.row(-1, named=True)
        return MarketData(
            timestamp=latest["timestamp"],
            symbol=latest["symbol"],
            open=latest["open"],
            high=latest["high"],
            low=latest["low"],
            close=latest["close"],
            volume=latest["volume"]
        )
```

```python
# quantbt/core/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import uuid4

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = 1
    SELL = -1

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """주문 엔티티"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    order_id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    def is_market_order(self) -> bool:
        return self.order_type == OrderType.MARKET
    
    def is_limit_order(self) -> bool:
        return self.order_type == OrderType.LIMIT
```

```python
# quantbt/core/entities/position.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Position:
    """포지션 엔티티"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """시장 가치"""
        return self.quantity * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익"""
        if self.quantity == 0:
            return 0.0
        return (self.market_price - self.avg_price) * self.quantity
    
    @property
    def is_long(self) -> bool:
        """롱 포지션 여부"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """숏 포지션 여부"""
        return self.quantity < 0
    
    def update_position(self, trade_quantity: float, trade_price: float) -> None:
        """포지션 업데이트"""
        if self.quantity == 0:
            self.quantity = trade_quantity
            self.avg_price = trade_price
        else:
            total_cost = self.quantity * self.avg_price + trade_quantity * trade_price
            self.quantity += trade_quantity
            if self.quantity != 0:
                self.avg_price = total_cost / self.quantity
            else:
                self.avg_price = 0.0
```

### 2. 인터페이스 정의

```python
# quantbt/core/interfaces/strategy.py
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional
from datetime import datetime

from ..entities.market_data import MarketData, MarketDataBatch
from ..entities.order import Order
from ..entities.trade import Trade

class BacktestContext:
    """백테스팅 컨텍스트"""
    def __init__(self, initial_cash: float, symbols: List[str]):
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.current_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

class IStrategy(Protocol):
    """전략 인터페이스"""
    
    def initialize(self, context: BacktestContext) -> None:
        """전략 초기화
        
        Args:
            context: 백테스팅 컨텍스트
        """
        ...
    
    def on_data(self, data: MarketDataBatch) -> List[Order]:
        """데이터 수신 시 호출
        
        Args:
            data: 시장 데이터 배치
            
        Returns:
            생성할 주문 리스트
        """
        ...
    
    def on_order_fill(self, trade: 'Trade') -> None:
        """주문 체결 시 호출
        
        Args:
            trade: 체결된 거래 정보
        """
        ...
    
    def finalize(self, context: BacktestContext) -> None:
        """백테스팅 종료 시 호출
        
        Args:
            context: 백테스팅 컨텍스트
        """
        ...

class StrategyBase(ABC):
    """전략 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.context: Optional[BacktestContext] = None
        self.indicators: Dict[str, Any] = {}
    
    def initialize(self, context: BacktestContext) -> None:
        """기본 초기화"""
        self.context = context
        self.setup_indicators()
    
    @abstractmethod
    def setup_indicators(self) -> None:
        """지표 설정"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성"""
        pass
    
    def on_data(self, data: MarketDataBatch) -> List[Order]:
        """데이터 처리"""
        return self.generate_signals(data)
    
    def on_order_fill(self, trade: 'Trade') -> None:
        """주문 체결 처리"""
        pass
    
    def finalize(self, context: BacktestContext) -> None:
        """종료 처리"""
        pass
```

```python
# quantbt/core/interfaces/data_provider.py
from abc import ABC, abstractmethod
from typing import Protocol, List, AsyncIterator, Optional
from datetime import datetime
import polars as pl

from ..entities.market_data import MarketDataBatch

class IDataProvider(Protocol):
    """데이터 제공자 인터페이스"""
    
    async def get_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> pl.DataFrame:
        """시장 데이터 조회
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframe: 시간 프레임
            
        Returns:
            시장 데이터 DataFrame
        """
        ...
    
    async def get_data_stream(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D"
    ) -> AsyncIterator[MarketDataBatch]:
        """시장 데이터 스트림
        
        Args:
            symbols: 심볼 리스트
            start: 시작 시간
            end: 종료 시간
            timeframe: 시간 프레임
            
        Yields:
            시간순으로 정렬된 시장 데이터 배치
        """
        ...
    
    def get_symbols(self) -> List[str]:
        """사용 가능한 심볼 목록"""
        ...
```

### 3. 백테스팅 엔진

```python
# quantbt/core/services/backtest_engine.py
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import polars as pl

from ..interfaces.strategy import IStrategy, BacktestContext
from ..interfaces.data_provider import IDataProvider
from ..interfaces.broker import IBroker
from ..entities.order import Order
from ..entities.position import Position
from ..entities.account import Account
from ..value_objects.backtest_config import BacktestConfig
from ..value_objects.backtest_result import BacktestResult

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
        broker: IBroker,
        config: BacktestConfig
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.broker = broker
        self.config = config
        self._account_history: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
    
    async def run(self) -> BacktestResult:
        """백테스팅 실행"""
        # 초기화
        context = BacktestContext(
            initial_cash=self.config.initial_cash,
            symbols=self.config.symbols
        )
        
        self.strategy.initialize(context)
        await self.broker.initialize(self.config)
        
        # 데이터 스트림 처리
        async for data_batch in self.data_provider.get_data_stream(
            symbols=self.config.symbols,
            start=self.config.start_date,
            end=self.config.end_date,
            timeframe=self.config.timeframe
        ):
            await self._process_data_batch(data_batch, context)
        
        # 종료 처리
        self.strategy.finalize(context)
        
        return await self._generate_result()
    
    async def _process_data_batch(
        self, 
        data_batch: 'MarketDataBatch', 
        context: BacktestContext
    ) -> None:
        """데이터 배치 처리"""
        # 시간 업데이트
        latest_time = data_batch.data.select(pl.col("timestamp").max()).item()
        context.current_time = latest_time
        
        # 포트폴리오 가치 업데이트
        await self.broker.update_portfolio_value(data_batch)
        
        # 전략 실행
        orders = self.strategy.on_data(data_batch)
        
        # 주문 제출
        for order in orders:
            await self.broker.submit_order(order)
        
        # 주문 체결 처리
        trades = await self.broker.process_orders(data_batch)
        for trade in trades:
            self.strategy.on_order_fill(trade)
        
        # 상태 기록
        await self._record_state(latest_time)
    
    async def _record_state(self, timestamp: datetime) -> None:
        """현재 상태 기록"""
        account = await self.broker.get_account()
        
        self._account_history.append({
            "timestamp": timestamp,
            "cash": account.cash,
            "portfolio_value": account.portfolio_value,
            "total_value": account.total_value
        })
    
    async def _generate_result(self) -> BacktestResult:
        """결과 생성"""
        account_df = pl.DataFrame(self._account_history)
        
        # 성과 지표 계산
        returns = account_df.select(
            pl.col("total_value").pct_change().alias("returns")
        ).drop_nulls()
        
        total_return = (
            account_df.select(pl.col("total_value").last()).item() /
            account_df.select(pl.col("total_value").first()).item() - 1
        )
        
        # 더 많은 지표들...
        
        return BacktestResult(
            total_return=total_return,
            account_history=account_df,
            trade_history=pl.DataFrame(self._trade_history),
            config=self.config
        )
```

### 4. 전략 예시

```python
# quantbt/strategies/examples/sma_crossover.py
from typing import List
import polars as pl

from ...core.interfaces.strategy import StrategyBase
from ...core.entities.market_data import MarketDataBatch
from ...core.entities.order import Order, OrderSide, OrderType
from ...indicators.trend.moving_average import SMA

class SMACrossoverStrategy(StrategyBase):
    """단순 이동평균 교차 전략"""
    
    def __init__(
        self, 
        short_window: int = 20, 
        long_window: int = 50,
        position_size: float = 0.1
    ):
        super().__init__("SMA_Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
    
    def setup_indicators(self) -> None:
        """지표 설정"""
        self.indicators["sma_short"] = SMA(self.short_window)
        self.indicators["sma_long"] = SMA(self.long_window)
    
    def generate_signals(self, data: MarketDataBatch) -> List[Order]:
        """신호 생성"""
        orders = []
        
        for symbol in data.symbols:
            signal = self._calculate_signal(data, symbol)
            
            if signal == 1:  # 매수 신호
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=self._calculate_quantity(symbol),
                    order_type=OrderType.MARKET
                ))
            elif signal == -1:  # 매도 신호
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=self._get_position_quantity(symbol),
                    order_type=OrderType.MARKET
                ))
        
        return orders
    
    def _calculate_signal(self, data: MarketDataBatch, symbol: str) -> int:
        """신호 계산"""
        symbol_data = data.data.filter(pl.col("symbol") == symbol)
        
        if symbol_data.height < self.long_window:
            return 0
        
        # 이동평균 계산
        sma_short = self.indicators["sma_short"].calculate(
            symbol_data.select("close").to_series()
        )
        sma_long = self.indicators["sma_long"].calculate(
            symbol_data.select("close").to_series()
        )
        
        # 교차 신호 확인
        if len(sma_short) < 2 or len(sma_long) < 2:
            return 0
        
        # 골든 크로스 (상승 신호)
        if sma_short[-1] > sma_long[-1] and sma_short[-2] <= sma_long[-2]:
            return 1
        
        # 데드 크로스 (하락 신호)
        if sma_short[-1] < sma_long[-1] and sma_short[-2] >= sma_long[-2]:
            return -1
        
        return 0
    
    def _calculate_quantity(self, symbol: str) -> float:
        """매수 수량 계산"""
        if not self.context:
            return 0.0
        
        # 포트폴리오 가치의 일정 비율로 매수
        target_value = self.context.portfolio_value * self.position_size
        current_price = self._get_current_price(symbol)
        
        if current_price > 0:
            return target_value / current_price
        return 0.0
    
    def _get_position_quantity(self, symbol: str) -> float:
        """현재 포지션 수량 조회"""
        # 브로커에서 현재 포지션 조회
        return 0.0  # 실제 구현 필요
    
    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        # 현재 가격 조회 로직
        return 0.0  # 실제 구현 필요
```

### 5. CLI 인터페이스

```python
# quantbt/cli/main.py
import click
import asyncio
from datetime import datetime
from pathlib import Path

from ..core.services.backtest_engine import BacktestEngine
from ..infrastructure.data.csv_provider import CSVDataProvider
from ..infrastructure.brokers.simple_broker import SimpleBroker
from ..strategies.examples.sma_crossover import SMACrossoverStrategy
from ..core.value_objects.backtest_config import BacktestConfig

@click.group()
def cli():
    """QuantBT - Quantitative Backtesting Engine"""
    pass

@cli.command()
@click.option('--data-path', required=True, type=click.Path(exists=True), 
              help='Path to data directory')
@click.option('--symbols', required=True, help='Comma-separated list of symbols')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--initial-cash', default=100000, help='Initial cash amount')
@click.option('--strategy', default='sma_crossover', help='Strategy to use')
@click.option('--output', default='result.html', help='Output file path')
async def run(data_path, symbols, start_date, end_date, initial_cash, strategy, output):
    """Run backtest"""
    
    # 설정 파싱
    symbols_list = [s.strip() for s in symbols.split(',')]
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 컴포넌트 생성
    config = BacktestConfig(
        start_date=start_dt,
        end_date=end_dt,
        initial_cash=initial_cash,
        symbols=symbols_list
    )
    
    data_provider = CSVDataProvider(Path(data_path))
    broker = SimpleBroker()
    
    if strategy == 'sma_crossover':
        strategy_obj = SMACrossoverStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 백테스팅 실행
    engine = BacktestEngine(strategy_obj, data_provider, broker, config)
    result = await engine.run()
    
    # 결과 출력
    click.echo(f"Total Return: {result.total_return:.2%}")
    click.echo(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    # 결과 저장
    result.export_html(output)
    click.echo(f"Results saved to {output}")

if __name__ == '__main__':
    cli()
```

## 배포 준비

### 1. README.md

```markdown
# QuantBT - Quantitative Backtesting Engine

High-performance, modular backtesting engine for quantitative trading strategies.

## Features

- 🚀 High-performance data processing with Polars
- 🔧 Modular and extensible architecture
- 📊 Rich performance analytics and visualizations
- 🐍 Type-safe Python implementation
- 🧪 Comprehensive testing suite

## Quick Start

```bash
pip install quantbt

# Run a simple backtest
quantbt run --data-path ./data --symbols AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31
```

## Documentation

Visit our [documentation](https://quantbt.readthedocs.io) for detailed guides and API reference.
```

### 2. GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install
    
    - name: Run tests
      run: poetry run pytest
    
    - name: Run linting
      run: |
        poetry run black --check .
        poetry run isort --check-only .
        poetry run mypy .
```

이 구현 가이드를 따라 체계적이고 확장 가능한 백테스팅 엔진을 개발할 수 있습니다. 핵심은 **인터페이스 기반 설계**, **모듈화**, **타입 안정성**을 통해 기존 시스템의 문제점들을 해결하는 것입니다. 