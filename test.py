from datetime import datetime
from quantbt import (
    BacktestEngine,
    BacktestConfig,
    UpbitDataProvider,
    SimpleBroker,
    SimpleSMAStrategy,
)

# 1. 데이터 프로바이더 설정
data_provider = UpbitDataProvider()

# 2. 백테스팅 설정
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000,
    commission_rate=0.001,
    slippage_rate=0.0,
    save_portfolio_history=True,
)

# 3. 전략 선택
strategy = SimpleSMAStrategy(buy_sma=10, sell_sma=30)

# 4. 브로커 설정
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
)

# 5. 백테스팅 엔진 설정 및 실행
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)

# 6. 결과 확인
result.print_summary()