# 튜토리얼 1: 간단한 전략 백테스팅

이 튜토리얼에서는 QuantBT를 사용하여 단일 종목에 대한 간단한 이동평균 교차 전략을 백테스팅하는 과정을 안내합니다. QuantBT의 가장 기본적인 사용법을 익힐 수 있습니다.

>  전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 01_simple_strategy.ipynb](../examples/01_simple_strategy.ipynb)**

## 1. 기본 개념

- **데이터 프로바이더 (Data Provider)**: 백테스팅에 필요한 가격 데이터(OHLCV)를 제공합니다. CSV 파일, 데이터베이스, 실시간 API 등 다양한 소스를 지원합니다.
- **전략 (Strategy)**: 매수/매도 신호를 생성하는 로직입니다. 지표 계산 및 신호 생성 규칙을 정의합니다.
- **브로커 (Broker)**: 가상의 거래소 역할을 하며, 주문을 접수하고 체결하며 포트폴리오를 관리합니다.
- **백테스트 엔진 (Backtest Engine)**: 데이터 프로바이더, 전략, 브로커를 연결하여 백테스팅 전 과정을 조율하고 실행합니다.

## 2. 백테스팅 과정

### 단계 1: 필요한 모듈 임포트

백테스팅에 필요한 주요 클래스들을 `quantbt` 라이브러리에서 직접 임포트합니다.

```python
from datetime import datetime
from quantbt import (
    BacktestEngine,
    BacktestConfig,
    UpbitDataProvider,
    SimpleBroker,
    SimpleSMAStrategy,
)
```

### 단계 2: 백테스팅 설정 정의

`BacktestConfig`를 사용하여 백테스팅의 기본 조건을 설정합니다. 여기에는 분석할 종목, 기간, 타임프레임, 초기 자본금, 수수료 및 슬리피지 비율 등이 포함됩니다.

```python
config = BacktestConfig(
    symbols=["KRW-BTC"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1d",
    initial_cash=10000,
    commission_rate=0.001,
    slippage_rate=0.0, # 슬리피지 비율
    save_portfolio_history=True,
)
```

### 단계 3: 구성 요소 초기화

백테스팅에 필요한 각 구성 요소(데이터 프로바이더, 전략, 브로커)를 초기화합니다.

```python
# 데이터 프로바이더 설정 (업비트 데이터 사용)
data_provider = UpbitDataProvider()

# 전략 선택 (단순 이동평균 교차 전략)
# 10일 이동평균이 30일 이동평균을 상향 돌파하면 매수, 하향 돌파하면 매도합니다.
strategy = SimpleSMAStrategy(buy_sma=10, sell_sma=30)

# 브로커 설정
broker = SimpleBroker(
    initial_cash=config.initial_cash,
    commission_rate=config.commission_rate,
)
```

### 단계 4: 백테스트 엔진 실행

`BacktestEngine`을 생성하고, 위에서 만든 구성 요소들을 설정한 뒤, `run` 메소드를 호출하여 백테스팅을 실행합니다.

```python
# 백테스팅 엔진 설정 및 실행
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.set_data_provider(data_provider)
engine.set_broker(broker)

result = engine.run(config)
```

## 3. 결과 분석

백테스팅이 완료되면 `result` 객체를 통해 다양한 성능 지표와 거래 내역을 확인할 수 있습니다.

```python
# 요약 통계 출력
result.print_summary()

# 포트폴리오 성과 시각화 (Jupyter Notebook 환경에서 실행 권장)
result.plot_portfolio_performance()
```

### 주요 성능 지표

- **Total Return**: 총 수익률
- **Sharpe Ratio**: 샤프 지수 (위험 대비 수익률)
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 거래 승률

이 가이드를 통해 QuantBT의 기본적인 사용 흐름을 파악하셨기를 바랍니다. 다음 튜토리얼에서는 더 복잡한 멀티 심볼 전략을 다루겠습니다.