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

백테스팅에 필요한 주요 클래스들을 임포트합니다.

```python
import quantbt as qbt
from quantbt.strategies import SMAStrategy
from datetime import datetime
```

### 단계 2: 데이터 불러오기

백테스팅에 사용할 데이터를 로드합니다. 여기서는 예제 CSV 파일을 사용합니다.

```python
ohlcv = qbt.load_data("BTCUSDT") 
```

### 단계 3: 전략 및 백테스트 설정

사용할 전략을 생성하고, 백테스팅 기간, 초기 자본금 등 기본 설정을 구성합니다.

```python
# 이동평균 교차 전략 생성
strategy = SMAStrategy(fast_sma=10, slow_sma=30)

# 백테스팅 실행
result = qbt.backtest(
    strategy=strategy,
    ohlcv=ohlcv,
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_cash=10000,
    commission=0.001
)
```

## 3. 결과 분석

백테스팅이 완료되면 `result` 객체를 통해 다양한 성능 지표와 거래 내역을 확인할 수 있습니다.

```python
# 요약 통계 출력
result.stats()

# 수익 곡선 및 주요 지표 시각화
result.plot()
```

### 주요 성능 지표

- **Total Return**: 총 수익률
- **Sharpe Ratio**: 샤프 지수 (위험 대비 수익률)
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 거래 승률

이 가이드를 통해 QuantBT의 기본적인 사용 흐름을 파악하셨기를 바랍니다. 다음 튜토리얼에서는 더 복잡한 멀티 심볼 전략을 다루겠습니다. 