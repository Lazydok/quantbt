# 튜토리얼 3: 크로스 심볼 지표 전략

이번 튜토리얼에서는 한 종목의 기술적 지표를 사용하여 다른 종목의 거래 신호를 생성하는 '크로스 심볼(Cross-symbol)' 전략을 구현하는 방법을 알아봅니다. 예를 들어, 비트코인(BTC)의 시장 동향을 파악하여 이더리움(ETH)의 매매 시점을 결정하는 것과 같은 고급 전략입니다.

> 전체 코드는 아래 Jupyter Notebook 링크에서 확인하고 직접 실행해볼 수 있습니다.
>
> 👉 **[예제 노트북 바로가기: 03_cross_symbol_indicator_strategy.ipynb](../examples/03_cross_symbol_indicator_strategy.ipynb)**

## 1. 크로스 심볼 전략의 개념

크로스 심볼 전략은 시장 전체의 분위기를 주도하는 자산(예: BTC)이나 특정 경제 지표를 기준으로, 변동성이 더 크거나 다른 특성을 가진 자산(예: 알트코인)을 거래하는 데 사용됩니다. 이를 통해 더 안정적이거나 선행하는 신호를 포착하여 거래의 정확도를 높일 수 있습니다.

## 2. 백테스팅 과정

### 단계 1: 커스텀 전략 생성

크로스 심볼 로직을 구현하기 위해 QuantBT의 `TradingStrategy`를 상속받아 커스텀 전략을 만듭니다.

- **`precompute_indicators`**: 백테스팅 시작 전, 필요한 모든 종목의 지표를 미리 계산합니다.
- **`generate_signals`**: 각 시점에서 데이터에 접근하여, 기준이 되는 종목(예: BTC)의 지표 값을 가져와 거래할 종목(예: ETH)의 주문을 생성합니다.

```python
import quantbt as qbt
import polars as pl

class CrossSymbolStrategy(qbt.TradingStrategy):
    def __init__(self, btc_sma_period=50, eth_rsi_period=14):
        self.btc_sma_period = btc_sma_period
        self.eth_rsi_period = eth_rsi_period

    def precompute_indicators(self, data):
        # BTC의 이동평균과 ETH의 RSI를 미리 계산
        data["BTCUSDT"] = data["BTCUSDT"].add_indicator(
            "sma", self.btc_sma_period
        )
        data["ETHUSDT"] = data["ETHUSDT"].add_indicator(
            "rsi", self.eth_rsi_period
        )
        return data

    def generate_signals(self, data):
        orders = []
        
        # 현재 시점의 BTC SMA와 ETH 가격, RSI 값 가져오기
        btc_sma = data.get_indicator_value("BTCUSDT", "sma")
        eth_price = data.get_price("ETHUSDT", "close")
        eth_rsi = data.get_indicator_value("ETHUSDT", "rsi")

        # 매수 조건: BTC가 장기 상승 추세(SMA 상향)이고 ETH가 과매도 상태일 때
        if eth_price > btc_sma and eth_rsi < 30:
            orders.append(self.create_market_order("ETHUSDT", "buy"))

        # 매도 조건: ETH가 과매수 상태일 때
        elif eth_rsi > 70:
            orders.append(self.create_market_order("ETHUSDT", "sell"))
            
        return orders
```

### 단계 2: 백테스팅 실행

생성한 커스텀 전략을 사용하여 백테스팅을 실행합니다. 데이터는 두 종목(`BTCUSDT`, `ETHUSDT`)을 모두 포함해야 합니다.

```python
# 데이터 로드
data = qbt.load_data(symbols=["BTCUSDT", "ETHUSDT"])

# 전략 인스턴스화 및 백테스팅
strategy = CrossSymbolStrategy()
result = qbt.backtest(
    strategy=strategy,
    ohlcv=data,
    symbols=["BTCUSDT", "ETHUSDT"],
    # ... other parameters
)
```

## 3. 결과 분석

결과 분석은 일반적인 백테스팅과 동일하지만, 거래는 `ETHUSDT`에 대해서만 발생했음을 확인할 수 있습니다.

```python
result.stats()
result.plot()

# 거래 내역 확인
print(result.trades)
```

이 가이드를 통해 서로 다른 종목을 연계하는 복잡한 전략도 QuantBT로 손쉽게 구현하고 검증할 수 있음을 확인했습니다. 다음 장에서는 여러 시간 프레임을 동시에 분석하는 멀티 타임프레임 전략을 살펴보겠습니다. 