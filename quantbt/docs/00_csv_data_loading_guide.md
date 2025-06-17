# 가이드: 로컬 CSV 파일로 나만의 데이터 사용하기

QuantBT의 모든 예제는 기본적으로 `UpbitDataProvider`를 사용하도록 설정되어 있습니다. 이 데이터 프로바이더는 Upbit API를 통해 시장 데이터를 실시간으로 조회하고, 한번 조회한 데이터는 로컬에 캐싱하여 다음 실행부터는 매우 빠르게 데이터를 불러오는 장점이 있습니다.

하지만 Upbit에서 지원하지 않는 자산을 분석하거나, 특정 기간의 데이터를 오프라인 환경에서 사용하고 싶을 때가 있습니다. 이런 상황을 위해 `CSVDataProvider`를 사용하여 자신만의 데이터를 백테스팅에 활용할 수 있습니다.

## 1. CSV 데이터 준비하기

먼저, 백테스팅에 사용할 CSV 파일을 준비해야 합니다. `CSVDataProvider`가 올바르게 데이터를 읽으려면 파일이 정해진 형식을 따르는 것이 중요합니다.

### 데이터 포맷

-   **필수 컬럼**: `date`, `open`, `high`, `low`, `close`, `volume`
-   **날짜 형식**: `YYYY-MM-DD` 또는 `YYYY-MM-DD HH:MM:SS`

아래는 `KRW-BTC`의 일봉 데이터 예시입니다.

```csv
date,open,high,low,close,volume
2024-01-01,57000000,58000000,56000000,57500000,100
2024-01-02,57500000,58500000,57000000,58000000,120
2024-01-03,58000000,59000000,57800000,58800000,150
...
```

> **💡 팁:** 만약 날짜 컬럼명이 `date`가 아니라 `timestamp`나 `time` 등 다른 이름이라면, `CSVDataProvider`를 생성할 때 `timestamp_column` 인자를 통해 직접 지정할 수 있습니다.

## 2. `CSVDataProvider` 설정 및 사용법

CSV 파일이 준비되었다면, 백테스트 코드에서 `UpbitDataProvider`를 `CSVDataProvider`로 교체하기만 하면 됩니다.

가장 중요한 단계는 **어떤 심볼의 어떤 타임프레임이 어떤 파일에 해당하는지**를 딕셔너리 형태로 명확하게 알려주는 것입니다.

```python
import sys
from pathlib import Path
from datetime import datetime

# --- QuantBT 라이브러리 임포트 ---
from quantbt import (
    BacktestEngine,
    SimpleBroker,
    BacktestConfig,
    CSVDataProvider,  # UpbitDataProvider 대신 CSVDataProvider를 임포트합니다.
)

# 1. 데이터 파일 경로 설정
# 이 코드는 프로젝트의 루트 디렉토리에서 실행하는 것을 가정합니다.
# 'data' 폴더를 만들고 그 안에 CSV 파일들을 위치시키세요.
data_path = Path("data")
data_files = {
    "KRW-BTC": {
        "1d": str(data_path / "KRW-BTC_1d.csv"),
        # 만약 분봉 데이터도 있다면 아래와 같이 추가할 수 있습니다.
        # "1m": str(data_path / "KRW-BTC_1m.csv") 
    },
    "KRW-ETH": {
        "1d": str(data_path / "KRW-ETH_1d.csv")
    },
}

# 2. CSVDataProvider 인스턴스 생성
csv_provider = CSVDataProvider(
    data_files=data_files,
    timestamp_column="date"  # CSV 파일의 날짜 컬럼명을 지정합니다.
)

# 3. 백테스트 엔진에 데이터 프로바이더 설정
# 기존: engine.set_data_provider(UpbitDataProvider())
# 변경: engine.set_data_provider(csv_provider)

# ... (이하 백테스트 설정 및 실행 코드는 동일)
```

## 3. 자동 리샘플링 기능

`CSVDataProvider`는 편리한 자동 리샘플링 기능을 제공합니다.

만약 `data_files` 설정에서 특정 심볼의 일봉(`1d`) 데이터는 제공하지 않았지만, 분봉(`1m`) 데이터는 제공했을 경우, 백테스트에서 해당 심볼의 일봉 데이터를 요청하면 **자동으로 분봉 데이터를 집계하여 일봉 데이터를 생성**해줍니다. 이를 통해 더 유연하게 데이터를 관리하고 다양한 타임프레임 전략을 테스트할 수 있습니다.

## 4. 전체 예제 코드

`CSVDataProvider`를 사용하여 멀티 심볼 전략을 실행하는 전체 코드는 아래 예제 파일에서 확인하실 수 있습니다.

🔗 **전체 예제 확인하기: [`quantbt/examples/00_csv_dataloader.py`](../examples/00_csv_dataloader.py)**

이제 `CSVDataProvider`를 사용하여 자신만의 소중한 데이터를 QuantBT 백테스팅에 자유롭게 활용해 보세요! 