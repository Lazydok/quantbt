# Guide: Using Your Own Data with Local CSV Files

All QuantBT examples are configured to use `UpbitDataProvider` by default. This data provider has the advantage of querying market data in real-time through the Upbit API and caching the data locally for very fast loading in subsequent runs.

However, there are times when you want to analyze assets not supported by Upbit or use data for specific periods in an offline environment. For such situations, you can use `CSVDataProvider` to utilize your own data for backtesting.

## 1. Preparing CSV Data

First, you need to prepare CSV files for backtesting. For `CSVDataProvider` to read data correctly, it's important that files follow the specified format.

### Data Format

-   **Required columns**: `date`, `open`, `high`, `low`, `close`, `volume`
-   **Date format**: `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`

Below is an example of daily bar data for `KRW-BTC`.

```csv
date,open,high,low,close,volume
2024-01-01,57000000,58000000,56000000,57500000,100
2024-01-02,57500000,58500000,57000000,58000000,120
2024-01-03,58000000,59000000,57800000,58800000,150
...
```

> **💡 Tip:** If your date column is not named `date` but something else like `timestamp` or `time`, you can specify it directly through the `timestamp_column` argument when creating `CSVDataProvider`.

## 2. `CSVDataProvider` Configuration and Usage

Once your CSV files are ready, you simply need to replace `UpbitDataProvider` with `CSVDataProvider` in your backtest code.

The most important step is to clearly specify **which symbol's which timeframe corresponds to which file** in dictionary format.

```python
import sys
from pathlib import Path
from datetime import datetime

# --- Import QuantBT library ---
from quantbt import (
    BacktestEngine,
    SimpleBroker,
    BacktestConfig,
    CSVDataProvider,  # Import CSVDataProvider instead of UpbitDataProvider
)

# 1. Set up data file paths
# This code assumes execution from the project's root directory.
# Create a 'data' folder and place your CSV files inside it.
data_path = Path("data")
data_files = {
    "KRW-BTC": {
        "1d": str(data_path / "KRW-BTC_1d.csv"),
        # If you also have minute data, you can add it like below:
        # "1m": str(data_path / "KRW-BTC_1m.csv") 
    },
    "KRW-ETH": {
        "1d": str(data_path / "KRW-ETH_1d.csv")
    },
}

# 2. Create CSVDataProvider instance
csv_provider = CSVDataProvider(
    data_files=data_files,
    timestamp_column="date"  # Specify the date column name in your CSV files
)

# 3. Set data provider in backtest engine
# Before: engine.set_data_provider(UpbitDataProvider())
# After: engine.set_data_provider(csv_provider)

# ... (rest of backtest configuration and execution code remains the same)
```

## 3. Automatic Resampling Feature

`CSVDataProvider` provides a convenient automatic resampling feature.

If you didn't provide daily (`1d`) data for a specific symbol in the `data_files` configuration but did provide minute (`1m`) data, when the backtest requests daily data for that symbol, it will **automatically aggregate minute data to generate daily data**. This allows for more flexible data management and testing of various timeframe strategies.

## 4. Complete Example Code

You can check the complete code for executing multi-symbol strategies using `CSVDataProvider` in the example file below.

🔗 **Check full example: [`quantbt/examples/00_csv_dataloader.py`](../examples/00_csv_dataloader.py)**

Now use `CSVDataProvider` to freely utilize your valuable data in QuantBT backtesting! 