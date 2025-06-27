"""
Multi-symbol strategy example using CSVDataProvider
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# --- Project root path setup ---
# Dynamically find project root based on script's current location.
try:
    # When running as a script
    current_dir = Path(__file__).resolve().parent
except NameError:
    # When running in interactive environment (e.g., Jupyter)
    current_dir = Path.cwd()

project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Import QuantBT library ---
from quantbt import (
    TradingStrategy,
    BacktestEngine,
    SimpleBroker,
    BacktestConfig,
    CSVDataProvider,  # Use CSVDataProvider instead of UpbitDataProvider
    Order,
    OrderSide,
    OrderType,
)

# === Multi-symbol SMA strategy definition ===
class MultiSymbolSMAStrategy(TradingStrategy):
    """
    Buy: Price above short SMA
    Sell: Price below long SMA
    """
    def __init__(self, buy_sma: int = 5, sell_sma: int = 10, symbols: List[str] = ["KRW-BTC", "KRW-ETH"]):
        super().__init__(
            name="MultiSymbolSMAStrategy",
            config={"buy_sma": buy_sma, "sell_sma": sell_sma},
            position_size_pct=0.8,
            max_positions=len(symbols), # Allow maximum positions equal to number of symbols
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.symbols = symbols

    def _compute_indicators_for_symbol(self, symbol_data):
        """Calculate moving average indicators per symbol (Polars vector operations)"""
        data = symbol_data.sort("timestamp")
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}")
        ])

    def generate_signals(self, current_data: Dict[str, Any]) -> List[Order]:
        """Generate signals based on Dict"""
        orders = []
        if not self.broker:
            return orders

        symbol = current_data['symbol']
        current_price = current_data['close']
        buy_sma = current_data.get(f'sma_{self.buy_sma}')
        sell_sma = current_data.get(f'sma_{self.sell_sma}')

        if buy_sma is None or sell_sma is None:
            return orders

        current_positions = self.get_current_positions()
        
        # Buy signal
        if current_price > buy_sma and symbol not in current_positions:
            portfolio_value = self.get_portfolio_value()
            # Allocate portion of assets to each symbol
            position_value = (portfolio_value / len(self.symbols)) * self.position_size_pct
            quantity = self.calculate_position_size(symbol, current_price, position_value)
            
            if quantity > 0:
                orders.append(Order(symbol, OrderSide.BUY, quantity, OrderType.MARKET))
                print(f"[{current_data['timestamp'].date()}] 📈 Buy signal: {symbol} @ {current_price:,.0f}")

        # Sell signal
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            orders.append(Order(symbol, OrderSide.SELL, current_positions[symbol], OrderType.MARKET))
            print(f"[{current_data['timestamp'].date()}] 📉 Sell signal: {symbol} @ {current_price:,.0f}")

        return orders

def main():
    """
    Main execution function
    """
    print("🚀 Starting multi-symbol backtest using CSV dataloader 🚀")
    
    # 1. CSV data provider setup
    print("🔄 Initializing data provider...")
    
    # --- Important ---
    # This script should be run from the project root directory.
    # Example: python quantbt/examples/00_csv_dataloader.py
    data_path = Path("data")
    data_files = {
        "KRW-BTC": {"1d": str(data_path / "KRW-BTC_1d.csv")},
        "KRW-ETH": {"1d": str(data_path / "KRW-ETH_1d.csv")},
    }

    # Create CSVDataProvider instance
    try:
        csv_provider = CSVDataProvider(data_files=data_files, timestamp_column="date")
        print("✅ Data provider initialization complete.")
        print(f"   Available symbols: {csv_provider.get_symbols()}")
    except ValueError as e:
        print(f"❌ Data provider initialization failed: {e}")
        print("   Please check if KRW-BTC_1d.csv, KRW-ETH_1d.csv files exist in 'data' folder.")
        return

    # 2. Backtesting configuration
    config = BacktestConfig(
        symbols=["KRW-BTC", "KRW-ETH"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        timeframe="1d",
        initial_cash=10_000_000,
        commission_rate=0.0005,
        slippage_rate=0.0001,
        save_portfolio_history=True
    )
    print("⚙️  Backtest configuration complete.")
    print(f"   - Period: {config.start_date.date()} ~ {config.end_date.date()}")
    print(f"   - Initial capital: {config.initial_cash:,.0f} KRW")

    # 3. Strategy and broker initialization
    strategy = MultiSymbolSMAStrategy(
        buy_sma=5,
        sell_sma=10,
        symbols=config.symbols
    )
    broker = SimpleBroker(
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate
    )
    print("📈 Strategy and broker ready.")

    # 4. Backtest engine setup and execution
    engine = BacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(csv_provider)
    engine.set_broker(broker)
    
    print("\n⏳ Running backtest...")
    result = engine.run(config)
    print("✅ Backtest complete!")

    # 5. Results output
    print("\n" + "="*50)
    print("                 Backtest Results Summary")
    print("="*50)
    result.print_summary()
    
    # 6. Visualization (Optional)
    # result.plot_portfolio_performance()
    # result.plot_returns_distribution()

if __name__ == "__main__":
    main()
