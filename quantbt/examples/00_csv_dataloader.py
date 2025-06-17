"""
CSVDataProvider를 사용한 멀티 심볼 전략 예제
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# --- 프로젝트 루트 경로 설정 ---
# 스크립트의 현재 위치를 기준으로 프로젝트 루트를 동적으로 찾습니다.
try:
    # 스크립트로 실행될 때
    current_dir = Path(__file__).resolve().parent
except NameError:
    # 대화형 환경 (예: Jupyter)에서 실행될 때
    current_dir = Path.cwd()

project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- QuantBT 라이브러리 임포트 ---
from quantbt import (
    TradingStrategy,
    BacktestEngine,
    SimpleBroker,
    BacktestConfig,
    CSVDataProvider,  # UpbitDataProvider 대신 CSVDataProvider 사용
    Order,
    OrderSide,
    OrderType,
)

# === 멀티 심볼 SMA 전략 정의 ===
class MultiSymbolSMAStrategy(TradingStrategy):
    """
    매수: 가격이 짧은 SMA를 상회
    매도: 가격이 긴 SMA를 하회
    """
    def __init__(self, buy_sma: int = 5, sell_sma: int = 10, symbols: List[str] = ["KRW-BTC", "KRW-ETH"]):
        super().__init__(
            name="MultiSymbolSMAStrategy",
            config={"buy_sma": buy_sma, "sell_sma": sell_sma},
            position_size_pct=0.8,
            max_positions=len(symbols), # 심볼 개수만큼 최대 포지션 허용
        )
        self.buy_sma = buy_sma
        self.sell_sma = sell_sma
        self.symbols = symbols

    def _compute_indicators_for_symbol(self, symbol_data):
        """심볼별 이동평균 지표 계산 (Polars 벡터 연산)"""
        data = symbol_data.sort("timestamp")
        buy_sma = self.calculate_sma(data["close"], self.buy_sma)
        sell_sma = self.calculate_sma(data["close"], self.sell_sma)
        return data.with_columns([
            buy_sma.alias(f"sma_{self.buy_sma}"),
            sell_sma.alias(f"sma_{self.sell_sma}")
        ])

    def generate_signals_dict(self, current_data: Dict[str, Any]) -> List[Order]:
        """Dict 기반 신호 생성"""
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
        
        # 매수 신호
        if current_price > buy_sma and symbol not in current_positions:
            portfolio_value = self.get_portfolio_value()
            # 각 심볼에 자산의 일부를 할당
            position_value = (portfolio_value / len(self.symbols)) * self.position_size_pct
            quantity = self.calculate_position_size(symbol, current_price, position_value)
            
            if quantity > 0:
                orders.append(Order(symbol, OrderSide.BUY, quantity, OrderType.MARKET))
                print(f"[{current_data['timestamp'].date()}] 📈 매수 신호: {symbol} @ {current_price:,.0f}")

        # 매도 신호
        elif current_price < sell_sma and symbol in current_positions and current_positions[symbol] > 0:
            orders.append(Order(symbol, OrderSide.SELL, current_positions[symbol], OrderType.MARKET))
            print(f"[{current_data['timestamp'].date()}] 📉 매도 신호: {symbol} @ {current_price:,.0f}")

        return orders

def main():
    """
    메인 실행 함수
    """
    print("🚀 CSV 데이터로더를 사용한 멀티 심볼 백테스트 시작 🚀")
    
    # 1. CSV 데이터 프로바이더 설정
    print("🔄 데이터 프로바이더 초기화 중...")
    
    # --- 중요 ---
    # 이 스크립트는 프로젝트 루트 디렉토리에서 실행되어야 합니다.
    # 예: python quantbt/examples/00_csv_dataloader.py
    data_path = Path("data")
    data_files = {
        "KRW-BTC": {"1d": str(data_path / "KRW-BTC_1d.csv")},
        "KRW-ETH": {"1d": str(data_path / "KRW-ETH_1d.csv")},
    }

    # CSVDataProvider 인스턴스 생성
    try:
        csv_provider = CSVDataProvider(data_files=data_files, timestamp_column="date")
        print("✅ 데이터 프로바이더 초기화 완료.")
        print(f"   사용 가능 심볼: {csv_provider.get_symbols()}")
    except ValueError as e:
        print(f"❌ 데이터 프로바이더 초기화 실패: {e}")
        print("   'data' 폴더에 KRW-BTC_1d.csv, KRW-ETH_1d.csv 파일이 있는지 확인하세요.")
        return

    # 2. 백테스팅 설정
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
    print("⚙️  백테스트 설정 완료.")
    print(f"   - 기간: {config.start_date.date()} ~ {config.end_date.date()}")
    print(f"   - 초기 자본: {config.initial_cash:,.0f}원")

    # 3. 전략 및 브로커 초기화
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
    print("📈 전략 및 브로커 준비 완료.")

    # 4. 백테스트 엔진 설정 및 실행
    engine = BacktestEngine()
    engine.set_strategy(strategy)
    engine.set_data_provider(csv_provider)
    engine.set_broker(broker)
    
    print("\n⏳ 백테스트 실행 중...")
    result = engine.run(config)
    print("✅ 백테스트 완료!")

    # 5. 결과 출력
    print("\n" + "="*50)
    print("                 백테스트 결과 요약")
    print("="*50)
    result.print_summary()
    
    # 6. 시각화 (Optional)
    # result.plot_portfolio_performance()
    # result.plot_returns_distribution()

if __name__ == "__main__":
    main()
