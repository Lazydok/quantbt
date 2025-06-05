# 📊 QuantBT 기본 전략 튜토리얼 실행 결과

이 문서는 [기본 전략 튜토리얼 노트북](../examples/simple_strategy_tutorial.ipynb)의 실행 결과를 보여줍니다.

## 🎯 환경 설정 및 모듈 임포트

```python
# 프로젝트 루트를 Python 경로에 추가
import sys
import os
from pathlib import Path

# 현재 노트북의 위치에서 프로젝트 루트 찾기
current_dir = Path.cwd()
if 'examples' in str(current_dir):
    # examples 폴더에서 실행하는 경우
    project_root = current_dir.parent.parent
else:
    # 프로젝트 루트에서 실행하는 경우
    project_root = current_dir

print(f"📁 현재 디렉토리: {current_dir}")
print(f"📁 프로젝트 루트: {project_root}")

# 프로젝트 루트를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ Python 경로에 프로젝트 루트 추가: {project_root}")

# 필요한 모듈 가져오기
from typing import List, Dict, Any, Optional
import polars as pl

try:
    from quantbt.core.interfaces.strategy import TradingStrategy, BacktestContext
    from quantbt.core.entities.market_data import MarketDataBatch
    from quantbt.core.entities.order import Order, OrderType, OrderSide
    from quantbt.core.entities.trade import Trade
    print("✅ 모든 QuantBT 모듈이 성공적으로 가져와졌습니다!")
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print("💡 해결 방법:")
    print("   1. 프로젝트 루트에서 노트북을 실행하세요")
    print("   2. 또는 다음 명령어로 패키지를 설치하세요: pip install -e .")
    raise
```

**실행 결과:**
```
📁 현재 디렉토리: /home/lazydok/src/quantbt/quantbt/examples
📁 프로젝트 루트: /home/lazydok/src/quantbt
✅ Python 경로에 프로젝트 루트 추가: /home/lazydok/src/quantbt
✅ 모든 QuantBT 모듈이 성공적으로 가져와졌습니다!
```

---

## 1️⃣ 바이 앤 홀드 전략 실행 결과

```python
# 전략 인스턴스 생성
buy_hold = BuyAndHoldStrategy()
print(f"📋 전략명: {buy_hold.name}")
print(f"💰 포지션 크기: {buy_hold.position_size_pct * 100}%")
print(f"📈 최대 포지션 수: {buy_hold.max_positions}")
```

**실행 결과:**
```
🏠 바이 앤 홀드 전략이 초기화되었습니다.
📋 전략명: BuyAndHoldStrategy
💰 포지션 크기: 100.0%
📈 최대 포지션 수: 10
```

### 특징 분석
- ✅ **가장 단순한 전략**: 지표 계산이 필요 없음
- ✅ **전체 자본 활용**: 100% 포지션으로 최대 수익 추구
- ✅ **거래 비용 최소화**: 한 번 매수 후 보유
- ⚠️ **하락장 취약**: 시장 하락 시 손실 감수

---

## 2️⃣ 이동평균 교차 전략 실행 결과

```python
# 전략 인스턴스 생성 및 테스트
sma_strategy = SimpleMovingAverageCrossStrategy(short_window=5, long_window=20)
print(f"📋 전략명: {sma_strategy.name}")
print(f"⚙️ 설정: {sma_strategy.config}")
print(f"💰 포지션 크기: {sma_strategy.position_size_pct * 100}%")
print(f"📊 필요 지표: {sma_strategy.indicator_columns}")
```

**실행 결과:**
```
📈 이동평균 교차 전략 초기화 (단기: 5일, 장기: 20일)
📋 전략명: SimpleMovingAverageCrossStrategy
⚙️ 설정: {'short_window': 5, 'long_window': 20}
💰 포지션 크기: 20.0%
📊 필요 지표: ['sma_5', 'sma_20']
```

### 특징 분석
- ✅ **트렌드 추종**: 상승 트렌드에서 강력한 성과
- ✅ **리스크 분산**: 20% 포지션으로 위험 관리
- ✅ **명확한 신호**: 골든/데드 크로스로 진입/청산 결정
- ⚠️ **횡보장 취약**: 잦은 가짜 신호로 손실 가능

---

## 3️⃣ RSI 전략 실행 결과

```python
# 전략 인스턴스 생성 및 테스트
rsi_strategy = RSIStrategy(rsi_period=14, oversold=25, overbought=75)
print(f"📋 전략명: {rsi_strategy.name}")
print(f"⚙️ 설정: {rsi_strategy.config}")
print(f"💰 포지션 크기: {rsi_strategy.position_size_pct * 100}%")
print(f"📊 필요 지표: {rsi_strategy.indicator_columns}")
```

**실행 결과:**
```
📈 RSI 전략 초기화 (기간: 14일, 과매도: 25, 과매수: 75)
📋 전략명: RSIStrategy
⚙️ 설정: {'rsi_period': 14, 'oversold': 25, 'overbought': 75}
💰 포지션 크기: 15.0%
📊 필요 지표: ['rsi']
```

### 특징 분석
- ✅ **평균 회귀**: 극단적 가격에서 반전 포착
- ✅ **보수적 접근**: 15% 포지션으로 안정성 추구
- ✅ **변동성 활용**: 높은 변동성에서 수익 기회 증가
- ⚠️ **강한 트렌드 불리**: 지속적 상승/하락에서 기회 상실

---

## 4️⃣ 랜덤 전략 실행 결과

```python
# 전략 인스턴스 생성 및 테스트
random_strategy = RandomStrategy(trade_probability=0.05)  # 5% 확률로 거래
print(f"📋 전략명: {random_strategy.name}")
print(f"⚙️ 설정: {random_strategy.config}")
print(f"💰 포지션 크기: {random_strategy.position_size_pct * 100}%")
print(f"🎲 거래 확률: {random_strategy.trade_probability * 100}%")
```

**실행 결과:**
```
🎲 랜덤 전략 초기화 (거래 확률: 5.0%)
📋 전략명: RandomStrategy
⚙️ 설정: {'trade_probability': 0.05}
💰 포지션 크기: 10.0%
🎲 거래 확률: 5.0%
```

### 특징 분석
- ✅ **편향 없음**: 순수한 랜덤으로 인간 편향 제거
- ✅ **벤치마크 역할**: 다른 전략 성과 비교 기준
- ✅ **최소 리스크**: 10% 포지션으로 위험 최소화
- ⚠️ **수익성 없음**: 장기적으로 0% 수익률 수렴

---

## 📊 전략 비교표 실행 결과

```python
import pandas as pd

# 전략 비교표 생성
strategies_comparison = {
    '전략명': ['Buy & Hold', 'SMA Cross', 'RSI', 'Random'],
    '타입': ['추세추종', '추세추종', '평균회귀', '랜덤'],
    '포지션크기': ['100%', '20%', '15%', '10%'],
    '최대포지션': [10, 5, 5, 3],
    '주요지표': ['없음', 'SMA(10,30)', 'RSI(14)', '없음'],
    '거래빈도': ['매우낮음', '낮음', '중간', '랜덤'],
    '장점': ['단순함, 낮은수수료', '트렌드 포착', '변동성 활용', '편향 없음'],
    '단점': ['하락장 취약', '횡보장 취약', '강한추세시 불리', '수익성 없음']
}

df_comparison = pd.DataFrame(strategies_comparison)
print("📊 전략 비교표")
print("=" * 100)
print(df_comparison.to_string(index=False))
print("=" * 100)
```

**실행 결과:**
```
📊 전략 비교표
====================================================================================================
    전략명     타입 포지션크기  최대포지션     주요지표   거래빈도              장점               단점
Buy & Hold  추세추종    100%        10       없음   매우낮음      단순함, 낮은수수료        하락장 취약
 SMA Cross  추세추종     20%         5  SMA(10,30)     낮음          트렌드 포착        횡보장 취약
       RSI  평균회귀     15%         5    RSI(14)     중간          변동성 활용  강한추세시 불리
    Random     랜덤     10%         3       없음     랜덤           편향 없음       수익성 없음
====================================================================================================
```

---

## 🚀 전략 데모 실행 결과

```python
# 백테스팅 실행 예제 (실제 실행을 위해서는 데이터와 엔진 설정 필요)

def demo_strategy_usage():
    """전략 사용법 데모"""
    
    print("🚀 QuantBT 전략 사용 예제")
    print("=" * 50)
    
    # 1. 전략들 생성
    strategies = {
        'conservative': BuyAndHoldStrategy(),
        'trend_following': SimpleMovingAverageCrossStrategy(short_window=5, long_window=20),
        'mean_reversion': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        'benchmark': RandomStrategy(trade_probability=0.02)
    }
    
    # 2. 각 전략의 기본 정보 출력
    for strategy_type, strategy in strategies.items():
        print(f"\n📈 {strategy_type.upper()}:")
        print(f"   이름: {strategy.name}")
        print(f"   포지션 크기: {strategy.position_size_pct * 100}%")
        print(f"   최대 포지션: {strategy.max_positions}")
        if hasattr(strategy, 'indicator_columns') and strategy.indicator_columns:
            print(f"   필요 지표: {', '.join(strategy.indicator_columns)}")
    
    print("\n✅ 모든 전략이 성공적으로 초기화되었습니다!")
    print("\n💡 실제 백테스팅을 위해서는 다음이 필요합니다:")
    print("   - 데이터 프로바이더 (CSV, Upbit 등)")
    print("   - 백테스트 엔진 설정")
    print("   - 백테스트 설정 (기간, 초기자본 등)")
    
    return strategies

# 데모 실행
demo_strategies = demo_strategy_usage()
```

**실행 결과:**
```
🚀 QuantBT 전략 사용 예제
==================================================
🏠 바이 앤 홀드 전략이 초기화되었습니다.
📈 이동평균 교차 전략 초기화 (단기: 5일, 장기: 20일)
📈 RSI 전략 초기화 (기간: 14일, 과매도: 30, 과매수: 70)
🎲 랜덤 전략 초기화 (거래 확률: 2.0%)

📈 CONSERVATIVE:
   이름: BuyAndHoldStrategy
   포지션 크기: 100.0%
   최대 포지션: 10

📈 TREND_FOLLOWING:
   이름: SimpleMovingAverageCrossStrategy
   포지션 크기: 20.0%
   최대 포지션: 5
   필요 지표: sma_5, sma_20

📈 MEAN_REVERSION:
   이름: RSIStrategy
   포지션 크기: 15.0%
   최대 포지션: 5
   필요 지표: rsi

📈 BENCHMARK:
   이름: RandomStrategy
   포지션 크기: 10.0%
   최대 포지션: 3

✅ 모든 전략이 성공적으로 초기화되었습니다!

💡 실제 백테스팅을 위해서는 다음이 필요합니다:
   - 데이터 프로바이더 (CSV, Upbit 등)
   - 백테스트 엔진 설정
   - 백테스트 설정 (기간, 초기자본 등)
```

---

## 📈 성능 분석 요약

### 전략별 특성 분석

| 구분 | Buy & Hold | SMA Cross | RSI | Random |
|------|------------|-----------|-----|--------|
| **위험도** | 높음 | 중간 | 낮음 | 매우낮음 |
| **수익잠재력** | 높음 | 중간 | 중간 | 없음 |
| **복잡도** | 매우낮음 | 낮음 | 낮음 | 매우낮음 |
| **거래비용** | 매우낮음 | 중간 | 높음 | 중간 |
| **적용성** | 초보자 | 중급자 | 중급자 | 벤치마크 |

### 💡 시장별 추천 전략

- **🔥 강세장 (Bull Market)**: Buy & Hold → SMA Cross
- **📉 약세장 (Bear Market)**: RSI → Random (방어적)
- **↔️ 횡보장 (Sideways)**: RSI → Random
- **🌊 변동성 높음**: RSI → SMA Cross

---

## 🎯 다음 단계 가이드

1. **실제 백테스팅 실행**
   ```python
   # UpbitDataProvider로 암호화폐 데이터 백테스팅
   from quantbt import UpbitDataProvider, SimpleBacktestEngine
   
   config = BacktestConfig(
       symbols=["KRW-BTC", "KRW-ETH"],
       start_date=datetime(2023, 1, 1),
       end_date=datetime(2023, 12, 31)
   )
   ```

2. **멀티심볼 전략 학습**
   - [멀티심볼 전략 가이드](multi_symbol_guide.md) 참조

3. **고급 지표 활용**
   - MACD, Bollinger Bands, Stochastic 등

4. **리스크 관리 추가**
   - 손절매, 익절매 로직 구현

---

**📚 참고 자료**
- [QuantBT GitHub](https://github.com/lazydok/quantbt)
- [멀티타임프레임 가이드](multi_timeframe_guide.md)
- [업비트 프로바이더 가이드](upbit_provider_guide.md)

*이 튜토리얼이 도움이 되셨나요? ⭐ GitHub에서 스타를 눌러주세요!* 