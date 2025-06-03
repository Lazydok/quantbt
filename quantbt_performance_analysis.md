# QuantBT 성과 분석 모듈

## 백테스팅 결과 분석의 중요성

백테스팅 결과 분석은 전략의 실제 성능을 평가하고 리스크를 파악하는 핵심 과정입니다. 단순한 수익률 이외에도 다양한 지표를 통해 전략의 안정성, 일관성, 위험 조정 수익률 등을 종합적으로 평가해야 합니다.

## 핵심 성과 지표

### 1. 수익률 지표

```python
# quantbt/analysis/metrics/returns.py
import numpy as np
import polars as pl
from typing import Optional, Tuple
from datetime import datetime

class ReturnMetrics:
    """수익률 관련 지표"""
    
    @staticmethod
    def total_return(returns: pl.Series) -> float:
        """총 수익률"""
        return (1 + returns).product() - 1
    
    @staticmethod
    def annualized_return(returns: pl.Series, trading_days: int = 252) -> float:
        """연율화 수익률"""
        n_periods = len(returns)
        total_ret = ReturnMetrics.total_return(returns)
        return (1 + total_ret) ** (trading_days / n_periods) - 1
    
    @staticmethod
    def cagr(start_value: float, end_value: float, years: float) -> float:
        """복리연평균성장률 (CAGR)"""
        return (end_value / start_value) ** (1 / years) - 1
    
    @staticmethod
    def monthly_returns(returns: pl.Series, dates: pl.Series) -> pl.DataFrame:
        """월별 수익률"""
        df = pl.DataFrame({
            "date": dates,
            "returns": returns
        })
        
        return df.group_by_dynamic(
            "date", 
            every="1mo", 
            closed="right"
        ).agg(
            pl.col("returns").map_elements(
                lambda x: (1 + x).product() - 1
            ).alias("monthly_return")
        )
    
    @staticmethod
    def rolling_returns(
        returns: pl.Series, 
        window: int = 252
    ) -> pl.Series:
        """롤링 수익률"""
        return returns.rolling_map(
            lambda x: (1 + x).product() - 1,
            window_size=window
        )
```

### 2. 리스크 지표

```python
# quantbt/analysis/metrics/risk.py
import numpy as np
import polars as pl
from typing import Optional

class RiskMetrics:
    """리스크 관련 지표"""
    
    @staticmethod
    def volatility(returns: pl.Series, trading_days: int = 252) -> float:
        """연율화 변동성"""
        return returns.std() * np.sqrt(trading_days)
    
    @staticmethod
    def maximum_drawdown(equity_curve: pl.Series) -> Tuple[float, int, int]:
        """최대 낙폭 (MDD)"""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.arg_min()
        
        # 고점 찾기
        peak_idx = running_max[:max_dd_idx + 1].arg_max()
        
        return abs(max_dd), peak_idx, max_dd_idx
    
    @staticmethod
    def value_at_risk(returns: pl.Series, confidence: float = 0.05) -> float:
        """VaR (Value at Risk)"""
        return returns.quantile(confidence)
    
    @staticmethod
    def conditional_var(returns: pl.Series, confidence: float = 0.05) -> float:
        """CVaR (Conditional Value at Risk)"""
        var = RiskMetrics.value_at_risk(returns, confidence)
        return returns.filter(returns <= var).mean()
    
    @staticmethod
    def downside_deviation(
        returns: pl.Series, 
        target_return: float = 0.0,
        trading_days: int = 252
    ) -> float:
        """하방 편차"""
        downside_returns = returns.filter(returns < target_return)
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(trading_days)
    
    @staticmethod
    def beta(
        portfolio_returns: pl.Series, 
        benchmark_returns: pl.Series
    ) -> float:
        """베타 (시장 대비 민감도)"""
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        market_variance = benchmark_returns.var()
        return covariance / market_variance if market_variance != 0 else 0.0
    
    @staticmethod
    def tracking_error(
        portfolio_returns: pl.Series,
        benchmark_returns: pl.Series,
        trading_days: int = 252
    ) -> float:
        """트래킹 에러"""
        excess_returns = portfolio_returns - benchmark_returns
        return excess_returns.std() * np.sqrt(trading_days)
```

### 3. 위험 조정 수익률 지표

```python
# quantbt/analysis/metrics/risk_adjusted.py
import numpy as np
import polars as pl

class RiskAdjustedMetrics:
    """위험 조정 수익률 지표"""
    
    @staticmethod
    def sharpe_ratio(
        returns: pl.Series,
        risk_free_rate: float = 0.02,
        trading_days: int = 252
    ) -> float:
        """샤프 비율"""
        excess_returns = returns.mean() * trading_days - risk_free_rate
        volatility = returns.std() * np.sqrt(trading_days)
        return excess_returns / volatility if volatility != 0 else 0.0
    
    @staticmethod
    def sortino_ratio(
        returns: pl.Series,
        target_return: float = 0.02,
        trading_days: int = 252
    ) -> float:
        """소르티노 비율"""
        excess_return = returns.mean() * trading_days - target_return
        downside_dev = RiskMetrics.downside_deviation(returns, target_return/trading_days)
        return excess_return / downside_dev if downside_dev != 0 else 0.0
    
    @staticmethod
    def calmar_ratio(returns: pl.Series, equity_curve: pl.Series) -> float:
        """칼마 비율 (연율화 수익률 / MDD)"""
        annual_return = ReturnMetrics.annualized_return(returns)
        max_dd, _, _ = RiskMetrics.maximum_drawdown(equity_curve)
        return annual_return / max_dd if max_dd != 0 else 0.0
    
    @staticmethod
    def information_ratio(
        portfolio_returns: pl.Series,
        benchmark_returns: pl.Series
    ) -> float:
        """정보 비율"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = RiskMetrics.tracking_error(portfolio_returns, benchmark_returns)
        return excess_returns.mean() / tracking_error if tracking_error != 0 else 0.0
    
    @staticmethod
    def alpha(
        portfolio_returns: pl.Series,
        benchmark_returns: pl.Series,
        risk_free_rate: float = 0.02,
        trading_days: int = 252
    ) -> float:
        """알파 (초과 수익률)"""
        beta = RiskMetrics.beta(portfolio_returns, benchmark_returns)
        portfolio_return = portfolio_returns.mean() * trading_days
        market_return = benchmark_returns.mean() * trading_days
        
        return portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
```

### 4. 거래 분석 지표

```python
# quantbt/analysis/metrics/trading.py
import polars as pl
from typing import Dict, Any

class TradingMetrics:
    """거래 관련 지표"""
    
    @staticmethod
    def win_rate(trades: pl.DataFrame) -> float:
        """승률"""
        profitable_trades = trades.filter(pl.col("pnl") > 0)
        return len(profitable_trades) / len(trades) if len(trades) > 0 else 0.0
    
    @staticmethod
    def profit_factor(trades: pl.DataFrame) -> float:
        """손익비 (총 수익 / 총 손실)"""
        profits = trades.filter(pl.col("pnl") > 0)["pnl"].sum()
        losses = abs(trades.filter(pl.col("pnl") < 0)["pnl"].sum())
        return profits / losses if losses != 0 else float('inf')
    
    @staticmethod
    def average_win_loss(trades: pl.DataFrame) -> Dict[str, float]:
        """평균 수익/손실"""
        winning_trades = trades.filter(pl.col("pnl") > 0)["pnl"]
        losing_trades = trades.filter(pl.col("pnl") < 0)["pnl"]
        
        return {
            "avg_win": winning_trades.mean() if len(winning_trades) > 0 else 0.0,
            "avg_loss": losing_trades.mean() if len(losing_trades) > 0 else 0.0,
            "win_loss_ratio": (
                winning_trades.mean() / abs(losing_trades.mean()) 
                if len(losing_trades) > 0 and losing_trades.mean() != 0 
                else 0.0
            )
        }
    
    @staticmethod
    def max_consecutive_wins_losses(trades: pl.DataFrame) -> Dict[str, int]:
        """최대 연속 승/패"""
        pnl_signs = (trades["pnl"] > 0).cast(pl.Int32)
        
        # 연속 승리 계산
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for is_win in pnl_signs:
            if is_win:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses
        }
    
    @staticmethod
    def trade_statistics(trades: pl.DataFrame) -> Dict[str, Any]:
        """종합 거래 통계"""
        return {
            "total_trades": len(trades),
            "winning_trades": len(trades.filter(pl.col("pnl") > 0)),
            "losing_trades": len(trades.filter(pl.col("pnl") < 0)),
            "win_rate": TradingMetrics.win_rate(trades),
            "profit_factor": TradingMetrics.profit_factor(trades),
            **TradingMetrics.average_win_loss(trades),
            **TradingMetrics.max_consecutive_wins_losses(trades),
            "largest_win": trades["pnl"].max(),
            "largest_loss": trades["pnl"].min(),
            "total_pnl": trades["pnl"].sum()
        }
```

## 성과 분석 리포트 클래스

```python
# quantbt/analysis/performance_report.py
import polars as pl
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metrics.returns import ReturnMetrics
from .metrics.risk import RiskMetrics
from .metrics.risk_adjusted import RiskAdjustedMetrics
from .metrics.trading import TradingMetrics

class PerformanceReport:
    """백테스팅 성과 분석 리포트"""
    
    def __init__(
        self,
        account_history: pl.DataFrame,
        trade_history: pl.DataFrame,
        benchmark_data: Optional[pl.DataFrame] = None,
        risk_free_rate: float = 0.02
    ):
        self.account_history = account_history
        self.trade_history = trade_history
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        
        # 수익률 계산
        self.returns = self._calculate_returns()
        self.benchmark_returns = self._calculate_benchmark_returns()
        
    def _calculate_returns(self) -> pl.Series:
        """일별 수익률 계산"""
        return self.account_history.select(
            pl.col("total_value").pct_change().alias("returns")
        ).drop_nulls()["returns"]
    
    def _calculate_benchmark_returns(self) -> Optional[pl.Series]:
        """벤치마크 수익률 계산"""
        if self.benchmark_data is None:
            return None
        return self.benchmark_data.select(
            pl.col("close").pct_change().alias("returns")
        ).drop_nulls()["returns"]
    
    def generate_summary(self) -> Dict[str, Any]:
        """종합 성과 요약"""
        equity_curve = self.account_history["total_value"]
        
        # 기본 수익률 지표
        total_return = ReturnMetrics.total_return(self.returns)
        annual_return = ReturnMetrics.annualized_return(self.returns)
        
        # 리스크 지표
        volatility = RiskMetrics.volatility(self.returns)
        max_dd, peak_idx, trough_idx = RiskMetrics.maximum_drawdown(equity_curve)
        var_95 = RiskMetrics.value_at_risk(self.returns, 0.05)
        cvar_95 = RiskMetrics.conditional_var(self.returns, 0.05)
        
        # 위험 조정 수익률
        sharpe = RiskAdjustedMetrics.sharpe_ratio(self.returns, self.risk_free_rate)
        sortino = RiskAdjustedMetrics.sortino_ratio(self.returns, self.risk_free_rate)
        calmar = RiskAdjustedMetrics.calmar_ratio(self.returns, equity_curve)
        
        summary = {
            # 수익률 지표
            "total_return": total_return,
            "annualized_return": annual_return,
            "volatility": volatility,
            
            # 리스크 지표
            "max_drawdown": max_dd,
            "var_95": var_95,
            "cvar_95": cvar_95,
            
            # 위험 조정 수익률
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            
            # 기간 정보
            "start_date": self.account_history["timestamp"].min(),
            "end_date": self.account_history["timestamp"].max(),
            "total_days": len(self.account_history),
        }
        
        # 벤치마크 비교 (있는 경우)
        if self.benchmark_returns is not None:
            benchmark_annual = ReturnMetrics.annualized_return(self.benchmark_returns)
            beta = RiskMetrics.beta(self.returns, self.benchmark_returns)
            alpha = RiskAdjustedMetrics.alpha(
                self.returns, self.benchmark_returns, self.risk_free_rate
            )
            info_ratio = RiskAdjustedMetrics.information_ratio(
                self.returns, self.benchmark_returns
            )
            tracking_error = RiskMetrics.tracking_error(
                self.returns, self.benchmark_returns
            )
            
            summary.update({
                "benchmark_return": benchmark_annual,
                "excess_return": annual_return - benchmark_annual,
                "beta": beta,
                "alpha": alpha,
                "information_ratio": info_ratio,
                "tracking_error": tracking_error
            })
        
        # 거래 통계 (있는 경우)
        if len(self.trade_history) > 0:
            trade_stats = TradingMetrics.trade_statistics(self.trade_history)
            summary.update({"trading_stats": trade_stats})
        
        return summary
    
    def generate_monthly_returns_table(self) -> pl.DataFrame:
        """월별 수익률 테이블"""
        monthly_df = ReturnMetrics.monthly_returns(
            self.returns, 
            self.account_history["timestamp"]
        )
        
        return monthly_df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month")
        ]).pivot(
            index="year",
            columns="month", 
            values="monthly_return"
        )
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """시각화 차트 생성"""
        figs = {}
        
        # 1. 누적 수익률 곡선
        figs["equity_curve"] = self._create_equity_curve()
        
        # 2. 드로우다운 차트
        figs["drawdown"] = self._create_drawdown_chart()
        
        # 3. 월별 수익률 히트맵
        figs["monthly_heatmap"] = self._create_monthly_heatmap()
        
        # 4. 수익률 분포
        figs["return_distribution"] = self._create_return_distribution()
        
        # 5. 롤링 통계
        figs["rolling_stats"] = self._create_rolling_stats()
        
        return figs
    
    def _create_equity_curve(self) -> go.Figure:
        """누적 수익률 곡선"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value', 'Daily Returns'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # 포트폴리오 가치
        fig.add_trace(
            go.Scatter(
                x=self.account_history["timestamp"],
                y=self.account_history["total_value"],
                name="Portfolio Value",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # 벤치마크 (있는 경우)
        if self.benchmark_data is not None:
            normalized_benchmark = (
                self.benchmark_data["close"] / 
                self.benchmark_data["close"].first() *
                self.account_history["total_value"].first()
            )
            fig.add_trace(
                go.Scatter(
                    x=self.benchmark_data["timestamp"],
                    y=normalized_benchmark,
                    name="Benchmark",
                    line=dict(color="red", width=1, dash="dash")
                ),
                row=1, col=1
            )
        
        # 일별 수익률
        fig.add_trace(
            go.Scatter(
                x=self.account_history["timestamp"][1:],
                y=self.returns * 100,
                mode="lines",
                name="Daily Returns (%)",
                line=dict(color="gray", width=1)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            height=600
        )
        
        return fig
    
    def _create_drawdown_chart(self) -> go.Figure:
        """드로우다운 차트"""
        equity_curve = self.account_history["total_value"]
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=self.account_history["timestamp"],
                y=drawdown,
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red'),
                name='Drawdown (%)'
            )
        )
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        return fig
    
    def _create_monthly_heatmap(self) -> go.Figure:
        """월별 수익률 히트맵"""
        monthly_table = self.generate_monthly_returns_table()
        
        # 월별 데이터를 히트맵 형식으로 변환
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_table.select(pl.col("^[0-9]+$")).to_numpy() * 100,
            x=months,
            y=monthly_table["year"].to_list(),
            colorscale="RdYlGn",
            zmid=0,
            text=monthly_table.select(pl.col("^[0-9]+$")).to_numpy() * 100,
            texttemplate="%{text:.1f}%",
            colorbar=dict(title="Return (%)")
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            height=400
        )
        
        return fig
    
    def _create_return_distribution(self) -> go.Figure:
        """수익률 분포"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Return Distribution', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 히스토그램
        fig.add_trace(
            go.Histogram(
                x=self.returns * 100,
                nbinsx=50,
                name="Returns",
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Q-Q 플롯 (정규분포와 비교)
        sorted_returns = np.sort(self.returns * 100)
        normal_quantiles = np.random.normal(
            sorted_returns.mean(), 
            sorted_returns.std(), 
            len(sorted_returns)
        )
        normal_quantiles.sort()
        
        fig.add_trace(
            go.Scatter(
                x=normal_quantiles,
                y=sorted_returns,
                mode="markers",
                name="Q-Q Plot"
            ),
            row=1, col=2
        )
        
        # 45도 참조선
        min_val = min(normal_quantiles.min(), sorted_returns.min())
        max_val = max(normal_quantiles.max(), sorted_returns.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Normal Distribution",
                line=dict(dash="dash", color="red")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Return Analysis",
            height=400
        )
        
        return fig
    
    def _create_rolling_stats(self) -> go.Figure:
        """롤링 통계"""
        window = 252  # 1년
        
        rolling_sharpe = self.returns.rolling_map(
            lambda x: RiskAdjustedMetrics.sharpe_ratio(x, self.risk_free_rate/252),
            window_size=window
        )
        
        rolling_vol = self.returns.rolling_std(window) * np.sqrt(252)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.1
        )
        
        # 롤링 샤프 비율
        fig.add_trace(
            go.Scatter(
                x=self.account_history["timestamp"][window:],
                y=rolling_sharpe[window:],
                name="Rolling Sharpe",
                line=dict(color="blue")
            ),
            row=1, col=1
        )
        
        # 롤링 변동성
        fig.add_trace(
            go.Scatter(
                x=self.account_history["timestamp"][window:],
                y=rolling_vol[window:] * 100,
                name="Rolling Volatility (%)",
                line=dict(color="red")
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Rolling Statistics (1-Year Window)",
            height=500
        )
        
        return fig
    
    def export_report(self, output_path: str, format: str = "html") -> None:
        """리포트 내보내기"""
        summary = self.generate_summary()
        charts = self.create_visualizations()
        
        if format == "html":
            self._export_html_report(summary, charts, output_path)
        elif format == "pdf":
            self._export_pdf_report(summary, charts, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_html_report(
        self, 
        summary: Dict[str, Any], 
        charts: Dict[str, go.Figure],
        output_path: str
    ) -> None:
        """HTML 리포트 생성"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Performance Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E8B57; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Backtesting Performance Report</h1>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <div class="metric">
                    <div class="metric-value">{summary['total_return']:.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['annualized_return']:.2%}</div>
                    <div class="metric-label">Annual Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['sharpe_ratio']:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['max_drawdown']:.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['volatility']:.2%}</div>
                    <div class="metric-label">Volatility</div>
                </div>
            </div>
            
            {"".join([f'<div class="chart" id="{name}"></div>' for name in charts.keys()])}
            
            <script>
                {"".join([f"Plotly.newPlot('{name}', {fig.to_json()});" for name, fig in charts.items()])}
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
```

## 사용 예시

```python
# 백테스팅 결과 분석 예시
from quantbt.analysis.performance_report import PerformanceReport

# 백테스팅 실행 후
result = await engine.run()

# 성과 분석 리포트 생성
report = PerformanceReport(
    account_history=result.account_history,
    trade_history=result.trade_history,
    benchmark_data=benchmark_data,  # 벤치마크 데이터 (선택적)
    risk_free_rate=0.02
)

# 요약 통계
summary = report.generate_summary()
print(f"Total Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']:.2%}")

# 월별 수익률 테이블
monthly_returns = report.generate_monthly_returns_table()
print(monthly_returns)

# 시각화 차트
charts = report.create_visualizations()

# HTML 리포트 내보내기
report.export_report("backtest_report.html", format="html")
```

## 개선된 BacktestResult 클래스

```python
# quantbt/core/value_objects/backtest_result.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import polars as pl

from ..analysis.performance_report import PerformanceReport

@dataclass
class BacktestResult:
    """백테스팅 결과"""
    config: 'BacktestConfig'
    account_history: pl.DataFrame
    trade_history: pl.DataFrame
    
    # 성과 지표
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    
    # 선택적 데이터
    benchmark_data: Optional[pl.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def create_performance_report(self) -> PerformanceReport:
        """성과 분석 리포트 생성"""
        return PerformanceReport(
            account_history=self.account_history,
            trade_history=self.trade_history,
            benchmark_data=self.benchmark_data
        )
    
    def export_detailed_report(self, output_path: str) -> None:
        """상세 리포트 내보내기"""
        report = self.create_performance_report()
        report.export_report(output_path)
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """요약 딕셔너리 반환"""
        report = self.create_performance_report()
        return report.generate_summary()
```

이 성과 분석 모듈을 통해 백테스팅 결과를 **포괄적이고 전문적으로** 분석할 수 있습니다. 주요 개선사항:

1. **포괄적인 지표**: 수익률, 리스크, 위험조정수익률, 거래통계
2. **벤치마크 비교**: 알파, 베타, 트래킹에러 등
3. **시각화**: 인터랙티브 차트와 히트맵
4. **리포트 생성**: HTML/PDF 형태의 전문적인 리포트
5. **모듈화**: 각 지표별로 독립적인 클래스 구조 