"""
백테스팅 결과

백테스팅 실행 결과를 담는 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np

from ..entities.trade import Trade
from ..entities.position import Portfolio
from .backtest_config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    """백테스팅 결과"""
    
    # 기본 정보
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    
    # 성과 지표
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 최종 상태
    final_portfolio: Portfolio
    final_equity: float
    
    # 상세 데이터 (선택적)
    trades: Optional[List[Trade]] = None
    equity_curve: Optional[pl.DataFrame] = None
    
    # 벤치마크 데이터 (save_portfolio_history=True일 때 수집)
    benchmark_equity_curve: Optional[pl.DataFrame] = None
    benchmark_returns: Optional[pl.DataFrame] = None
    
    # 상세 분석용 추가 데이터 (save_portfolio_history=True일 때 수집)
    daily_returns: Optional[pl.DataFrame] = None
    monthly_returns: Optional[pl.DataFrame] = None
    drawdown_periods: Optional[pl.DataFrame] = None
    trade_signals: Optional[pl.DataFrame] = None  # 매수/매도 시그널
    
    # 추가 메타데이터
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """실행 시간 (초)"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.final_equity - self.config.initial_cash
    
    @property
    def total_return_pct(self) -> float:
        """총 수익률 (%)"""
        return self.total_return * 100
    
    @property
    def annual_return_pct(self) -> float:
        """연간 수익률 (%)"""
        return self.annual_return * 100
    
    @property
    def volatility_pct(self) -> float:
        """변동성 (%)"""
        return self.volatility * 100
    
    @property
    def max_drawdown_pct(self) -> float:
        """최대 낙폭 (%)"""
        return self.max_drawdown * 100
    
    @property
    def win_rate_pct(self) -> float:
        """승률 (%)"""
        return self.win_rate * 100
    
    # 시각화 관련 메서드들
    def plot_portfolio_performance(self, 
                                 figsize: tuple = (15, 10), 
                                 show_benchmark: bool = True,
                                 show_drawdown: bool = True,
                                 show_signals: bool = True,
                                 resample_to_daily: bool = True) -> None:
        """포트폴리오 성과 시각화"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("시각화를 위해 plotly를 설치해주세요: pip install plotly")
            return
        
        # 시각화 데이터 검증
        if not self._check_visualization_data():
            return
        
        # 서브플롯 구성 (제목 제거, 간격 축소)
        rows = 3 if show_drawdown else 2
        subplot_titles = ["Portfolio Value"]
        row_heights = [0.7, 0.3] if not show_drawdown else [0.6, 0.2, 0.2]
        
        if show_drawdown:
            subplot_titles.extend(["", ""])  # 제목 제거
        else:
            subplot_titles.append("")  # 제목 제거
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.02,  # 간격 축소
            row_heights=row_heights
        )
        
        # 1. 포트폴리오 가치 차트 (실제 평가금액)
        dates = self.equity_curve["timestamp"].to_list()
        equity = self.equity_curve["equity"].to_list()
        
        if len(dates) != len(equity):
            return
        
        # 데이터 리샘플링 (일 단위)
        if resample_to_daily:
            dates, equity = self._resample_to_daily(dates, equity)
        
        # 실제 포트폴리오 평가금액 표시 (정규화 없음)
        fig.add_trace(
            go.Scatter(
                x=dates, y=equity,
                name="Portfolio",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # 벤치마크 추가 (실제 금액)
        benchmark_equity_values = None
        dates_for_benchmark = dates
        if show_benchmark and self.benchmark_equity_curve is not None:
            benchmark_dates = self.benchmark_equity_curve["timestamp"].to_list()
            benchmark_equity = self.benchmark_equity_curve["equity"].to_list()
            
            # 벤치마크 데이터 검증
            if len(benchmark_equity) == 0:
                show_benchmark = False
            else:
                # 벤치마크 데이터 리샘플링 (일 단위)
                if resample_to_daily:
                    benchmark_dates, benchmark_equity = self._resample_to_daily(benchmark_dates, benchmark_equity)
                
                # 벤치마크와 포트폴리오 데이터 길이 맞추기
                if len(benchmark_equity) != len(dates):
                    # 짧은 길이에 맞춤
                    min_length = min(len(benchmark_equity), len(dates))
                    benchmark_equity_values = benchmark_equity[:min_length]
                    dates_for_benchmark = benchmark_dates[:min_length]
                    
                    # 포트폴리오 데이터도 맞춤
                    if len(dates) > min_length:
                        dates = dates[:min_length]
                        equity = equity[:min_length]
                else:
                    benchmark_equity_values = benchmark_equity
                    dates_for_benchmark = benchmark_dates
                
                if show_benchmark:  # 검증 통과 시에만 표시
                    fig.add_trace(
                        go.Scatter(
                            x=dates_for_benchmark, y=benchmark_equity_values,
                            name="Benchmark",
                            line=dict(color="gray", width=2, dash="dot")
                        ),
                        row=1, col=1
                    )
        
        # 매수/매도 시그널 추가 (벤치마크 기준으로 변경)
        if show_signals and self.trade_signals is not None:
            buy_signals = self.trade_signals.filter(pl.col("signal") == "BUY")
            sell_signals = self.trade_signals.filter(pl.col("signal") == "SELL")
            
            # 시그널 표시를 위한 기준 값 선택 (벤치마크 우선, 없으면 포트폴리오)
            if show_benchmark and benchmark_equity_values is not None:
                reference_values = benchmark_equity_values
                reference_dates = dates_for_benchmark
            else:
                reference_values = equity
                reference_dates = dates
            
            # 날짜-인덱스 매핑 생성 (timezone 호환성 처리)
            date_to_index = {}
            for i, date in enumerate(reference_dates):
                # timezone 정규화
                if date.tzinfo is not None:
                    date_key = date.replace(tzinfo=None)
                else:
                    date_key = date
                date_to_index[date_key] = i
            
            # timezone 안전 거리 계산 함수
            def find_closest_date_index(target_date, dates_list):
                min_diff = float('inf')
                closest_idx = 0
                
                # target_date timezone 정규화
                if target_date.tzinfo is not None:
                    target_normalized = target_date.replace(tzinfo=None)
                else:
                    target_normalized = target_date
                
                for i, date in enumerate(dates_list):
                    # date timezone 정규화
                    if date.tzinfo is not None:
                        date_normalized = date.replace(tzinfo=None)
                    else:
                        date_normalized = date
                    
                    try:
                        diff = abs((date_normalized - target_normalized).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            closest_idx = i
                    except:
                        continue
                
                return closest_idx
            
            if len(buy_signals) > 0:
                buy_dates = buy_signals["timestamp"].to_list()
                
                # 중복 제거: 동일 날짜의 신호는 하나만 유지 (리샘플링 최적화)
                from collections import OrderedDict
                unique_buy_signals = OrderedDict()
                
                for buy_date in buy_dates:
                    # timezone 정규화
                    buy_date_key = buy_date.replace(tzinfo=None) if buy_date.tzinfo is not None else buy_date
                    
                    # 일자별로 그룹핑 (시:분:초 제거하여 날짜만 사용)
                    date_only = buy_date_key.date() if hasattr(buy_date_key, 'date') else buy_date_key
                    
                    # 동일 날짜의 첫 번째 신호만 유지
                    if date_only not in unique_buy_signals:
                        unique_buy_signals[date_only] = buy_date
                
                # 중복 제거된 매수 신호 처리
                buy_dates_unique = list(unique_buy_signals.values())
                buy_y_values = []
                
                for buy_date in buy_dates_unique:
                    # timezone 정규화 후 검색
                    buy_date_key = buy_date.replace(tzinfo=None) if buy_date.tzinfo is not None else buy_date
                    
                    if buy_date_key in date_to_index:
                        idx = date_to_index[buy_date_key]
                        # 인덱스 범위 검증 강화
                        if 0 <= idx < len(reference_values):
                            buy_y_values.append(reference_values[idx])
                        else:
                            # 안전한 대체값 사용 (마지막 값)
                            buy_y_values.append(reference_values[-1] if reference_values else 1.0)
                    else:
                        # 가장 가까운 날짜 찾기
                        closest_idx = find_closest_date_index(buy_date, reference_dates)
                        # 인덱스 범위 검증 강화
                        if 0 <= closest_idx < len(reference_values):
                            buy_y_values.append(reference_values[closest_idx])
                        else:
                            # 안전한 대체값 사용
                            buy_y_values.append(reference_values[-1] if reference_values else 1.0)
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates_unique,
                        y=buy_y_values,
                        mode="markers",
                        name="Buy",
                        marker=dict(color="green", size=8, symbol="triangle-up")
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                sell_dates = sell_signals["timestamp"].to_list()
                
                # 중복 제거: 동일 날짜의 신호는 하나만 유지 (리샘플링 최적화)
                unique_sell_signals = OrderedDict()
                
                for sell_date in sell_dates:
                    # timezone 정규화
                    sell_date_key = sell_date.replace(tzinfo=None) if sell_date.tzinfo is not None else sell_date
                    
                    # 일자별로 그룹핑 (시:분:초 제거하여 날짜만 사용)
                    date_only = sell_date_key.date() if hasattr(sell_date_key, 'date') else sell_date_key
                    
                    # 동일 날짜의 첫 번째 신호만 유지
                    if date_only not in unique_sell_signals:
                        unique_sell_signals[date_only] = sell_date
                
                # 중복 제거된 매도 신호 처리
                sell_dates_unique = list(unique_sell_signals.values())
                sell_y_values = []
                
                for sell_date in sell_dates_unique:
                    # timezone 정규화 후 검색
                    sell_date_key = sell_date.replace(tzinfo=None) if sell_date.tzinfo is not None else sell_date
                    
                    if sell_date_key in date_to_index:
                        idx = date_to_index[sell_date_key]
                        # 인덱스 범위 검증 강화
                        if 0 <= idx < len(reference_values):
                            sell_y_values.append(reference_values[idx])
                        else:
                            # 안전한 대체값 사용 (마지막 값)
                            sell_y_values.append(reference_values[-1] if reference_values else 1.0)
                    else:
                        # 가장 가까운 날짜 찾기
                        closest_idx = find_closest_date_index(sell_date, reference_dates)
                        # 인덱스 범위 검증 강화
                        if 0 <= closest_idx < len(reference_values):
                            sell_y_values.append(reference_values[closest_idx])
                        else:
                            # 안전한 대체값 사용
                            sell_y_values.append(reference_values[-1] if reference_values else 1.0)
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates_unique,
                        y=sell_y_values,
                        mode="markers",
                        name="Sell",
                        marker=dict(color="red", size=8, symbol="triangle-down")
                    ),
                    row=1, col=1
                )
        
        # 2. 일간 수익률 차트 (막대그래프)
        if self.daily_returns is not None:
            returns_dates = self.daily_returns["timestamp"].to_list()
            returns_values = self.daily_returns["return"].to_list()
            
            # 리샘플링 적용
            if resample_to_daily:
                returns_dates, returns_values = self._resample_to_daily(returns_dates, returns_values, mode='mean')
            
            # 일간 수익률을 백분율로 변환
            returns_pct = [r * 100 for r in returns_values]
            
            # 수익률에 따른 색상 결정 (양수: 초록, 음수: 빨강)
            colors = ['green' if r >= 0 else 'red' for r in returns_pct]
            
            fig.add_trace(
                go.Bar(
                    x=returns_dates, y=returns_pct,
                    name="Daily Returns",
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False  # 범례에서 제거
                ),
                row=2, col=1
            )
        
        # 3. 드로다운 차트 (선택적)
        if show_drawdown and self.drawdown_periods is not None:
            drawdown_dates = self.drawdown_periods["timestamp"].to_list()
            drawdown_values = self.drawdown_periods["drawdown"].to_list()
            
            # 리샘플링 적용
            if resample_to_daily:
                drawdown_dates, drawdown_values = self._resample_to_daily(drawdown_dates, drawdown_values, mode='min')
            
            # 드로다운을 백분율로 변환
            drawdown_pct = [d * 100 for d in drawdown_values]
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown_dates, y=drawdown_pct,
                    name="Drawdown",
                    line=dict(color="red", width=1),
                    fill='tozeroy',
                    showlegend=False  # 범례에서 제거
                ),
                row=3, col=1
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title="Portfolio Performance Analysis",
            height=figsize[1] * 60,  # 높이 증가 (50 → 60)
            showlegend=True,
            hovermode='x unified'
        )
        
        # Y축 레이블 설정
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        # X축 설정 (하위 차트들의 x축 숨김)
        fig.update_xaxes(showticklabels=False, row=2, col=1)  # Daily Returns x축 숨김
        if show_drawdown:
            fig.update_xaxes(showticklabels=False, row=3, col=1)  # Drawdown x축 숨김
            # 마지막 차트(Drawdown)에만 x축 레이블 표시
            fig.update_xaxes(showticklabels=True, title_text="Date", row=3, col=1)
        else:
            # Drawdown이 없으면 마지막 차트(Daily Returns)에 x축 레이블 표시
            fig.update_xaxes(showticklabels=True, title_text="Date", row=2, col=1)
        
        fig.show()
    
    def plot_returns_distribution(self, 
                                period: str = "daily",
                                bins: int = 50,
                                figsize: tuple = (12, 8),
                                resample_to_daily: bool = True) -> None:
        """
        Returns distribution histogram (Result Chart 2)
        Args:
            period: Choose from 'daily', 'weekly', 'monthly'
            bins: Number of histogram bins
        """
        if not self._check_visualization_data():
            return
        
        try:
            import plotly.graph_objects as go
            import plotly.figure_factory as ff
        except ImportError:
            print("Please install plotly for visualization: pip install plotly")
            return
        
        # Select returns data
        if period == "monthly" and self.monthly_returns is not None:
            returns = self.monthly_returns["return"].to_numpy()
            title = "Monthly Returns Distribution"
        elif period == "daily" and self.daily_returns is not None:
            returns = self.daily_returns["return"].to_numpy()
            title = "Daily Returns Distribution"
        else:
            print(f"{period} returns data is not available.")
            return
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=bins,
            name="Returns Distribution",
            marker_color="lightblue",
            opacity=0.7
        ))
        
        # Add normal distribution curve
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Prevent division by zero
        if std_return > 0:
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                         np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        else:
            # Don't draw normal distribution curve when std is 0
            x_range = np.array([])
            normal_dist = np.array([])
        
        # Add normal distribution curve only if calculated
        if len(x_range) > 0:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist * len(returns) * (returns.max() - returns.min()) / bins,
                mode="lines",
                name="Normal Distribution",
                line=dict(color="red", width=2)
            ))
        
        # Add statistical information
        fig.add_annotation(
            x=0.7, y=0.9,
            xref="paper", yref="paper",
            text=f"Mean: {mean_return:.4f}<br>" +
                 f"Std Dev: {std_return:.4f}<br>" +
                 f"Skewness: {self._calculate_skewness(returns):.2f}<br>" +
                 f"Kurtosis: {self._calculate_kurtosis(returns):.2f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template="plotly_white",
            height=500
        )
        
        fig.show()
    
    def plot_monthly_returns_heatmap(self, figsize: tuple = (12, 8), resample_to_daily: bool = True) -> None:
        """월별 수익률 히트맵 (결과 차트3)"""
        if not self._check_visualization_data() or self.monthly_returns is None:
            print("월별 수익률 데이터가 없습니다.")
            return
        
        try:
            import plotly.graph_objects as go
            import pandas as pd
        except ImportError:
            print("시각화를 위해 plotly와 pandas를 설치해주세요")
            return
        
        # 월별 데이터를 연도-월 매트릭스로 변환
        monthly_df = self.monthly_returns.to_pandas()
        monthly_df['year'] = monthly_df['timestamp'].dt.year
        monthly_df['month'] = monthly_df['timestamp'].dt.month
        
        # 피벗 테이블 생성
        heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')
        
        # 월 이름 매핑 (영어)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # 연도를 정수 형태로 변환
        year_labels = [str(int(year)) for year in heatmap_data.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[month_names[i-1] for i in heatmap_data.columns],
            y=year_labels,
            colorscale='RdYlGn',
            zmin=-0.2,  # -20% 고정
            zmax=0.2,   # +20% 고정
            text=np.round(heatmap_data.values * 100, 2),
            texttemplate="%{text}%",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len([month_names[i-1] for i in heatmap_data.columns]))),
                ticktext=[month_names[i-1] for i in heatmap_data.columns]
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(year_labels))),
                ticktext=year_labels,
                # 연도 표시를 깔끔하게
                type='category'
            )
        )
        
        fig.show()
    
    def show_performance_comparison(self, benchmark_name: str = "Benchmark") -> None:
        """Performance comparison table with benchmark (Result Chart 4) - 안전성 강화"""
        if not self._check_visualization_data():
            return
        
        try:
            import pandas as pd
        except ImportError:
            print("📊 성과 비교표를 보려면 pandas를 설치해주세요: pip install pandas")
            return
        
        try:
            # Calculate benchmark metrics with safety check
            benchmark_metrics = {}
            if self.benchmark_returns is not None:
                benchmark_metrics = self._calculate_benchmark_metrics()
            
            # Create comparison data
            comparison_data = {
                "Metric": [
                    "Total Return (%)",
                    "Annual Return (%)", 
                    "Volatility (%)",
                    "Sharpe Ratio",
                    "Calmar Ratio",
                    "Sortino Ratio",
                    "Max Drawdown (%)",
                    "Beta",
                    "Alpha",
                    "Total Trades",
                    "Win Rate (%)",
                    "Profit Factor",
                    # "Avg Holding Period (days)",
                    # "Max Consecutive Wins",
                    # "Max Consecutive Losses"
                ],
                "Strategy": [
                    f"{self.total_return_pct:.2f}",
                    f"{self.annual_return_pct:.2f}",
                    f"{self.volatility_pct:.2f}",
                    f"{self.sharpe_ratio:.2f}",
                    f"{self._calculate_calmar_ratio():.2f}",
                    f"{self._calculate_sortino_ratio():.2f}",
                    f"{self.max_drawdown_pct:.2f}",
                    f"{self._calculate_beta():.2f}",
                    f"{self._calculate_alpha():.2f}",
                    f"{self.total_trades}",
                    f"{self.win_rate_pct:.1f}",
                    f"{self.profit_factor:.2f}",
                    # f"{self._calculate_avg_holding_period():.1f}",
                    # f"{self._calculate_max_consecutive_wins()}",
                    # f"{self._calculate_max_consecutive_losses()}"
                ]
            }
            
            # Add benchmark data only if successfully calculated
            if benchmark_metrics:
                comparison_data[benchmark_name] = [
                    f"{benchmark_metrics.get('total_return_pct', 0):.2f}",
                    f"{benchmark_metrics.get('annual_return_pct', 0):.2f}",
                    f"{benchmark_metrics.get('volatility_pct', 0):.2f}",
                    f"{benchmark_metrics.get('sharpe_ratio', 0):.2f}",
                    f"{benchmark_metrics.get('calmar_ratio', 0):.2f}",
                    f"{benchmark_metrics.get('sortino_ratio', 0):.2f}",
                    f"{benchmark_metrics.get('max_drawdown_pct', 0):.2f}",
                    "1.00",  # Benchmark beta is always 1
                    "0.00",  # Benchmark alpha is always 0
                    "-", "-", "-",
                    # "-", "-", "-"  # Trade-related metrics not applicable (6개)
                ]
            else:
                print("⚠️ 벤치마크 데이터가 없거나 계산에 실패했습니다. 전략 성과만 표시합니다.")
            
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Check if we're in a Jupyter notebook environment
            try:
                # Try to import and use Jupyter display functions
                from IPython.display import display, HTML
                from IPython import get_ipython
                
                # Check if we're actually in a notebook environment
                if get_ipython() is not None and get_ipython().__class__.__name__ in ['ZMQInteractiveShell', 'TerminalInteractiveShell']:
                    # Jupyter notebook environment - use rich display with styling
                    if benchmark_name in comparison_data:
                        def highlight_better(row):
                            if row.name < 9:  # Performance metrics
                                try:
                                    strategy_val = float(row['Strategy'].replace('%', ''))
                                    if len(row) > 2:  # If benchmark data exists
                                        benchmark_val = float(row.iloc[2].replace('%', ''))
                                        
                                        # Color based on better value with high contrast text
                                        if row.name in [2, 6]:  # 변동성, Max drawdown (lower is better)
                                            if strategy_val < benchmark_val:
                                                return ['', 
                                                       'background-color: #d4edda; color: #155724; font-weight: bold',  # Strategy better
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold']  # Benchmark worse
                                            else:
                                                return ['', 
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold',  # Strategy worse
                                                       'background-color: #d4edda; color: #155724; font-weight: bold']  # Benchmark better
                                        elif row.name in [7]: # 베타는 -0.5 ~ 0.5 어느정도 시장 중립이라 생각
                                            if strategy_val > -0.5 and strategy_val < 0.5:
                                                return ['', 
                                                       'background-color: #d4edda; color: #155724; font-weight: bold',  # Strategy better
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold']  # Benchmark worse
                                            else:
                                                return ['', 
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold',  # Strategy worse
                                                       'background-color: #d4edda; color: #155724; font-weight: bold']  # Benchmark better
                                        else:  # Other metrics (higher is better)
                                            if strategy_val > benchmark_val:
                                                return ['', 
                                                       'background-color: #d4edda; color: #155724; font-weight: bold',  # Strategy better
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold']  # Benchmark worse
                                            else:
                                                return ['', 
                                                       'background-color: #f8d7da; color: #721c24; font-weight: bold',  # Strategy worse
                                                       'background-color: #d4edda; color: #155724; font-weight: bold']  # Benchmark better
                                except Exception:
                                    pass
                            
                            return [''] * len(row)
                        
                        styled_df = df.style.apply(highlight_better, axis=1)
                    else:
                        styled_df = df.style
                    
                    # Display in Jupyter notebook
                    display(HTML("<h3>📊 Strategy Performance Comparison</h3>"))
                    display(styled_df)
                    return
            except:
                pass
            
            # Fallback to text output for non-Jupyter environments
            print("\n" + "=" * 60)
            print("                📊 Strategy Performance Comparison")
            print("=" * 60)
            
            # Calculate column widths for better formatting
            metric_width = max(len(str(metric)) for metric in comparison_data["Metric"])
            strategy_width = max(len(str(val)) for val in comparison_data["Strategy"])
            
            if benchmark_name in comparison_data:
                benchmark_width = max(len(str(val)) for val in comparison_data[benchmark_name])
                header_format = f"{{:<{metric_width}}} | {{:>{strategy_width}}} | {{:>{benchmark_width}}}"
                row_format = f"{{:<{metric_width}}} | {{:>{strategy_width}}} | {{:>{benchmark_width}}}"
                
                # Print header
                print(header_format.format("Metric", "Strategy", benchmark_name))
                print("-" * (metric_width + strategy_width + benchmark_width + 6))
                
                # Print data rows
                for i, metric in enumerate(comparison_data["Metric"]):
                    strategy_val = comparison_data["Strategy"][i]
                    benchmark_val = comparison_data[benchmark_name][i]
                    print(row_format.format(metric, strategy_val, benchmark_val))
            else:
                header_format = f"{{:<{metric_width}}} | {{:>{strategy_width}}}"
                row_format = f"{{:<{metric_width}}} | {{:>{strategy_width}}}"
                
                # Print header
                print(header_format.format("Metric", "Strategy"))
                print("-" * (metric_width + strategy_width + 3))
                
                # Print data rows
                for i, metric in enumerate(comparison_data["Metric"]):
                    strategy_val = comparison_data["Strategy"][i]
                    print(row_format.format(metric, strategy_val))
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error occurred while generating performance comparison: {e}")
            print("Displaying basic performance summary instead:")
            self.print_summary()
    
    def _resample_to_daily(self, dates, values, mode='last'):
        """
        데이터를 일 단위로 리샘플링
        
        Args:
            dates: 날짜 리스트
            values: 값 리스트
            mode: 리샘플링 방식 ('last', 'first', 'mean', 'max', 'min')
        
        Returns:
            tuple: (리샘플링된 날짜 리스트, 리샘플링된 값 리스트)
        """
        try:
            import pandas as pd
        except ImportError:
            print("⚠️ 리샘플링을 위해 pandas가 필요합니다. 원본 데이터를 반환합니다.")
            return dates, values
        
        if len(dates) == 0 or len(values) == 0:
            return dates, values
        
        # DataFrame 생성
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        # timestamp를 datetime으로 변환 (timezone 처리)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # DataFrame을 timestamp 기준으로 정렬
        df = df.sort_values('timestamp')
        
        # 중복 제거 (같은 날짜의 마지막 값 사용)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # 날짜별로 그룹화하여 리샘플링
        df['date'] = df['timestamp'].dt.date
        
        # 리샘플링 방식에 따라 집계
        if mode == 'last':
            daily_df = df.groupby('date').agg({
                'timestamp': 'last',
                'value': 'last'
            }).reset_index()
        elif mode == 'first':
            daily_df = df.groupby('date').agg({
                'timestamp': 'first', 
                'value': 'first'
            }).reset_index()
        elif mode == 'mean':
            daily_df = df.groupby('date').agg({
                'timestamp': 'last',  # 날짜는 마지막 시점
                'value': 'mean'
            }).reset_index()
        elif mode == 'max':
            daily_df = df.groupby('date').agg({
                'timestamp': 'last',
                'value': 'max'
            }).reset_index()
        elif mode == 'min':
            daily_df = df.groupby('date').agg({
                'timestamp': 'last',
                'value': 'min'
            }).reset_index()
        else:
            daily_df = df.groupby('date').agg({
                'timestamp': 'last',
                'value': 'last'
            }).reset_index()
        
        # timestamp 정렬
        daily_df = daily_df.sort_values('timestamp')
        
        return daily_df['timestamp'].tolist(), daily_df['value'].tolist()
    
    def _check_visualization_data(self) -> bool:
        """시각화에 필요한 데이터가 있는지 확인"""
        if not self.config.save_portfolio_history:
            return False
        
        if self.equity_curve is None:
            return False
        
        return True
    
    def _calculate_benchmark_metrics(self) -> Dict[str, float]:
        """벤치마크 성과 지표 계산 - 이미 계산된 benchmark_returns 활용 (메타데이터 캐싱)"""
        if self.benchmark_returns is None:
            return {}
        
        # 메타데이터에 캐시된 결과가 있으면 재사용
        if (self.metadata and 
            '_cached_benchmark_metrics' in self.metadata and 
            self.metadata['_cached_benchmark_metrics']):
            return self.metadata['_cached_benchmark_metrics']
        
        try:
            returns = self.benchmark_returns["return"].to_numpy()
            
            # NaN 값 제거
            valid_returns = returns[~np.isnan(returns)]
            if len(valid_returns) == 0:
                return {}
            
            # 총 수익률 계산 (복리)
            total_return = (1 + valid_returns).prod() - 1
            
            # 실제 백테스팅 기간을 사용하여 연간 수익률 계산
            actual_years = (self.config.end_date - self.config.start_date).days / 365.25
            
            # 연간 수익률: 실제 백테스팅 기간 기준으로 정확히 계산
            if actual_years > 0 and total_return > -1:  # -100% 미만 손실 방지
                annual_return = (1 + total_return) ** (1/actual_years) - 1
            else:
                annual_return = 0
            
            # 변동성 계산
            volatility = valid_returns.std() * np.sqrt(365.25) if len(valid_returns) > 1 else 0
            
            # 샤프 비율 (무위험 수익률 0% 가정)
            risk_free_rate = 0.00
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 드로다운 계산 (이미 계산된 벤치마크 equity curve 활용)
            if self.benchmark_equity_curve is not None:
                # 벤치마크 equity curve에서 드로다운 계산 (최적화)
                equity_values = self.benchmark_equity_curve["equity"].to_numpy()
                if len(equity_values) > 1:
                    running_max = np.maximum.accumulate(equity_values)
                    drawdown = (equity_values - running_max) / running_max
                    max_drawdown = drawdown.min()  # 이미 음수값
                else:
                    max_drawdown = 0
            else:
                # fallback: returns에서 계산
                cumulative = (1 + valid_returns).cumprod()
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()  # 이미 음수값
            
            # 소르티노 비율 계산
            negative_returns = valid_returns[valid_returns < 0]
            downside_std = negative_returns.std() * np.sqrt(365.25) if len(negative_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # 칼마 비율 계산 (연간수익률 / |최대낙폭|)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            result = {
                'total_return_pct': total_return * 100,
                'annual_return_pct': annual_return * 100,
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': abs(max_drawdown) * 100,  # 절댓값으로 양수 표시
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            }
            
            # 계산 결과를 메타데이터에 캐싱 (frozen=True 대응)
            if self.metadata is not None:
                # metadata가 변경 가능한 경우에만 캐싱
                try:
                    self.metadata['_cached_benchmark_metrics'] = result
                except:
                    pass  # frozen으로 인해 실패해도 계속 진행
            
            return result
            
        except (ValueError, ZeroDivisionError, FloatingPointError) as e:
            print(f"⚠️ 벤치마크 지표 계산 중 오류 발생: {e}")
            return {}
    
    def _calculate_calmar_ratio(self) -> float:
        """칼마 비율 계산"""
        if self.max_drawdown == 0:
            return 0
        return self.annual_return / abs(self.max_drawdown)
    
    def _calculate_sortino_ratio(self) -> float:
        """소르티노 비율 계산 - 이미 계산된 daily_returns 활용"""
        if self.daily_returns is None:
            return 0
        
        try:
            returns = self.daily_returns["return"].to_numpy()
            
            # NaN 값 제거
            valid_returns = returns[~np.isnan(returns)]
            if len(valid_returns) == 0:
                return 0
            
            negative_returns = valid_returns[valid_returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_std = negative_returns.std() * np.sqrt(365.25)
            return self.annual_return / downside_std if downside_std > 0 else 0
        except Exception:
            return 0
    
    def _calculate_beta(self) -> float:
        """베타 계산 - 이미 계산된 daily_returns와 benchmark_returns 활용"""
        if self.daily_returns is None or self.benchmark_returns is None:
            return 0
        
        try:
            strategy_returns = self.daily_returns["return"].to_numpy()
            benchmark_returns = self.benchmark_returns["return"].to_numpy()
            
            # 배열 유효성 검사
            if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
                return 0
            
            # 길이 다를 경우 짧은 쪽에 맞춤
            min_length = min(len(strategy_returns), len(benchmark_returns))
            if min_length == 0:
                return 0
                
            strategy_returns = strategy_returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
            
            # NaN 값 제거
            valid_mask = ~(np.isnan(strategy_returns) | np.isnan(benchmark_returns))
            if not np.any(valid_mask):
                return 0
            
            strategy_returns = strategy_returns[valid_mask]
            benchmark_returns = benchmark_returns[valid_mask]
            
            # 최소 데이터 점수 확인
            if len(strategy_returns) < 2:
                return 0
            
            # 공분산 및 분산 계산
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 1e-10:  # 매우 작은 값도 0으로 처리
                return covariance / benchmark_variance
            else:
                return 0
                
        except (ZeroDivisionError, FloatingPointError, IndexError, ValueError):
            return 0
    
    def _calculate_alpha(self) -> float:
        """알파 계산 - 벤치마크 지표를 일회성으로 계산하여 활용"""
        if self.benchmark_returns is None or self.daily_returns is None:
            return 0
        
        try:
            # 벤치마크 연간 수익률을 계산 (캐싱 시도하되 실패해도 계속 진행)
            benchmark_metrics = self._calculate_benchmark_metrics()
            
            if not benchmark_metrics:
                return 0
            
            benchmark_annual_return = benchmark_metrics.get('annual_return_pct', 0) / 100.0
            
            # 베타 계산 (이미 최적화된 메서드 사용)
            beta = self._calculate_beta()
            
            # 무위험 수익률 (0% 가정)
            risk_free_rate = 0.00
            
            # 알파 = 전략수익률 - [무위험수익률 + 베타 × (벤치마크수익률 - 무위험수익률)]
            alpha = self.annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            return alpha
            
        except (ZeroDivisionError, FloatingPointError, ValueError):
            return 0
    
    def _calculate_avg_holding_period(self) -> float:
        """평균 보유 기간 계산"""
        if not self.trades:
            return 0
        
        # Trade 객체에는 exit_time이 없으므로 임시로 0 반환
        # TODO: Trade 클래스에 exit_time 속성 추가 또는 다른 방식으로 계산
        return 0
        
        # holding_periods = []
        # for trade in self.trades:
        #     if hasattr(trade, 'exit_time') and trade.exit_time:
        #         days = (trade.exit_time - trade.entry_time).days
        #         holding_periods.append(days)
        # 
        # return np.mean(holding_periods) if holding_periods else 0
    
    def _calculate_max_consecutive_wins(self) -> int:
        """최대 연속 승리 횟수"""
        if not self.trades:
            return 0
        
        # Trade 객체에는 pnl 속성이 없으므로 임시로 0 반환
        # TODO: 포지션 기반 손익 계산 로직 구현 필요
        return 0
        
        # max_wins = 0
        # current_wins = 0
        # 
        # for trade in self.trades:
        #     # Trade 클래스에 pnl 속성이 없음 - pnl_impact는 개별 거래 영향만 나타냄
        #     if hasattr(trade, 'pnl') and trade.pnl > 0:
        #         current_wins += 1
        #         max_wins = max(max_wins, current_wins)
        #     else:
        #         current_wins = 0
        # 
        # return max_wins
    
    def _calculate_max_consecutive_losses(self) -> int:
        """최대 연속 패배 횟수"""
        if not self.trades:
            return 0
        
        # Trade 객체에는 pnl 속성이 없으므로 임시로 0 반환
        # TODO: 포지션 기반 손익 계산 로직 구현 필요
        return 0
        
        # max_losses = 0
        # current_losses = 0
        # 
        # for trade in self.trades:
        #     # Trade 클래스에 pnl 속성이 없음 - pnl_impact는 개별 거래 영향만 나타냄
        #     if hasattr(trade, 'pnl') and trade.pnl < 0:
        #         current_losses += 1
        #         max_losses = max(max_losses, current_losses)
        #     else:
        #         current_losses = 0
        # 
        # return max_losses
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """왜도 계산"""
        try:
            mean = np.mean(returns)
            std = np.std(returns)
            if std <= 1e-10:  # 매우 작은 값도 0으로 처리
                return 0
            return np.mean(((returns - mean) / std) ** 3)
        except (ZeroDivisionError, FloatingPointError):
            return 0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """첨도 계산"""
        try:
            mean = np.mean(returns)
            std = np.std(returns)
            if std <= 1e-10:  # 매우 작은 값도 0으로 처리
                return 0
            return np.mean(((returns - mean) / std) ** 4) - 3
        except (ZeroDivisionError, FloatingPointError):
            return 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            
            # 성과 지표
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annual_return": self.annual_return,
            "annual_return_pct": self.annual_return_pct,
            "volatility": self.volatility,
            "volatility_pct": self.volatility_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self._calculate_calmar_ratio(),
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            
            # 거래 통계
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "win_rate_pct": self.win_rate_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            
            # 최종 상태
            "final_portfolio": self.final_portfolio.to_dict(),
            "final_equity": self.final_equity,
            "total_pnl": self.total_pnl,
            
            # 메타데이터
            "metadata": self.metadata or {}
        }
        
        # 거래 내역 포함 (선택적)
        if self.trades:
            result["trades"] = [trade.to_dict() for trade in self.trades]
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Return summary information"""
        return {
            "Period": f"{self.config.start_date.date()} ~ {self.config.end_date.date()}",
            "Initial Capital": f"${self.config.initial_cash:,.0f}",
            "Final Equity": f"${self.final_equity:,.0f}",
            "Total Return": f"{self.total_return_pct:.2f}%",
            "Annual Return": f"{self.annual_return_pct:.2f}%",
            "Volatility": f"{self.volatility_pct:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Calmar Ratio": f"{self._calculate_calmar_ratio():.2f}",
            "Sortino Ratio": f"{self._calculate_sortino_ratio():.2f}",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate_pct:.1f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Execution Time": f"{self.duration:.2f}s"
        }
    
    def print_summary(self) -> None:
        """Print backtest summary"""
        print("=" * 60)
        print("                 BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key:16}: {value}")
        
        print("=" * 60) 