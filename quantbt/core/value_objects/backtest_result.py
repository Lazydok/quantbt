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
    portfolio_history: Optional[pl.DataFrame] = None
    equity_curve: Optional[pl.DataFrame] = None
    
    # 벤치마크 데이터 (시각화 모드에서 수집)
    benchmark_equity_curve: Optional[pl.DataFrame] = None
    benchmark_returns: Optional[pl.DataFrame] = None
    
    # 시각화용 추가 데이터
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
                                 show_signals: bool = True) -> None:
        """
        포트폴리오 성과 시각화 (기본 결과 차트1)
        - 포트폴리오 평가금 차트 (정규화)
        - 벤치마크 비교 (정규화)
        - MDD 시각화
        - 매수/매도 시그널 표기 (정규화된 포트폴리오 값 기준)
        """
        if not self._check_visualization_data():
            return
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
        except ImportError:
            print("시각화를 위해 plotly를 설치해주세요: pip install plotly")
            return
        
        # 서브플롯 생성 (영어 제목)
        rows = 3 if show_drawdown else 2
        subplot_titles = ["Portfolio Value (Normalized)", "Daily Returns"]
        if show_drawdown:
            subplot_titles.append("Drawdown")
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,  # 여백 증가하여 겹침 방지
            row_heights=[0.5, 0.3, 0.2] if show_drawdown else [0.7, 0.3],
            specs=[[{"secondary_y": False}]] * rows
        )
        
        # 1. 포트폴리오 평가금 차트 (정규화)
        dates = self.equity_curve["timestamp"].to_list()
        equity = self.equity_curve["equity"].to_list()
        
        # 초기값 기준 정규화 (시작점 = 1.0)
        initial_equity = equity[0]
        normalized_equity = [val / initial_equity for val in equity]
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=normalized_equity,
                name="Portfolio",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # 벤치마크 추가 (정규화)
        normalized_benchmark = None
        if show_benchmark and self.benchmark_equity_curve is not None:
            benchmark_equity = self.benchmark_equity_curve["equity"].to_list()
            # 벤치마크도 동일한 기준으로 정규화
            initial_benchmark = benchmark_equity[0]
            normalized_benchmark = [val / initial_benchmark for val in benchmark_equity]
            fig.add_trace(
                go.Scatter(
                    x=dates, y=normalized_benchmark,
                    name="Benchmark",
                    line=dict(color="gray", width=1, dash="dash")
                ),
                row=1, col=1
            )
        
        # 매수/매도 시그널 추가 (정규화된 포트폴리오 값 기준)
        if show_signals and self.trade_signals is not None:
            buy_signals = self.trade_signals.filter(pl.col("signal") == "BUY")
            sell_signals = self.trade_signals.filter(pl.col("signal") == "SELL")
            
            # 날짜-인덱스 매핑 생성
            date_to_index = {date: i for i, date in enumerate(dates)}
            
            if len(buy_signals) > 0:
                buy_dates = buy_signals["timestamp"].to_list()
                # 해당 날짜의 정규화된 포트폴리오 값 찾기
                buy_y_values = []
                for buy_date in buy_dates:
                    if buy_date in date_to_index:
                        idx = date_to_index[buy_date]
                        buy_y_values.append(normalized_equity[idx])
                    else:
                        # 가장 가까운 날짜 찾기
                        closest_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - buy_date).total_seconds()))
                        buy_y_values.append(normalized_equity[closest_idx])
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_y_values,
                        mode="markers",
                        name="Buy",
                        marker=dict(color="green", size=8, symbol="triangle-up")
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                sell_dates = sell_signals["timestamp"].to_list()
                # 해당 날짜의 정규화된 포트폴리오 값 찾기
                sell_y_values = []
                for sell_date in sell_dates:
                    if sell_date in date_to_index:
                        idx = date_to_index[sell_date]
                        sell_y_values.append(normalized_equity[idx])
                    else:
                        # 가장 가까운 날짜 찾기
                        closest_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - sell_date).total_seconds()))
                        sell_y_values.append(normalized_equity[closest_idx])
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_y_values,
                        mode="markers",
                        name="Sell",
                        marker=dict(color="red", size=8, symbol="triangle-down")
                    ),
                    row=1, col=1
                )
        
        # 2. 일간 수익률
        if self.daily_returns is not None:
            returns = self.daily_returns["return"].to_list()
            colors = ["green" if r > 0 else "red" for r in returns]
            
            fig.add_trace(
                go.Bar(
                    x=dates[1:], y=returns,
                    name="Daily Returns",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 3. 드로다운
        if show_drawdown and self.drawdown_periods is not None:
            drawdown = self.drawdown_periods["drawdown"].to_list()
            drawdown_dates = self.drawdown_periods["timestamp"].to_list()
            fig.add_trace(
                go.Scatter(
                    x=drawdown_dates, y=drawdown,
                    name="Drawdown",
                    fill="tozeroy",
                    line=dict(color="red", width=1)
                ),
                row=3, col=1
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title="Backtesting Results - Portfolio Performance",
            height=600 if not show_drawdown else 800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Y축 레이블 설정
        fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return", row=2, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown", row=3, col=1)
        
        # # X축 레이블 설정
        # for i in range(1, rows + 1):
        #     fig.update_xaxes(title_text="Date", row=i, col=1)
        
        fig.show()
    
    def plot_returns_distribution(self, 
                                period: str = "daily",
                                bins: int = 50,
                                figsize: tuple = (12, 8)) -> None:
        """
        수익률 분포 히스토그램 (결과 차트2)
        Args:
            period: 'daily', 'weekly', 'monthly' 중 선택
            bins: 히스토그램 구간 수
        """
        if not self._check_visualization_data():
            return
        
        try:
            import plotly.graph_objects as go
            import plotly.figure_factory as ff
        except ImportError:
            print("시각화를 위해 plotly를 설치해주세요: pip install plotly")
            return
        
        # 수익률 데이터 선택
        if period == "monthly" and self.monthly_returns is not None:
            returns = self.monthly_returns["return"].to_numpy()
            title = "월간 수익률 분포"
        elif period == "daily" and self.daily_returns is not None:
            returns = self.daily_returns["return"].to_numpy()
            title = "일간 수익률 분포"
        else:
            print(f"{period} 수익률 데이터가 없습니다.")
            return
        
        # 히스토그램 생성
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=bins,
            name="수익률 분포",
            marker_color="lightblue",
            opacity=0.7
        ))
        
        # 정규분포 곡선 추가
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # division by zero 방지
        if std_return > 0:
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                         np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        else:
            # 표준편차가 0인 경우 정규분포 곡선을 그리지 않음
            x_range = np.array([])
            normal_dist = np.array([])
        
        # 정규분포 곡선이 계산된 경우에만 추가
        if len(x_range) > 0:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist * len(returns) * (returns.max() - returns.min()) / bins,
                mode="lines",
                name="정규분포",
                line=dict(color="red", width=2)
            ))
        
        # 통계 정보 추가
        fig.add_annotation(
            x=0.7, y=0.9,
            xref="paper", yref="paper",
            text=f"평균: {mean_return:.4f}<br>" +
                 f"표준편차: {std_return:.4f}<br>" +
                 f"왜도: {self._calculate_skewness(returns):.2f}<br>" +
                 f"첨도: {self._calculate_kurtosis(returns):.2f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="수익률",
            yaxis_title="빈도",
            template="plotly_white",
            height=500
        )
        
        fig.show()
    
    def plot_monthly_returns_heatmap(self, figsize: tuple = (12, 8)) -> None:
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
    
    def show_performance_comparison(self, benchmark_name: str = "벤치마크") -> None:
        """벤치마크와의 성과 비교 표 (결과 차트4)"""
        if not self._check_visualization_data():
            return
        
        try:
            import pandas as pd
            from IPython.display import display, HTML
        except ImportError:
            print("성과 비교를 위해 pandas와 IPython을 설치해주세요")
            return
        
        # 벤치마크 지표 계산
        benchmark_metrics = self._calculate_benchmark_metrics() if self.benchmark_returns is not None else {}
        
        # 비교 데이터 생성
        comparison_data = {
            "지표": [
                "총 수익률 (%)",
                "연간 수익률 (%)", 
                "변동성 (%)",
                "샤프 비율",
                "칼마 비율",
                "소르티노 비율",
                "최대 낙폭 (%)",
                "베타",
                "알파",
                "총 거래 횟수",
                "승률 (%)",
                "수익 인수",
                "평균 보유기간 (일)",
                "최대 연속 승리",
                "최대 연속 패배"
            ],
            "전략": [
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
                f"{self._calculate_avg_holding_period():.1f}",
                f"{self._calculate_max_consecutive_wins()}",
                f"{self._calculate_max_consecutive_losses()}"
            ]
        }
        
        # 벤치마크 데이터 추가
        if benchmark_metrics:
            comparison_data[benchmark_name] = [
                f"{benchmark_metrics.get('total_return_pct', 0):.2f}",
                f"{benchmark_metrics.get('annual_return_pct', 0):.2f}",
                f"{benchmark_metrics.get('volatility_pct', 0):.2f}",
                f"{benchmark_metrics.get('sharpe_ratio', 0):.2f}",
                f"{benchmark_metrics.get('calmar_ratio', 0):.2f}",
                f"{benchmark_metrics.get('sortino_ratio', 0):.2f}",
                f"{benchmark_metrics.get('max_drawdown_pct', 0):.2f}",
                "1.00",  # 벤치마크의 베타는 항상 1
                "0.00",  # 벤치마크의 알파는 항상 0
                "-", "-", "-", "-", "-", "-"  # 거래 관련 지표는 해당 없음
            ]
        
        # DataFrame 생성 및 표시
        df = pd.DataFrame(comparison_data)
        
        # 스타일링
        def highlight_better(row):
            if row.name < 9:  # 성과 지표들
                try:
                    strategy_val = float(row['전략'].replace('%', ''))
                    if len(row) > 2:  # 벤치마크 데이터가 있는 경우
                        benchmark_val = float(row.iloc[2].replace('%', ''))
                        
                        # 더 좋은 값에 따라 색상 결정
                        if row.name in [6]:  # 최대 낙폭 (낮을수록 좋음)
                            if strategy_val < benchmark_val:
                                return ['', 'background-color: lightgreen', 'background-color: lightcoral']
                            else:
                                return ['', 'background-color: lightcoral', 'background-color: lightgreen']
                        else:  # 나머지 지표들 (높을수록 좋음)
                            if strategy_val > benchmark_val:
                                return ['', 'background-color: lightgreen', 'background-color: lightcoral']
                            else:
                                return ['', 'background-color: lightcoral', 'background-color: lightgreen']
                except:
                    pass
            
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_better, axis=1)
        
        # 주피터 노트북에서 표시
        display(HTML("<h3>전략 vs 벤치마크 성과 비교</h3>"))
        display(styled_df)
    
    def _check_visualization_data(self) -> bool:
        """시각화에 필요한 데이터가 있는지 확인"""
        if not self.config.visualization_mode:
            print("시각화 기능을 사용하려면 백테스팅 시 visualization_mode=True로 설정하세요.")
            return False
        
        if self.equity_curve is None:
            print("equity_curve 데이터가 없습니다.")
            return False
        
        return True
    
    def _calculate_benchmark_metrics(self) -> Dict[str, float]:
        """벤치마크 성과 지표 계산"""
        if self.benchmark_returns is None:
            return {}
        
        returns = self.benchmark_returns["return"].to_numpy()
        
        total_return = (1 + returns).prod() - 1
        days = len(returns)
        years = days / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 드로다운 계산
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 소르티노 비율 계산
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        # 칼마 비율 계산
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_calmar_ratio(self) -> float:
        """칼마 비율 계산"""
        if self.max_drawdown == 0:
            return 0
        return self.annual_return / abs(self.max_drawdown)
    
    def _calculate_sortino_ratio(self) -> float:
        """소르티노 비율 계산"""
        if self.daily_returns is None:
            return 0
        
        returns = self.daily_returns["return"].to_numpy()
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_std = negative_returns.std() * np.sqrt(252)
        return self.annual_return / downside_std if downside_std > 0 else 0
    
    def _calculate_beta(self) -> float:
        """베타 계산"""
        if self.daily_returns is None or self.benchmark_returns is None:
            return 0
        
        strategy_returns = self.daily_returns["return"].to_numpy()
        benchmark_returns = self.benchmark_returns["return"].to_numpy()
        
        if len(strategy_returns) != len(benchmark_returns):
            return 0
        
        # division by zero 방지 강화
        try:
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 1e-10:  # 매우 작은 값도 0으로 처리
                return covariance / benchmark_variance
            else:
                return 0
        except (ZeroDivisionError, FloatingPointError):
            return 0
    
    def _calculate_alpha(self) -> float:
        """알파 계산"""
        if self.benchmark_returns is None:
            return 0
        
        benchmark_return = self.benchmark_returns["return"].mean() * 252
        beta = self._calculate_beta()
        
        return self.annual_return - beta * benchmark_return
    
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
        """요약 정보 반환"""
        return {
            "기간": f"{self.config.start_date.date()} ~ {self.config.end_date.date()}",
            "초기자본": f"{self.config.initial_cash:,.0f}",
            "최종자본": f"{self.final_equity:,.0f}",
            "총수익률": f"{self.total_return_pct:.2f}%",
            "연간수익률": f"{self.annual_return_pct:.2f}%",
            "변동성": f"{self.volatility_pct:.2f}%",
            "샤프비율": f"{self.sharpe_ratio:.2f}",
            "최대낙폭": f"{self.max_drawdown_pct:.2f}%",
            "총거래수": self.total_trades,
            "승률": f"{self.win_rate_pct:.1f}%",
            "수익인수": f"{self.profit_factor:.2f}",
            "실행시간": f"{self.duration:.2f}초"
        }
    
    def print_summary(self) -> None:
        """요약 정보 출력"""
        print("=" * 50)
        print("백테스팅 결과 요약")
        print("=" * 50)
        
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key:12}: {value}")
        
        print("=" * 50) 