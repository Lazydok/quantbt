"""
그리드 서치 결과

병렬 백테스팅 그리드 서치 결과를 담는 값 객체입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
import numpy as np

from .grid_search_config import GridSearchConfig
from .backtest_result import BacktestResult


@dataclass(frozen=True)
class GridSearchSummary:
    """개별 백테스트 요약 결과"""
    params: Dict[str, Any]  # 전략 파라미터들
    calmar_ratio: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    final_equity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "calmar_ratio": self.calmar_ratio,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "final_equity": self.final_equity
        }
        # 파라미터들도 포함
        result.update(self.params)
        return result


@dataclass(frozen=True)
class GridSearchResult:
    """그리드 서치 결과"""
    
    # 기본 정보
    config: GridSearchConfig
    start_time: datetime
    end_time: datetime
    
    # 결과 요약
    summaries: List[GridSearchSummary]  # 모든 백테스트 요약
    
    # 최적 결과
    best_params: Dict[str, Any]  # 최적 파라미터 조합
    best_summary: GridSearchSummary
    
    # 상위 결과들 (선택적으로 상세 결과 포함)
    top_results: Optional[List[BacktestResult]] = None
    
    # 실행 통계
    total_executed: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # 메타데이터
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """실행 시간 (초)"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_executed == 0:
            return 0.0
        return self.successful_runs / self.total_executed
    
    @property
    def results_df(self) -> pl.DataFrame:
        """결과를 DataFrame으로 변환"""
        data = [summary.to_dict() for summary in self.summaries]
        return pl.DataFrame(data)
    
    def get_top_n_summaries(self, n: int = 10, metric: str = "calmar_ratio") -> List[GridSearchSummary]:
        """상위 N개 결과 반환"""
        sorted_summaries = sorted(
            self.summaries,
            key=lambda x: getattr(x, metric, 0),
            reverse=True
        )
        return sorted_summaries[:n]
    
    def get_metric_statistics(self, metric: str = "calmar_ratio") -> Dict[str, float]:
        """지표별 통계 정보"""
        values = [getattr(summary, metric, 0) for summary in self.summaries]
        if not values:
            return {}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75)
        }
    
    def print_summary(self, top_n: int = 10) -> None:
        """결과 요약 출력"""
        print("=" * 80)
        print("                    GRID SEARCH RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"실행 시간: {self.duration:.1f}초")
        print(f"총 실행: {self.total_executed}개")
        print(f"성공: {self.successful_runs}개")
        print(f"실패: {self.failed_runs}개")
        print(f"성공률: {self.success_rate:.1%}")
        print()
        
        print(f"최적 파라미터 (Calmar Ratio 기준):")
        for param_name, param_value in self.best_params.items():
            print(f"  {param_name}: {param_value}")
        print(f"  Calmar Ratio: {self.best_summary.calmar_ratio:.3f}")
        print(f"  Annual Return: {self.best_summary.annual_return:.2%}")
        print(f"  Max Drawdown: {self.best_summary.max_drawdown:.2%}")
        print(f"  Sharpe Ratio: {self.best_summary.sharpe_ratio:.3f}")
        print()
        
        # Calmar Ratio 통계
        calmar_stats = self.get_metric_statistics("calmar_ratio")
        print("Calmar Ratio 분포:")
        print(f"  평균: {calmar_stats.get('mean', 0):.3f}")
        print(f"  표준편차: {calmar_stats.get('std', 0):.3f}")
        print(f"  최대: {calmar_stats.get('max', 0):.3f}")
        print(f"  최소: {calmar_stats.get('min', 0):.3f}")
        print()
        
        # 상위 결과 
        top_results = self.get_top_n_summaries(top_n)
        print(f"상위 {len(top_results)}개 결과:")
        print("-" * 120)
        
        # 헤더 출력 (파라미터들 + 성과지표들)
        header = f"{'Rank':>4} "
        if top_results:
            # 첫 번째 결과의 파라미터들로 헤더 구성
            for param_name in top_results[0].params.keys():
                header += f"{param_name:>8} "
        header += f"{'Calmar':>8} {'Annual':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7}"
        print(header)
        print("-" * 120)
        
        for i, summary in enumerate(top_results, 1):
            row = f"{i:>4} "
            # 파라미터 값들 출력
            for param_value in summary.params.values():
                row += f"{param_value:>8} "
            # 성과 지표들 출력  
            row += (f"{summary.calmar_ratio:>8.3f} {summary.annual_return:>7.2%} "
                   f"{summary.max_drawdown:>7.2%} {summary.sharpe_ratio:>8.3f} "
                   f"{summary.total_trades:>7}")
            print(row)
        
        print("=" * 80)
    
    def plot_heatmap(self, 
                     x_param: str, 
                     y_param: str, 
                     metric: str = "calmar_ratio", 
                     figsize: tuple = (12, 8)) -> None:
        """파라미터 조합별 성과 히트맵"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("시각화를 위해 plotly를 설치해주세요: pip install plotly")
            return
        
        # 데이터 준비
        df = self.results_df.to_pandas()
        
        # 지정된 파라미터들이 데이터에 있는지 확인
        if x_param not in df.columns or y_param not in df.columns:
            print(f"파라미터 {x_param} 또는 {y_param}을 찾을 수 없습니다.")
            print(f"사용 가능한 파라미터: {[col for col in df.columns if col not in ['calmar_ratio', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades', 'win_rate', 'profit_factor', 'total_return', 'volatility', 'final_equity']]}")
            return
        
        # 피벗 테이블 생성
        heatmap_data = df.pivot(index=y_param, columns=x_param, values=metric)
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # 최적점 표시
        best_x = self.best_params.get(x_param)
        best_y = self.best_params.get(y_param)
        if best_x is not None and best_y is not None:
            fig.add_trace(go.Scatter(
                x=[best_x],
                y=[best_y],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='white',
                    line=dict(color='black', width=2)
                ),
                name='Best',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"Parameter Heatmap - {metric.replace('_', ' ').title()}",
            xaxis_title=x_param,
            yaxis_title=y_param,
            template="plotly_white"
        )
        
        fig.show()
    
    def plot_distribution(self, metric: str = "calmar_ratio", bins: int = 30) -> None:
        """성과 지표 분포 히스토그램"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("시각화를 위해 plotly를 설치해주세요: pip install plotly")
            return
        
        values = [getattr(summary, metric, 0) for summary in self.summaries]
        
        fig = go.Figure(data=[go.Histogram(
            x=values,
            nbinsx=bins,
            name=metric.replace('_', ' ').title(),
            marker_color='lightblue',
            opacity=0.7
        )])
        
        # 최적값 표시
        best_value = getattr(self.best_summary, metric, 0)
        fig.add_vline(
            x=best_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {best_value:.3f}"
        )
        
        # 통계 정보 추가
        stats = self.get_metric_statistics(metric)
        fig.add_annotation(
            x=0.7, y=0.9,
            xref="paper", yref="paper",
            text=f"Mean: {stats.get('mean', 0):.3f}<br>" +
                 f"Std: {stats.get('std', 0):.3f}<br>" +
                 f"Max: {stats.get('max', 0):.3f}<br>" +
                 f"Min: {stats.get('min', 0):.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Distribution",
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        fig.show()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "summaries": [s.to_dict() for s in self.summaries],
            "best_params": self.best_params,
            "best_summary": self.best_summary.to_dict(),
            "total_executed": self.total_executed,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.success_rate,
            "metadata": self.metadata or {}
        } 