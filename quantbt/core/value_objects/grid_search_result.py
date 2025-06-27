"""
Grid Search Result

Value object that contains the results of parallel backtesting grid search.
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
    """Individual backtest summary result"""
    params: Dict[str, Any]  # Strategy parameters
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
        """Convert to dictionary"""
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
        # Include parameters as well
        result.update(self.params)
        return result


@dataclass(frozen=True)
class GridSearchResult:
    """Grid search result"""
    
    # Basic information
    config: GridSearchConfig
    start_time: datetime
    end_time: datetime
    
    # Result summary
    summaries: List[GridSearchSummary]  # All backtest summaries
    
    # Optimal result
    best_params: Dict[str, Any]  # Optimal parameter combination
    best_summary: GridSearchSummary
    
    # Top results (optionally include detailed results)
    top_results: Optional[List[BacktestResult]] = None
    
    # Execution statistics
    total_executed: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """Execution time (seconds)"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Success rate"""
        if self.total_executed == 0:
            return 0.0
        return self.successful_runs / self.total_executed
    
    @property
    def results_df(self) -> pl.DataFrame:
        """Convert results to DataFrame"""
        data = [summary.to_dict() for summary in self.summaries]
        return pl.DataFrame(data)
    
    def get_top_n_summaries(self, n: int = 10, metric: str = "calmar_ratio") -> List[GridSearchSummary]:
        """Return top N results"""
        sorted_summaries = sorted(
            self.summaries,
            key=lambda x: getattr(x, metric, 0),
            reverse=True
        )
        return sorted_summaries[:n]
    
    def get_metric_statistics(self, metric: str = "calmar_ratio") -> Dict[str, float]:
        """Statistical information by metric"""
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
        """Print result summary"""
        print("=" * 80)
        print("                    GRID SEARCH RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"Execution time: {self.duration:.1f}s")
        print(f"Total executed: {self.total_executed}")
        print(f"Successful: {self.successful_runs}")
        print(f"Failed: {self.failed_runs}")
        print(f"Success rate: {self.success_rate:.1%}")
        print()
        
        print(f"Optimal parameters (based on Calmar Ratio):")
        for param_name, param_value in self.best_params.items():
            print(f"  {param_name}: {param_value}")
        print(f"  Calmar Ratio: {self.best_summary.calmar_ratio:.3f}")
        print(f"  Annual Return: {self.best_summary.annual_return:.2%}")
        print(f"  Max Drawdown: {self.best_summary.max_drawdown:.2%}")
        print(f"  Sharpe Ratio: {self.best_summary.sharpe_ratio:.3f}")
        print()
        
        # Calmar Ratio statistics
        calmar_stats = self.get_metric_statistics("calmar_ratio")
        print("Calmar Ratio distribution:")
        print(f"  Mean: {calmar_stats.get('mean', 0):.3f}")
        print(f"  Std: {calmar_stats.get('std', 0):.3f}")
        print(f"  Max: {calmar_stats.get('max', 0):.3f}")
        print(f"  Min: {calmar_stats.get('min', 0):.3f}")
        print()
        
        # Top results
        top_results = self.get_top_n_summaries(top_n)
        print(f"Top {len(top_results)} results:")
        print("-" * 120)
        
        # Print header (parameters + performance metrics)
        header = f"{'Rank':>4} "
        if top_results:
            # Construct header with parameters from first result
            for param_name in top_results[0].params.keys():
                header += f"{param_name:>8} "
        header += f"{'Calmar':>8} {'Annual':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7}"
        print(header)
        print("-" * 120)
        
        for i, summary in enumerate(top_results, 1):
            row = f"{i:>4} "
            # Print parameter values
            for param_value in summary.params.values():
                row += f"{param_value:>8} "
            # Print performance metrics
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
        """Performance heatmap by parameter combination"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Please install plotly for visualization: pip install plotly")
            return
        
        # Prepare data
        df = self.results_df.to_pandas()
        
        # Check if specified parameters exist in data
        if x_param not in df.columns or y_param not in df.columns:
            print(f"Parameter {x_param} or {y_param} not found.")
            print(f"Available parameters: {[col for col in df.columns if col not in ['calmar_ratio', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'total_trades', 'win_rate', 'profit_factor', 'total_return', 'volatility', 'final_equity']]}")
            return
        
        # Create pivot table
        heatmap_data = df.pivot(index=y_param, columns=x_param, values=metric)
        
        # Create heatmap
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
        
        # Mark optimal point
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
        """Performance metric distribution histogram"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Please install plotly for visualization: pip install plotly")
            return
        
        values = [getattr(summary, metric, 0) for summary in self.summaries]
        
        fig = go.Figure(data=[go.Histogram(
            x=values,
            nbinsx=bins,
            name=metric.replace('_', ' ').title(),
            marker_color='lightblue',
            opacity=0.7
        )])
        
        # Mark optimal value
        best_value = getattr(self.best_summary, metric, 0)
        fig.add_vline(
            x=best_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {best_value:.3f}"
        )
        
        # Add statistical information
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
        """Convert to dictionary"""
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