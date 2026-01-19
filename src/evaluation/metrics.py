"""Performance metrics calculation and comparison."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceMetrics:
    """Calculate and analyze trading performance metrics."""

    def __init__(self):
        """Initialize performance metrics calculator."""
        pass

    def calculate_metrics(
        self,
        portfolio_values: List[float],
        trades: List[Dict],
        starting_balance: float,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio_values: List of portfolio values over time
            trades: List of trade dictionaries
            starting_balance: Initial balance
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary of performance metrics
        """
        portfolio_values = np.array(portfolio_values)

        # Basic returns
        total_return = (portfolio_values[-1] - starting_balance) / starting_balance

        # Daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Sharpe ratio (annualized)
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10))
        else:
            sharpe_ratio = 0

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = np.sqrt(252) * (np.mean(daily_returns) / (downside_std + 1e-10))
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        trade_metrics = self._analyze_trades(trades)

        # Risk metrics
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns[daily_returns <= var_95]) > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / max(len(daily_returns), 1)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0,
            **trade_metrics
        }

        # Benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(
                daily_returns, benchmark_returns
            )
            metrics.update(benchmark_metrics)

        return metrics

    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Analyze trade statistics.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_period': 0
            }

        # Separate buy and sell trades
        buy_trades = [t for t in trades if t.get('action') == 'BUY']
        sell_trades = [t for t in trades if t.get('action') == 'SELL']

        # Calculate profits from sell trades
        profits = [t.get('profit', 0) for t in sell_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]

        # Win rate
        win_rate = len(winning_trades) / max(len(profits), 1)

        # Average profits
        avg_profit = np.mean(profits) if profits else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average holding period (simplified)
        avg_holding_period = 0
        if len(buy_trades) == len(sell_trades):
            holding_periods = []
            for buy, sell in zip(buy_trades, sell_trades):
                holding_periods.append(sell['step'] - buy['step'])
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        return {
            'num_trades': len(sell_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'total_buy_trades': len(buy_trades),
            'total_sell_trades': len(sell_trades)
        }

    def _calculate_benchmark_metrics(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics relative to benchmark.

        Args:
            strategy_returns: Strategy daily returns
            benchmark_returns: Benchmark daily returns

        Returns:
            Dictionary of benchmark comparison metrics
        """
        # Align lengths
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Alpha and Beta
        if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / (benchmark_variance + 1e-10)

            benchmark_return = np.mean(benchmark_returns) * 252
            strategy_return = np.mean(strategy_returns) * 252
            alpha = strategy_return - beta * benchmark_return
        else:
            alpha = 0
            beta = 0

        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = (np.mean(excess_returns) * 252) / (tracking_error + 1e-10)

        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'benchmark_total_return': np.prod(1 + benchmark_returns) - 1
        }

    def compare_models(
        self,
        results: Dict[str, Dict],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model results.

        Args:
            results: Dictionary of model_name -> metrics
            metric_names: Specific metrics to compare

        Returns:
            DataFrame with comparison
        """
        if metric_names is None:
            metric_names = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'num_trades', 'profit_factor'
            ]

        comparison_data = []
        for model_name, metrics in results.items():
            row = {'model': model_name}
            for metric in metric_names:
                row[metric] = metrics.get(metric, np.nan)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.set_index('model')

        return df

    def plot_performance(
        self,
        portfolio_values: List[float],
        trades: List[Dict],
        title: str = "Portfolio Performance"
    ) -> plt.Figure:
        """
        Plot portfolio performance.

        Args:
            portfolio_values: List of portfolio values
            trades: List of trades
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value over time
        axes[0].plot(portfolio_values, label='Portfolio Value')
        axes[0].set_title(f'{title} - Portfolio Value')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)
        axes[0].legend()

        # Add trade markers
        buy_steps = [t['step'] for t in trades if t.get('action') == 'BUY']
        sell_steps = [t['step'] for t in trades if t.get('action') == 'SELL']

        if buy_steps:
            buy_values = [portfolio_values[min(step, len(portfolio_values)-1)] for step in buy_steps]
            axes[0].scatter(buy_steps, buy_values, color='green', marker='^', s=100, label='Buy', zorder=5)

        if sell_steps:
            sell_values = [portfolio_values[min(step, len(portfolio_values)-1)] for step in sell_steps]
            axes[0].scatter(sell_steps, sell_values, color='red', marker='v', s=100, label='Sell', zorder=5)

        # Ensure legend is shown with all elements
        axes[0].legend(loc='best')

        # Drawdown
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100

        axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)

        # Daily returns distribution
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        axes[2].hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_title('Daily Returns Distribution')
        axes[2].set_xlabel('Return (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)

        plt.tight_layout()
        return fig

    def create_performance_report(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a formatted performance report.

        Args:
            metrics: Performance metrics dictionary
            save_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)

        # Returns
        report.append("\nRETURNS:")
        report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")

        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"  VaR (95%): {metrics.get('var_95', 0):.2%}")
        report.append(f"  CVaR (95%): {metrics.get('cvar_95', 0):.2%}")

        # Trade Statistics
        report.append("\nTRADE STATISTICS:")
        report.append(f"  Total Trades: {metrics.get('num_trades', 0)}")
        report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"  Avg Profit: ${metrics.get('avg_profit', 0):.2f}")
        report.append(f"  Avg Win: ${metrics.get('avg_win', 0):.2f}")
        report.append(f"  Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
        report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"  Avg Holding Period: {metrics.get('avg_holding_period', 0):.1f} periods")

        # Benchmark Comparison (if available)
        if 'alpha' in metrics:
            report.append("\nBENCHMARK COMPARISON:")
            report.append(f"  Alpha: {metrics.get('alpha', 0):.2%}")
            report.append(f"  Beta: {metrics.get('beta', 0):.2f}")
            report.append(f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}")
            report.append(f"  Tracking Error: {metrics.get('tracking_error', 0):.2%}")

        report.append("\n" + "=" * 60)

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)

        return report_str