"""Out-of-sample validation for model selection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from ..trading.environment import TradingEnvironment
from ..training.trainer import DQNTrainer
from .metrics import PerformanceMetrics


class OutOfSampleValidator:
    """Perform out-of-sample validation for model selection."""

    def __init__(
        self,
        config: Dict,
        feature_columns: List[str]
    ):
        """
        Initialize validator.

        Args:
            config: Configuration dictionary
            feature_columns: List of feature column names
        """
        self.config = config
        self.feature_columns = feature_columns
        self.metrics_calculator = PerformanceMetrics()
        self.validation_results = []

    def validate_model(
        self,
        trainer: DQNTrainer,
        validation_periods: List[pd.DataFrame],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate model on out-of-sample periods.

        Args:
            trainer: Trained DQN trainer
            validation_periods: List of validation period DataFrames
            verbose: Print progress

        Returns:
            Aggregated validation metrics
        """
        period_results = []

        for i, period_data in enumerate(validation_periods):
            if verbose:
                print(f"\nValidating on period {i + 1}/{len(validation_periods)}")
                print(f"Period range: {period_data.index.min()} to {period_data.index.max()}")

            # Create environment for validation period
            env = TradingEnvironment(
                period_data,
                self.feature_columns,
                self.config,
                mode='test'
            )

            # Evaluate model
            metrics = trainer.evaluate(env, verbose=False)

            # Calculate additional metrics
            full_metrics = self.metrics_calculator.calculate_metrics(
                env.portfolio_values,
                env.trades,
                env.starting_balance
            )

            # Store results
            period_results.append(full_metrics)

            if verbose:
                self._print_period_results(i + 1, full_metrics)

        # Aggregate results
        aggregated = self._aggregate_results(period_results)

        # Store for later comparison
        self.validation_results.append({
            'period_results': period_results,
            'aggregated': aggregated
        })

        return aggregated

    def compare_models(
        self,
        models: Dict[str, DQNTrainer],
        validation_periods: List[pd.DataFrame],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models on validation sets.

        Args:
            models: Dictionary of model_name -> trainer
            validation_periods: List of validation period DataFrames
            verbose: Print progress

        Returns:
            DataFrame with model comparison
        """
        model_results = {}

        for model_name, trainer in models.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating model: {model_name}")
                print('='*60)

            results = self.validate_model(trainer, validation_periods, verbose)
            model_results[model_name] = results

        # Create comparison table
        comparison = self.metrics_calculator.compare_models(model_results)

        if verbose:
            self._print_comparison(comparison)

        return comparison

    def select_best_model(
        self,
        models: Dict[str, DQNTrainer],
        validation_periods: List[pd.DataFrame],
        selection_metric: str = 'sharpe_ratio',
        verbose: bool = True
    ) -> Tuple[str, DQNTrainer, Dict]:
        """
        Select best model based on validation performance.

        Args:
            models: Dictionary of model_name -> trainer
            validation_periods: Validation period DataFrames
            selection_metric: Metric to use for selection
            verbose: Print progress

        Returns:
            Tuple of (best_model_name, best_trainer, metrics)
        """
        comparison = self.compare_models(models, validation_periods, verbose)

        # Select best based on metric
        if selection_metric in ['max_drawdown']:
            # For drawdown, lower is better
            best_model = comparison[selection_metric].idxmax()
        else:
            # For most metrics, higher is better
            best_model = comparison[selection_metric].idxmax()

        best_trainer = models[best_model]
        best_metrics = comparison.loc[best_model].to_dict()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Best Model: {best_model}")
            print(f"Selected based on: {selection_metric} = {best_metrics[selection_metric]:.3f}")
            print('='*60)

        return best_model, best_trainer, best_metrics

    def cross_validate(
        self,
        trainer: DQNTrainer,
        train_data: pd.DataFrame,
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            trainer: DQN trainer
            train_data: Training data
            n_folds: Number of CV folds
            verbose: Print progress

        Returns:
            Cross-validation metrics
        """
        fold_results = []
        data_length = len(train_data)
        fold_size = data_length // (n_folds + 1)

        for fold in range(n_folds):
            if verbose:
                print(f"\nFold {fold + 1}/{n_folds}")

            # Create train/val split for this fold
            val_start = (fold + 1) * fold_size
            val_end = min(val_start + fold_size, data_length)

            fold_train = train_data.iloc[:val_start]
            fold_val = train_data.iloc[val_start:val_end]

            # Create environments
            train_env = TradingEnvironment(
                fold_train,
                self.feature_columns,
                self.config,
                mode='train'
            )

            val_env = TradingEnvironment(
                fold_val,
                self.feature_columns,
                self.config,
                mode='test'
            )

            # Train on fold
            trainer.train(train_env, validation_env=None, verbose=False)

            # Evaluate on validation
            metrics = trainer.evaluate(val_env, verbose=False)

            # Calculate full metrics
            full_metrics = self.metrics_calculator.calculate_metrics(
                val_env.portfolio_values,
                val_env.trades,
                val_env.starting_balance
            )

            fold_results.append(full_metrics)

        # Aggregate CV results
        cv_metrics = self._aggregate_results(fold_results)

        if verbose:
            self._print_cv_results(cv_metrics, n_folds)

        return cv_metrics

    def _aggregate_results(
        self,
        period_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple periods.

        Args:
            period_results: List of period metric dictionaries

        Returns:
            Aggregated metrics
        """
        if not period_results:
            return {}

        aggregated = {}

        # Metrics to average
        avg_metrics = [
            'total_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'win_rate', 'num_trades',
            'profit_factor', 'volatility'
        ]

        for metric in avg_metrics:
            values = [r.get(metric, 0) for r in period_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)

        # Keep the main metric without suffix for compatibility
        for metric in avg_metrics:
            aggregated[metric] = aggregated[f'{metric}_mean']

        return aggregated

    def _print_period_results(self, period_num: int, metrics: Dict[str, float]):
        """Print results for a validation period."""
        print(f"\nPeriod {period_num} Results:")
        print(f"  Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Trades: {metrics.get('num_trades', 0)}")

    def _print_comparison(self, comparison: pd.DataFrame):
        """Print model comparison table."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison.round(3))
        print("="*60)

    def _print_cv_results(self, cv_metrics: Dict[str, float], n_folds: int):
        """Print cross-validation results."""
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION RESULTS ({n_folds} folds)")
        print("="*60)
        print(f"Avg Return: {cv_metrics.get('total_return', 0):.2%} "
              f"(±{cv_metrics.get('total_return_std', 0):.2%})")
        print(f"Avg Sharpe: {cv_metrics.get('sharpe_ratio', 0):.2f} "
              f"(±{cv_metrics.get('sharpe_ratio_std', 0):.2f})")
        print(f"Avg Drawdown: {cv_metrics.get('max_drawdown', 0):.2%} "
              f"(±{cv_metrics.get('max_drawdown_std', 0):.2%})")
        print(f"Avg Win Rate: {cv_metrics.get('win_rate', 0):.2%} "
              f"(±{cv_metrics.get('win_rate_std', 0):.2%})")
        print("="*60)