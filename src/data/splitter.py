"""Data splitting module for train/validation/test sets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random


class DataSplitter:
    """Split data into train, validation, and test sets."""

    def __init__(self, config: Dict):
        """
        Initialize data splitter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_validation_periods = config['validation']['n_periods']
        self.validation_unit = config['validation']['period_unit']
        self.test_duration = config['test']['period_duration']
        self.test_unit = config['test']['period_unit']
        self.random_seed = config['validation'].get('random_seed', 42)

        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def split_data(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: Combined DataFrame with all features
            verbose: Print split information

        Returns:
            Dictionary with 'train', 'validation', and 'test' DataFrames
        """
        # Sort data by date
        data = data.sort_index()

        # 1. Extract test period (last N period_units)
        test_data, remaining_data = self._extract_test_period(data)

        # 2. Extract validation periods (random N periods from remaining)
        validation_data, train_periods = self._extract_validation_periods(remaining_data)

        # 3. Combine remaining periods for training
        train_data = self._combine_train_periods(train_periods)

        if verbose:
            self._print_split_info(train_data, validation_data, test_data)

        return {
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        }

    def _extract_test_period(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract test period from the end of the data.

        Args:
            data: Full dataset

        Returns:
            Tuple of (test_data, remaining_data)
        """
        # Calculate test period start date
        end_date = data.index.max()

        if self.test_unit == 'year':
            test_start = end_date - pd.DateOffset(years=self.test_duration)
        elif self.test_unit == 'month':
            test_start = end_date - pd.DateOffset(months=self.test_duration)
        elif self.test_unit == 'week':
            test_start = end_date - pd.DateOffset(weeks=self.test_duration)
        else:
            test_start = end_date - pd.DateOffset(days=self.test_duration)

        # Ensure test_start is in the data
        test_start = max(test_start, data.index.min())

        # Split data
        test_data = data.loc[test_start:].copy()
        remaining_data = data.loc[:test_start].iloc[:-1].copy()  # Exclude last day to avoid overlap

        return test_data, remaining_data

    def _extract_validation_periods(
        self,
        data: pd.DataFrame
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Extract random validation periods from data.

        Args:
            data: Data excluding test period

        Returns:
            Tuple of (validation_periods, remaining_train_periods)
        """
        # Create period boundaries based on validation_unit
        periods = self._create_periods(data)

        if len(periods) < self.n_validation_periods + 1:
            raise ValueError(
                f"Not enough periods for validation. "
                f"Need {self.n_validation_periods} validation + at least 1 training, "
                f"but only {len(periods)} periods available."
            )

        # Randomly select validation periods
        validation_indices = random.sample(
            range(len(periods)),
            min(self.n_validation_periods, len(periods) - 1)
        )

        validation_periods = []
        train_periods = []

        for i, period in enumerate(periods):
            if i in validation_indices:
                validation_periods.append(period)
            else:
                train_periods.append(period)

        return validation_periods, train_periods

    def _create_periods(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create periods based on the validation unit.

        Args:
            data: DataFrame to split into periods

        Returns:
            List of period DataFrames
        """
        periods = []

        if self.validation_unit == 'year':
            # Group by year
            for year in data.index.year.unique():
                year_data = data[data.index.year == year]
                if len(year_data) > 20:  # Minimum days for a valid period
                    periods.append(year_data)

        elif self.validation_unit == 'month':
            # Group by year-month
            for year in data.index.year.unique():
                for month in range(1, 13):
                    month_data = data[(data.index.year == year) & (data.index.month == month)]
                    if len(month_data) > 5:  # Minimum days for a valid period
                        periods.append(month_data)

        elif self.validation_unit == 'week':
            # Group by week
            data['year_week'] = data.index.isocalendar().week + data.index.year * 100
            for week in data['year_week'].unique():
                week_data = data[data['year_week'] == week].drop(columns=['year_week'])
                if len(week_data) > 2:  # Minimum days for a valid period
                    periods.append(week_data)
            # Clean up
            if 'year_week' in data.columns:
                data.drop(columns=['year_week'], inplace=True)

        else:
            raise ValueError(f"Unsupported period unit: {self.validation_unit}")

        return periods

    def _combine_train_periods(self, periods: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine training periods into single DataFrame.

        Args:
            periods: List of period DataFrames

        Returns:
            Combined training DataFrame
        """
        if not periods:
            raise ValueError("No training periods available")

        # Concatenate all periods
        train_data = pd.concat(periods, axis=0)

        # Sort by date and remove duplicates
        train_data = train_data.sort_index()
        train_data = train_data[~train_data.index.duplicated(keep='first')]

        return train_data

    def _print_split_info(
        self,
        train_data: pd.DataFrame,
        validation_data: List[pd.DataFrame],
        test_data: pd.DataFrame
    ) -> None:
        """Print information about the data split."""
        print("\n" + "="*60)
        print("DATA SPLIT SUMMARY")
        print("="*60)

        # Training data
        print(f"\nTraining Data:")
        print(f"  - Records: {len(train_data):,}")
        print(f"  - Date range: {train_data.index.min().date()} to {train_data.index.max().date()}")
        print(f"  - Duration: {(train_data.index.max() - train_data.index.min()).days} days")

        # Validation data
        print(f"\nValidation Data ({self.n_validation_periods} {self.validation_unit}s):")
        for i, val_period in enumerate(validation_data):
            print(f"  Period {i+1}:")
            print(f"    - Records: {len(val_period):,}")
            print(f"    - Date range: {val_period.index.min().date()} to {val_period.index.max().date()}")

        # Test data
        print(f"\nTest Data:")
        print(f"  - Records: {len(test_data):,}")
        print(f"  - Date range: {test_data.index.min().date()} to {test_data.index.max().date()}")
        print(f"  - Duration: {(test_data.index.max() - test_data.index.min()).days} days")

        # Total summary
        total_records = len(train_data) + sum(len(v) for v in validation_data) + len(test_data)
        print(f"\nTotal records: {total_records:,}")
        print("="*60 + "\n")

    def create_time_series_splits(
        self,
        train_data: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series cross-validation splits for training.

        Args:
            train_data: Training data
            n_splits: Number of splits

        Returns:
            List of (train, validation) tuples
        """
        splits = []
        total_days = (train_data.index.max() - train_data.index.min()).days

        # Calculate split sizes
        min_train_days = 30  # Minimum training period
        val_days = 30  # Validation period

        if total_days < min_train_days + val_days:
            # If not enough data, return single split
            split_point = train_data.index.min() + pd.DateOffset(days=int(total_days * 0.8))
            train_split = train_data.loc[:split_point]
            val_split = train_data.loc[split_point:]
            return [(train_split, val_split)]

        # Create expanding window splits
        for i in range(n_splits):
            # Calculate split points
            train_end_pct = 0.3 + (0.5 * i / max(n_splits - 1, 1))
            train_end = train_data.index.min() + pd.DateOffset(
                days=int(total_days * train_end_pct)
            )
            val_end = min(
                train_end + pd.DateOffset(days=val_days),
                train_data.index.max()
            )

            # Create splits
            train_split = train_data.loc[:train_end]
            val_split = train_data.loc[train_end:val_end]

            if len(train_split) > min_train_days and len(val_split) > 5:
                splits.append((train_split, val_split))

        return splits

    def get_validation_periods(self, validation_data: List[pd.DataFrame]) -> List[Dict]:
        """
        Get information about validation periods.

        Args:
            validation_data: List of validation DataFrames

        Returns:
            List of period information dictionaries
        """
        periods_info = []

        for i, period in enumerate(validation_data):
            info = {
                'period_num': i + 1,
                'start_date': period.index.min(),
                'end_date': period.index.max(),
                'n_days': len(period),
                'year': period.index[0].year,
                'month': period.index[0].month if self.validation_unit == 'month' else None,
                'week': period.index[0].isocalendar().week if self.validation_unit == 'week' else None,
            }
            periods_info.append(info)

        return periods_info