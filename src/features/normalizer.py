"""Rolling Z-score normalization for features."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class RollingNormalizer:
    """Apply rolling Z-score normalization to features."""

    def __init__(self, config: Dict):
        """
        Initialize normalizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window_size = config['data']['normalization_window']
        self.feature_stats = {}

    def fit_transform(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        preserve_original: bool = True
    ) -> pd.DataFrame:
        """
        Apply rolling Z-score normalization to features.

        IMPORTANT: Uses BACKWARD-ONLY rolling window to avoid lookahead bias.
        Each data point is normalized using only historical data (past 30 days).

        Args:
            data: DataFrame with features
            feature_columns: List of columns to normalize
            preserve_original: Keep original columns with '_orig' suffix

        Returns:
            DataFrame with normalized features
        """
        normalized_data = data.copy()

        for col in feature_columns:
            if col not in data.columns:
                continue

            # Preserve original if requested
            if preserve_original:
                normalized_data[f"{col}_orig"] = data[col]

            # Calculate rolling statistics (BACKWARD-ONLY, no lookahead)
            # .rolling() by default only looks backward, but we make it explicit
            rolling_mean = data[col].rolling(
                window=self.window_size,
                min_periods=1,
                center=False  # Explicitly no centering (backward-only)
            ).mean()

            rolling_std = data[col].rolling(
                window=self.window_size,
                min_periods=1,
                center=False  # Explicitly no centering (backward-only)
            ).std()

            # Replace zero std with small value to avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-10)

            # Apply normalization
            normalized_data[col] = (data[col] - rolling_mean) / rolling_std

            # Store statistics for the latest window (for inference on new data)
            self.feature_stats[col] = {
                'mean': rolling_mean.iloc[-1],
                'std': rolling_std.iloc[-1]
            }

        # Handle any infinite or NaN values
        normalized_data = normalized_data.replace([np.inf, -np.inf], np.nan)
        normalized_data = normalized_data.fillna(0)

        return normalized_data

    def transform(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        use_stored_stats: bool = False
    ) -> pd.DataFrame:
        """
        Transform data using rolling normalization.

        Args:
            data: DataFrame with features
            feature_columns: List of columns to normalize
            use_stored_stats: Use stored statistics instead of rolling

        Returns:
            DataFrame with normalized features
        """
        if use_stored_stats and self.feature_stats:
            return self._transform_with_stored_stats(data, feature_columns)
        else:
            return self.fit_transform(data, feature_columns, preserve_original=False)

    def _transform_with_stored_stats(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Transform using stored statistics (for inference).

        Args:
            data: DataFrame with features
            feature_columns: List of columns to normalize

        Returns:
            DataFrame with normalized features
        """
        normalized_data = data.copy()

        for col in feature_columns:
            if col not in data.columns or col not in self.feature_stats:
                continue

            mean = self.feature_stats[col]['mean']
            std = self.feature_stats[col]['std']

            if std == 0:
                std = 1e-10

            normalized_data[col] = (data[col] - mean) / std

        # Handle any infinite or NaN values
        normalized_data = normalized_data.replace([np.inf, -np.inf], np.nan)
        normalized_data = normalized_data.fillna(0)

        return normalized_data

    def inverse_transform(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data.

        Args:
            data: Normalized DataFrame
            feature_columns: List of columns to denormalize

        Returns:
            DataFrame with original scale features
        """
        denormalized_data = data.copy()

        for col in feature_columns:
            if col not in data.columns or col not in self.feature_stats:
                continue

            mean = self.feature_stats[col]['mean']
            std = self.feature_stats[col]['std']

            denormalized_data[col] = (data[col] * std) + mean

        return denormalized_data

    def get_normalization_stats(self) -> Dict:
        """
        Get normalization statistics.

        Returns:
            Dictionary with normalization statistics
        """
        return self.feature_stats.copy()

    def save_stats(self, filepath: str) -> None:
        """
        Save normalization statistics to file.

        Args:
            filepath: Path to save statistics
        """
        import json

        with open(filepath, 'w') as f:
            json.dump(self.feature_stats, f, indent=2)

    def load_stats(self, filepath: str) -> None:
        """
        Load normalization statistics from file.

        Args:
            filepath: Path to load statistics from
        """
        import json

        with open(filepath, 'r') as f:
            self.feature_stats = json.load(f)

    def create_normalized_states(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        window_size: int
    ) -> np.ndarray:
        """
        Create normalized state features for DQN model.

        Args:
            data: DataFrame with features
            feature_columns: List of feature columns
            window_size: Number of time steps

        Returns:
            Array of normalized states
        """
        # First normalize the data
        normalized_data = self.fit_transform(data, feature_columns, preserve_original=True)

        # Extract normalized features
        normalized_features = normalized_data[feature_columns].values

        # Create windowed features
        n_samples = len(normalized_features) - window_size + 1
        n_features = len(feature_columns)

        states = np.zeros((n_samples, window_size, n_features))

        for i in range(n_samples):
            states[i] = normalized_features[i:i + window_size]

        return states

    def clip_outliers(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Clip outliers beyond n standard deviations.

        Args:
            data: DataFrame with features
            feature_columns: List of columns to clip
            n_std: Number of standard deviations for clipping

        Returns:
            DataFrame with clipped features
        """
        clipped_data = data.copy()

        for col in feature_columns:
            if col not in data.columns:
                continue

            # Clip values beyond n_std
            clipped_data[col] = np.clip(data[col], -n_std, n_std)

        return clipped_data