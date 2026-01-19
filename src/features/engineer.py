"""Feature engineering for technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ta


class FeatureEngineer:
    """Create technical indicators and features for trading."""

    def __init__(self, config: Dict):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicators_config = config['data']['indicators']
        self.ticker = config['ticker']  # Get ticker name from config

    def create_features(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Create all technical features.

        Args:
            data: Combined SPY and VIX data
            verbose: Print progress

        Returns:
            DataFrame with all features
        """
        if verbose:
            print("Creating technical indicators...")

        # Create a copy to avoid modifying original data
        featured_data = data.copy()

        # Calculate indicators for ticker data using Close price
        close_col = f'{self.ticker}_Close'
        if close_col in featured_data.columns:
            featured_data = self._add_bollinger_bands(featured_data)
            featured_data = self._add_moving_averages(featured_data)
            featured_data = self._add_rsi(featured_data)
            featured_data = self._add_adx(featured_data)

        # Add additional derived features
        featured_data = self._add_price_features(featured_data)
        featured_data = self._add_volume_features(featured_data)

        # Drop any rows with NaN values from indicator calculations
        before_rows = len(featured_data)
        featured_data = featured_data.dropna()
        after_rows = len(featured_data)

        if verbose:
            print(f"Features created. Dropped {before_rows - after_rows} rows with NaN values.")
            print(f"Total features: {len(featured_data.columns)}")

        return featured_data

    def _add_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        period = self.indicators_config['bollinger_period']
        std = self.indicators_config['bollinger_std']
        close_col = f'{self.ticker}_Close'

        # Calculate Bollinger Bands
        indicator = ta.volatility.BollingerBands(
            close=data[close_col],
            window=period,
            window_dev=std
        )

        data['BB_High'] = indicator.bollinger_hband()
        data['BB_Low'] = indicator.bollinger_lband()
        data['BB_Middle'] = indicator.bollinger_mavg()
        data['BB_Width'] = data['BB_High'] - data['BB_Low']
        data['BB_Position'] = (data[close_col] - data['BB_Low']) / (data['BB_Width'] + 1e-10)

        return data

    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and SMA indicators."""
        close_col = f'{self.ticker}_Close'

        # EMAs
        data['EMA_8'] = ta.trend.ema_indicator(
            data[close_col],
            window=self.indicators_config['ema_short']
        )
        data['EMA_21'] = ta.trend.ema_indicator(
            data[close_col],
            window=self.indicators_config['ema_medium']
        )

        # SMAs
        data['SMA_50'] = ta.trend.sma_indicator(
            data[close_col],
            window=self.indicators_config['sma_short']
        )
        data['SMA_200'] = ta.trend.sma_indicator(
            data[close_col],
            window=self.indicators_config['sma_long']
        )

        # Moving average signals
        data['EMA_Signal'] = (data['EMA_8'] > data['EMA_21']).astype(int)
        data['Price_Above_SMA50'] = (data[close_col] > data['SMA_50']).astype(int)
        data['Price_Above_SMA200'] = (data[close_col] > data['SMA_200']).astype(int)

        return data

    def _add_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator."""
        period = self.indicators_config['rsi_period']
        close_col = f'{self.ticker}_Close'

        data['RSI'] = ta.momentum.RSIIndicator(
            close=data[close_col],
            window=period
        ).rsi()

        # Add RSI zones
        data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)

        return data

    def _add_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ADX indicator."""
        period = self.indicators_config['adx_period']
        high_col = f'{self.ticker}_High'
        low_col = f'{self.ticker}_Low'
        close_col = f'{self.ticker}_Close'

        adx = ta.trend.ADXIndicator(
            high=data[high_col],
            low=data[low_col],
            close=data[close_col],
            window=period
        )

        data['ADX'] = adx.adx()
        data['ADX_Pos'] = adx.adx_pos()
        data['ADX_Neg'] = adx.adx_neg()
        data['ADX_Strong'] = (data['ADX'] > 25).astype(int)

        return data

    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close_col = f'{self.ticker}_Close'
        high_col = f'{self.ticker}_High'
        low_col = f'{self.ticker}_Low'

        # Price changes
        data['Price_Change'] = data[close_col].pct_change()
        data['Price_Change_5d'] = data[close_col].pct_change(5)
        data['Price_Change_20d'] = data[close_col].pct_change(20)

        # High-Low spread
        data['HL_Spread'] = (data[high_col] - data[low_col]) / data[close_col]

        # Close position in daily range
        data['Close_Position'] = (data[close_col] - data[low_col]) / \
                                  (data[high_col] - data[low_col] + 1e-10)

        # Volatility measures
        data['Volatility_5d'] = data['Price_Change'].rolling(5).std()
        data['Volatility_20d'] = data['Price_Change'].rolling(20).std()

        return data

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        close_col = f'{self.ticker}_Close'
        volume_col = f'{self.ticker}_Volume'

        # Volume moving averages
        data['Volume_MA_10'] = data[volume_col].rolling(10).mean()
        data['Volume_MA_20'] = data[volume_col].rolling(20).mean()

        # Volume ratio
        data['Volume_Ratio'] = data[volume_col] / (data['Volume_MA_20'] + 1e-10)

        # On-Balance Volume (OBV)
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=data[close_col],
            volume=data[volume_col]
        ).on_balance_volume()

        return data

    def get_feature_names(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of feature names for the model.

        Args:
            data: DataFrame with features

        Returns:
            List of feature column names
        """
        # Define features to use in the model
        feature_columns = []
        close_col = f'{self.ticker}_Close'
        volume_col = f'{self.ticker}_Volume'

        # Price features
        price_features = [
            close_col, 'Price_Change', 'Price_Change_5d',
            'Close_Position', 'HL_Spread'
        ]

        # Technical indicators
        technical_features = [
            'BB_Position', 'BB_Width',
            'EMA_8', 'EMA_21', 'SMA_50', 'SMA_200',
            'EMA_Signal', 'Price_Above_SMA50', 'Price_Above_SMA200',
            'RSI', 'RSI_Oversold', 'RSI_Overbought',
            'ADX', 'ADX_Strong'
        ]

        # Volume features
        volume_features = [
            volume_col, 'Volume_Ratio', 'OBV'
        ]

        # VIX features
        vix_features = ['VIX_Close'] if 'VIX_Close' in data.columns else []

        # Volatility features
        volatility_features = ['Volatility_5d', 'Volatility_20d']

        # Combine all features
        all_features = (price_features + technical_features +
                        volume_features + vix_features + volatility_features)

        # Filter to only include features that exist in the data
        feature_columns = [col for col in all_features if col in data.columns]

        return feature_columns

    def create_state_features(
        self,
        data: pd.DataFrame,
        window_size: int
    ) -> np.ndarray:
        """
        Create state features for DQN model.

        Args:
            data: DataFrame with features
            window_size: Number of time steps to include

        Returns:
            Array of shape (n_samples, window_size, n_features)
        """
        feature_columns = self.get_feature_names(data)
        feature_data = data[feature_columns].values

        # Create windowed features
        n_samples = len(feature_data) - window_size + 1
        n_features = len(feature_columns)

        states = np.zeros((n_samples, window_size, n_features))

        for i in range(n_samples):
            states[i] = feature_data[i:i + window_size]

        return states

    def get_feature_info(self, data: pd.DataFrame) -> Dict:
        """
        Get information about created features.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with feature information
        """
        feature_columns = self.get_feature_names(data)

        info = {
            'total_features': len(feature_columns),
            'feature_names': feature_columns,
            'price_features': [f for f in feature_columns if 'Price' in f or 'Close' in f],
            'technical_indicators': [f for f in feature_columns if any(
                ind in f for ind in ['BB', 'EMA', 'SMA', 'RSI', 'ADX']
            )],
            'volume_features': [f for f in feature_columns if 'Volume' in f or 'OBV' in f],
            'vix_features': [f for f in feature_columns if 'VIX' in f],
            'data_shape': data.shape,
            'date_range': f"{data.index.min()} to {data.index.max()}"
        }

        return info