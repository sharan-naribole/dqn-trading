"""Data collection module with smart caching using yfinance."""

import os
import glob
from typing import Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    """Collect market data from Yahoo Finance with intelligent caching."""

    def __init__(self, config: Dict):
        """
        Initialize data collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ticker = config['ticker']
        self.start_date = pd.to_datetime(config['start_date'])
        self.end_date = pd.to_datetime(config['end_date'])
        self.data_dir = config.get('output', {}).get('data_dir', 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Buffer for technical indicators (need historical data)
        self.buffer_days = 250  # ~1 year of trading days

    def collect_data(
        self,
        force_download: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect ticker and VIX data with smart caching.

        Caching strategy:
        1. Check if we have cached files that contain the requested date range
        2. If yes, load and filter to requested dates
        3. If no, download new data and save with proper naming

        Args:
            force_download: Force download even if cached data exists

        Returns:
            Tuple of (ticker_data, vix_data) DataFrames
        """
        # Get data for main ticker
        ticker_data = self._get_ticker_data(self.ticker, force_download)

        # Get VIX data
        vix_data = self._get_ticker_data("^VIX", force_download)

        return ticker_data, vix_data

    def _get_ticker_data(self, ticker: str, force_download: bool) -> pd.DataFrame:
        """
        Get data for a specific ticker with caching logic.

        Args:
            ticker: Ticker symbol
            force_download: Force new download

        Returns:
            DataFrame with ticker data filtered to requested dates
        """
        # Calculate actual date range needed (with buffer)
        start_with_buffer = self.start_date - timedelta(days=self.buffer_days)

        if not force_download:
            # Try to find cached file that contains our date range
            cached_file = self._find_cached_file(ticker, start_with_buffer, self.end_date)

            if cached_file:
                print(f"Using cached data from {cached_file}")
                data = pd.read_csv(cached_file, index_col='Date', parse_dates=True)

                # Filter to requested date range (with buffer)
                data = data[(data.index >= start_with_buffer) & (data.index <= self.end_date)]

                if not data.empty:
                    print(f"  Loaded {len(data)} records for {ticker}")
                    return data
                else:
                    print(f"  Cached file doesn't have sufficient data, downloading new...")

        # Download new data
        print(f"Downloading {ticker} data from {start_with_buffer.date()} to {self.end_date.date()}")
        data = self._download_and_cache_data(ticker, start_with_buffer, self.end_date)

        return data

    def _find_cached_file(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[str]:
        """
        Find cached file that contains the requested date range.

        File format: ticker_YYYYMMDD_YYYYMMDD.csv

        Args:
            ticker: Ticker symbol
            start_date: Start date needed
            end_date: End date needed

        Returns:
            Path to cached file if found, None otherwise
        """
        # Clean ticker for filename (^VIX becomes VIX)
        clean_ticker = ticker.replace('^', '')

        # Look for existing files
        pattern = os.path.join(self.data_dir, f"{clean_ticker}_*_*.csv")
        cached_files = glob.glob(pattern)

        for filepath in cached_files:
            # Parse dates from filename
            filename = os.path.basename(filepath).replace('.csv', '')

            # Extract date parts (last two parts separated by underscore)
            # Format: TICKER_YYYYMMDD_YYYYMMDD
            parts = filename.split('_')
            if len(parts) >= 3:
                try:
                    # Last two parts are always dates
                    file_start_str = parts[-2]
                    file_end_str = parts[-1]

                    # Parse dates
                    file_start = pd.to_datetime(file_start_str, format='%Y%m%d')
                    file_end = pd.to_datetime(file_end_str, format='%Y%m%d')

                    # Check if this file contains our requested range
                    if file_start <= start_date and file_end >= end_date:
                        # Verify file actually has data in this range
                        test_df = pd.read_csv(filepath, index_col='Date', parse_dates=True, nrows=5)
                        if not test_df.empty:
                            print(f"  Found matching cache: {filename}")
                            return filepath
                except Exception as e:
                    # Skip files that don't match expected format
                    continue

        return None

    def _download_and_cache_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Download data and save with proper naming convention.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            Downloaded DataFrame
        """
        # Download data with auto_adjust=True (default) - Close is already adjusted
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
            # auto_adjust=True is the default - Close prices are automatically adjusted
        )

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # For multi-level columns (if downloading multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                if ticker == "^VIX" and col == 'Volume':
                    # VIX doesn't have volume
                    data['Volume'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")

        # Rename columns to include ticker prefix
        data = data.rename(columns={
            col: f"{self.ticker}_{col}" if ticker == self.ticker else f"VIX_{col}"
            for col in data.columns
        })

        # Save with proper naming: ticker_startdate_enddate.csv
        clean_ticker = ticker.replace('^', '')
        actual_start = data.index.min()
        actual_end = data.index.max()

        filename = f"{clean_ticker}_{actual_start.strftime('%Y%m%d')}_{actual_end.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, filename)

        data.to_csv(filepath)
        print(f"  Saved {len(data)} records to {filename}")

        return data

    def combine_data(
        self,
        ticker_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine ticker and VIX data.

        Args:
            ticker_data: Main ticker DataFrame
            vix_data: VIX DataFrame

        Returns:
            Combined DataFrame
        """
        # Ensure both have ticker prefixes in column names
        ticker_cols = [col for col in ticker_data.columns if self.ticker in col]
        if not ticker_cols:
            # Add ticker prefix if not present
            ticker_data = ticker_data.rename(columns={
                col: f"{self.ticker}_{col}" for col in ticker_data.columns
            })

        vix_cols = [col for col in vix_data.columns if 'VIX' in col]
        if not vix_cols:
            # Add VIX prefix if not present
            vix_data = vix_data.rename(columns={
                col: f"VIX_{col}" for col in vix_data.columns
            })

        # Merge on date index
        combined = pd.merge(
            ticker_data,
            vix_data[['VIX_Close']],  # Only need VIX close price
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Drop any rows with NaN values
        combined = combined.dropna()

        return combined

    def get_data_info(
        self,
        ticker_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> Dict:
        """
        Get information about collected data.

        Args:
            ticker_data: Ticker DataFrame
            vix_data: VIX DataFrame

        Returns:
            Dictionary with data information
        """
        return {
            f"{self.ticker}_records": len(ticker_data),
            f"{self.ticker}_start": ticker_data.index.min().strftime('%Y-%m-%d'),
            f"{self.ticker}_end": ticker_data.index.max().strftime('%Y-%m-%d'),
            "vix_records": len(vix_data),
            "vix_start": vix_data.index.min().strftime('%Y-%m-%d'),
            "vix_end": vix_data.index.max().strftime('%Y-%m-%d')
        }

    def cleanup_old_cache(self, days_to_keep: int = 30):
        """
        Clean up old cached files.

        Args:
            days_to_keep: Keep files modified within this many days
        """
        current_time = datetime.now()

        for filepath in glob.glob(os.path.join(self.data_dir, "*.csv")):
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (current_time - file_modified).days > days_to_keep:
                os.remove(filepath)
                print(f"Removed old cache file: {os.path.basename(filepath)}")