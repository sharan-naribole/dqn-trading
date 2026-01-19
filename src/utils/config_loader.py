"""Configuration loader for DQN Trading project."""

import json
import os
from typing import Dict, Any
from datetime import datetime
import argparse


class ConfigLoader:
    """Load and manage configuration for the DQN trading system."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join('config', 'default_config.json')

        self.config_path = config_path
        self.config = self.load_config()
        self.validate_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def validate_config(self) -> None:
        """Validate configuration parameters based on what's present."""
        # Validate dates if present
        if 'start_date' in self.config and 'end_date' in self.config:
            try:
                start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
                end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')

                if start_date >= end_date:
                    raise ValueError("Start date must be before end date")

            except ValueError as e:
                raise ValueError(f"Invalid date format: {e}")

        # Validate trading parameters if present
        if 'trading' in self.config:
            if self.config['trading']['max_shares'] < 1:
                raise ValueError("max_shares must be at least 1")

            if self.config['trading']['starting_balance'] <= 0:
                raise ValueError("starting_balance must be positive")

            if not 0 <= self.config['trading']['stop_loss_pct'] <= 100:
                raise ValueError("stop_loss_pct must be between 0 and 100")

            if not 0 <= self.config['trading']['take_profit_pct'] <= 10000:
                raise ValueError("take_profit_pct must be between 0 and 10000")

        # Validate training parameters if present
        if 'training' in self.config:
            if self.config['training']['episodes'] < 1:
                raise ValueError("episodes must be at least 1")

        # Validate data parameters if present
        if 'data' in self.config:
            if self.config['data']['window_size'] < 1:
                raise ValueError("window_size must be at least 1")

        # Validate mode if present
        if 'mode' in self.config:
            valid_modes = ['train', 'test', 'dry_run']
            if self.config['mode'] not in valid_modes:
                raise ValueError(f"mode must be one of {valid_modes}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).

        Args:
            key: Configuration key (e.g., 'trading.max_shares')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            key: Configuration key (e.g., 'trading.max_shares')
            value: New value
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: str = None) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save configuration. If None, overwrites original file.
        """
        if path is None:
            path = self.config_path

        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ConfigLoader':
        """
        Create ConfigLoader from command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            ConfigLoader instance
        """
        # Load base configuration
        config_path = args.config if hasattr(args, 'config') else None
        loader = cls(config_path)

        # Override with command line arguments
        arg_mapping = {
            'ticker': 'ticker',
            'start_date': 'start_date',
            'end_date': 'end_date',
            'mode': 'mode',
            'episodes': 'training.episodes',
            'max_shares': 'trading.max_shares',
            'window_size': 'data.window_size',
        }

        for arg_name, config_key in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                loader.update(config_key, getattr(args, arg_name))

        return loader

    def create_model_identifier(self) -> str:
        """
        Create unique identifier for model based on configuration.

        Returns:
            Model identifier string
        """
        ticker = self.config['ticker']
        start = self.config['start_date']
        end = self.config['end_date']
        episodes = self.config['training']['episodes']
        window = self.config['data']['window_size']
        max_shares = self.config['trading']['max_shares']

        return f"{ticker}_{start}_{end}_ep{episodes}_ws{window}_ms{max_shares}"