"""Unit tests for reward calculation logic."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.environment import TradingEnvironment
from src.utils.config_loader import ConfigLoader


class TestRewardCalculation:
    """Test suite for reward calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Load and merge configs from new structure
        import json
        data_config = ConfigLoader('config/dry_run/data_config.json').config
        with open('config/dry_run/trading_dry_run.json', 'r') as f:
            trading_config = json.load(f)
        self.config = {**data_config, **trading_config}

        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        self.test_data = pd.DataFrame({
            'SPY_Close': np.linspace(100, 110, 50),
            'SPY_Close_orig': np.linspace(100, 110, 50),
            'SPY_High': np.linspace(101, 111, 50),
            'SPY_Low': np.linspace(99, 109, 50),
            'SPY_Volume': np.ones(50) * 1000000,
            'test_feature_1': np.random.randn(50),
            'test_feature_2': np.random.randn(50),
        }, index=dates)

        self.feature_columns = ['SPY_Close', 'test_feature_1', 'test_feature_2']

        self.env = TradingEnvironment(
            self.test_data,
            self.feature_columns,
            self.config,
            mode='test'
        )

    def test_hold_reward(self):
        """Test that HOLD action gives small negative reward."""
        self.env.reset()

        # Execute HOLD action
        _, reward, _, _ = self.env.step(0)  # 0 = HOLD

        # Should be idle reward
        expected_reward = self.config['trading']['idle_reward']
        assert reward == pytest.approx(expected_reward, abs=0.0001)

    def test_buy_reward_includes_transaction_cost(self):
        """Test that BUY action includes transaction cost penalty."""
        self.env.reset()

        # Execute BUY action (buy 1 share)
        _, reward, _, _ = self.env.step(1)  # 1 = BUY 1 share

        # Reward should be negative (transaction cost)
        expected_cost = -self.config['trading']['buy_transaction_cost_per_share']
        assert reward == pytest.approx(expected_cost, abs=0.0001)

    def test_profitable_sell_reward(self):
        """Test that profitable sell gives positive reward."""
        self.env.reset()

        # Buy at lower price
        self.env.step(1)  # Buy 1 share
        entry_price = self.env.entry_price

        # Move forward several steps to increase price
        for _ in range(5):
            self.env.step(0)  # HOLD

        current_price = self.env._get_current_price()

        # Sell at higher price (use first sell action)
        _, sell_reward, _, _ = self.env.step(self.env.action_masker.SELL_ACTIONS[0])

        # Reward should be positive (log return)
        # Formula: log(1 + net_profit / position_value) * 100
        gross_profit = current_price - entry_price
        buy_cost = self.config['trading']['buy_transaction_cost_per_share']
        sell_cost = self.config['trading']['sell_transaction_cost_per_share']
        net_profit = gross_profit - (buy_cost + sell_cost)

        simple_return = net_profit / entry_price
        expected_reward = np.log(1 + simple_return) * 100

        if current_price > entry_price:
            assert sell_reward > 0, "Profitable trade should have positive reward"
        assert sell_reward == pytest.approx(expected_reward, abs=0.01)

    def test_losing_sell_reward(self):
        """Test that losing sell gives negative reward."""
        # Create data with price decrease
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        decreasing_data = pd.DataFrame({
            'SPY_Close': np.linspace(110, 100, 50),  # Decreasing
            'SPY_Close_orig': np.linspace(110, 100, 50),
            'SPY_High': np.linspace(111, 101, 50),
            'SPY_Low': np.linspace(109, 99, 50),
            'SPY_Volume': np.ones(50) * 1000000,
            'test_feature_1': np.random.randn(50),
            'test_feature_2': np.random.randn(50),
        }, index=dates)

        env = TradingEnvironment(
            decreasing_data,
            self.feature_columns,
            self.config,
            mode='test'
        )

        env.reset()

        # Buy at higher price
        env.step(1)
        entry_price = env.entry_price

        # Move forward
        for _ in range(5):
            env.step(0)

        current_price = env._get_current_price()

        # Sell at lower price (use first sell action)
        _, sell_reward, _, _ = env.step(env.action_masker.SELL_ACTIONS[0])

        # Reward should be negative
        if current_price < entry_price:
            assert sell_reward < 0, "Losing trade should have negative reward"

    def test_no_duplicate_portfolio_reward(self):
        """Test that reward comes ONLY from execute_action (not portfolio return)."""
        self.env.reset()

        # Get initial portfolio value
        initial_value = self.env._get_portfolio_value()

        # Execute HOLD
        _, hold_reward, _, _ = self.env.step(0)

        # Get new portfolio value
        new_value = self.env._get_portfolio_value()

        # Even if portfolio value changed, reward should ONLY be idle_reward
        # NOT idle_reward + portfolio_return
        expected_reward = self.config['trading']['idle_reward']
        assert hold_reward == pytest.approx(expected_reward, abs=0.0001)

        # This confirms no duplicate portfolio return reward

    def test_failed_buy_negative_reward(self):
        """Test that failed BUY (insufficient balance) gives idle reward."""
        # Set balance too low
        self.env.balance = 1.0

        # Try to buy (will fail, action masked to HOLD)
        _, reward, _, info = self.env.step(1)

        # Action masking converts invalid buy to HOLD, so get idle_reward
        assert reward == self.config['trading']['idle_reward']
        assert info['action_taken'] == 'HOLD'

    def test_invalid_sell_negative_reward(self):
        """Test that invalid SELL (no shares) gives idle reward."""
        self.env.reset()

        # Try to sell without holding (use first sell action)
        _, reward, _, info = self.env.step(self.env.action_masker.SELL_ACTIONS[0])

        # Action masking converts invalid sell to HOLD, so get idle_reward
        assert reward == self.config['trading']['idle_reward']
        assert info['action_taken'] == 'HOLD'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
