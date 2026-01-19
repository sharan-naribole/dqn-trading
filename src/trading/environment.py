"""
Trading Environment Module

Provides a gym-like environment for training DQN agents to trade stocks.
Implements action masking, position management, and risk guardrails.

Author: DQN Trading System
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from ..models.action_masking import ActionMasker
from .guardrails import TradingGuardrails


class TradingEnvironment:
    """
    Trading environment with action masking and guardrails.

    This environment simulates stock trading with:
    - Variable position sizing (share_step to max_shares in increments of share_step)
    - Action masking to prevent invalid trades
    - Stop-loss and take-profit guardrails
    - Transaction costs
    - Configurable reward functions

    Attributes:
        data (pd.DataFrame): Market data with features
        config (Dict): Configuration parameters
        action_masker (ActionMasker): Handles action space and masking
        guardrails (TradingGuardrails): Risk management rules
        balance (float): Current cash balance
        shares_held (int): Current position size
        entry_price (float): Price at which position was opened
        trades (List[Dict]): History of executed trades
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        config: Dict,
        mode: str = 'train'
    ):
        """
        Initialize trading environment.

        Parameters:
            data (pd.DataFrame): DataFrame with price and feature data
                Must contain 'Close_orig' column for trading prices
            feature_columns (List[str]): Names of feature columns to use in state
            config (Dict): Configuration dictionary containing:
                - ticker: Stock symbol
                - data.window_size: Lookback window for state
                - trading.max_shares: Maximum shares per trade
                - trading.share_step: Step size for share quantities (default: 1)
                - trading.starting_balance: Initial capital
                - trading.stop_loss_pct: Stop-loss threshold
                - trading.take_profit_pct: Take-profit threshold
                - trading.transaction_cost: Cost per trade
            mode (str): 'train' or 'test' mode

        Raises:
            KeyError: If required configuration parameters are missing
            ValueError: If data doesn't contain required columns
        """
        self.data = data
        self.feature_columns = feature_columns
        self.config = config
        self.mode = mode

        # Extract configuration
        self.ticker = config['ticker']
        self.window_size = config['data']['window_size']
        self.max_shares = config['trading']['max_shares']
        self.share_step = config['trading'].get('share_step', 1)  # Default to 1 for backward compatibility
        self.starting_balance = config['trading']['starting_balance']

        # Initialize components
        self.action_masker = ActionMasker(self.max_shares, self.share_step)
        self.guardrails = TradingGuardrails(
            stop_loss_pct=config['trading']['stop_loss_pct'],
            take_profit_pct=config['trading']['take_profit_pct'],
            buy_transaction_cost_per_share=config['trading'].get('buy_transaction_cost_per_share', 0.01),
            sell_transaction_cost_per_share=config['trading'].get('sell_transaction_cost_per_share', 0.01)
        )

        # Get price column (for trading) - use original Close price
        close_orig = f'{self.ticker}_Close_orig'
        close_col = f'{self.ticker}_Close'
        self.price_column = close_orig if close_orig in data.columns else close_col

        # Environment state
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Resets all trading variables to their initial values and returns
        the initial state observation.

        Returns:
            Tuple[np.ndarray, Dict]:
                - initial_state: Initial observation array of shape (window_size, n_features+2)
                - info: Dictionary with environment information
        """
        # Reset variables
        self.current_step = self.window_size
        self.balance = self.starting_balance
        self.shares_held = 0
        self.entry_price = None
        self.total_shares_traded = 0
        self.total_profit = 0

        # Reset tracking
        self.trades = []
        self.balance_history = [self.balance]
        self.portfolio_values = [self.balance]
        self.actions_taken = []
        self.rewards_received = []

        # Reset components
        self.guardrails.reset()

        # Get initial state
        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.

        Processes the given action, updates the environment state,
        calculates rewards, and returns the new observation.

        Parameters:
            action (int): Action to execute
                - 0: Hold
                - 1 to n_buy_actions: Buy (action * share_step) shares
                - n_buy_actions+1: Sell all shares

        Returns:
            Tuple[np.ndarray, float, bool, Dict]:
                - next_state: New observation after action
                - reward: Reward for the action taken
                - done: Whether episode has ended
                - info: Additional information dictionary

        Raises:
            ValueError: If action is invalid according to action mask
        """
        # Validate action
        action_mask = self._get_action_mask()
        if not action_mask[action]:
            # Force to HOLD if invalid action
            action = self.action_masker.HOLD

        # Get current price
        current_price = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()

        # Check guardrails before executing action
        should_exit, exit_reason = self.guardrails.check_guardrails(
            current_price, self.shares_held, self.entry_price
        )

        # Override action if guardrails trigger
        if should_exit and self.shares_held > 0:
            action = self.action_masker.SELL
            self._log_guardrail_trigger(exit_reason)

        # Execute action
        reward = self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Get next state
        next_state = self._get_state()
        done = self._is_done()

        # Track portfolio value for metrics (reward comes only from _execute_action)
        current_portfolio_value = self._get_portfolio_value()

        # Track metrics
        self.actions_taken.append(action)
        self.rewards_received.append(reward)
        self.portfolio_values.append(current_portfolio_value)

        # Get info
        info = self._get_info()
        info['action_taken'] = self.action_masker.get_action_name(action)
        info['reward'] = reward

        return next_state, reward, done, info

    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute trading action and return reward.

        Args:
            action: Action to execute
            current_price: Current stock price

        Returns:
            Reward for the action
        """
        reward = 0

        if action == self.action_masker.HOLD:
            # No trading, configurable reward/penalty for being idle
            idle_reward = self.config.get('trading', {}).get('idle_reward', -0.001)
            reward = idle_reward

        elif action in self.action_masker.BUY_ACTIONS:
            # Buy shares
            shares_to_buy = self.action_masker.action_to_shares(action)

            new_balance, cost, success = self.guardrails.execute_buy(
                shares_to_buy, current_price, self.balance
            )

            if success:
                self.balance = new_balance
                self.shares_held = shares_to_buy
                self.entry_price = current_price
                self.total_shares_traded += shares_to_buy

                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.balance
                })

                # Buy reward with two components:
                # 1. Buy reward (fixed per share, to encourage/discourage buying)
                buy_reward_per_share = self.config.get('trading', {}).get('buy_reward_per_share', 0.0)
                buy_reward = buy_reward_per_share * shares_to_buy

                # 2. Transaction cost penalty (fixed cost per share)
                buy_transaction_cost_per_share = self.config.get('trading', {}).get('buy_transaction_cost_per_share', 0.01)
                transaction_penalty = -buy_transaction_cost_per_share * shares_to_buy

                reward = buy_reward + transaction_penalty
            else:
                # Failed to buy (insufficient balance)
                reward = -1

        elif action == self.action_masker.SELL:
            # Sell all shares
            if self.shares_held > 0:
                new_balance, proceeds, success = self.guardrails.execute_sell(
                    self.shares_held, current_price, self.balance
                )

                if success:
                    # Calculate profit (absolute profit in dollars)
                    total_profit = (current_price - self.entry_price) * self.shares_held
                    profit_pct = (current_price - self.entry_price) / self.entry_price

                    self.balance = new_balance
                    self.total_profit += total_profit

                    # Record trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'SELL',
                        'shares': self.shares_held,
                        'price': current_price,
                        'proceeds': proceeds,
                        'profit': total_profit,
                        'profit_pct': profit_pct,
                        'balance': self.balance
                    })

                    # Calculate net profit after ALL transaction costs (buy + sell)
                    buy_transaction_cost_per_share = self.config.get('trading', {}).get('buy_transaction_cost_per_share', 0.01)
                    sell_transaction_cost_per_share = self.config.get('trading', {}).get('sell_transaction_cost_per_share', 0.01)
                    total_transaction_cost = (buy_transaction_cost_per_share + sell_transaction_cost_per_share) * self.shares_held
                    net_profit = total_profit - total_transaction_cost

                    # Reward based on net profit as percentage of position value
                    position_value = self.entry_price * self.shares_held
                    reward = (net_profit / position_value) * 100
                    # Example: Buy 10 shares at $100, sell at $110
                    # Buy cost = $0.10, sell cost = $0.10, total transaction = $0.20
                    # Gross profit = $100, net profit = $99.80, position value = $1000
                    # Reward = 9.98%

                    # Reset position
                    self.shares_held = 0
                    self.entry_price = None
            else:
                # No shares to sell
                reward = -1

        self.balance_history.append(self.balance)
        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        if self.current_step < self.window_size:
            # Not enough data for full window
            return np.zeros((self.window_size, len(self.feature_columns)))

        # Get windowed features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        state_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values

        # Add position information to state
        position_info = np.array([
            self.shares_held / self.max_shares,  # Normalized position
            self.balance / self.starting_balance,  # Normalized balance
        ])

        # Broadcast position info to all time steps
        position_features = np.tile(position_info, (self.window_size, 1))

        # Concatenate features
        state = np.concatenate([state_data, position_features], axis=1)

        return state

    def _get_action_mask(self) -> np.ndarray:
        """Get current action mask."""
        current_price = self._get_current_price()
        return self.action_masker.get_action_mask(
            self.shares_held, self.balance, current_price
        )

    def _get_current_price(self) -> float:
        """Get current stock price."""
        return self.data.iloc[self.current_step][self.price_column]

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = self._get_current_price()
        return self.balance + (self.shares_held * current_price)

    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Episode ends when we reach the end of data
        return self.current_step >= len(self.data) - 1

    def _get_info(self) -> Dict:
        """Get current environment info."""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()

        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_profit': self.total_profit,
            'num_trades': len(self.trades),
            'position_metrics': self.guardrails.get_position_metrics(current_price)
        }

        return info

    def _log_guardrail_trigger(self, reason: str):
        """Log guardrail trigger event."""
        if self.mode == 'test':
            print(f"Guardrail Triggered at step {self.current_step}: {reason}")

    def render(self, mode: str = 'simple'):
        """
        Render environment state.

        Args:
            mode: Rendering mode ('simple' or 'detailed')
        """
        info = self._get_info()

        if mode == 'simple':
            print(f"Step: {info['step']}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}, "
                  f"Shares: {info['shares_held']}, "
                  f"Price: ${info['current_price']:.2f}")
        else:
            print("\n" + "="*60)
            print(f"Step: {info['step']}/{len(self.data)-1}")
            print(f"Balance: ${info['balance']:.2f}")
            print(f"Shares Held: {info['shares_held']}")
            print(f"Current Price: ${info['current_price']:.2f}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Total Profit: ${info['total_profit']:.2f}")
            print(f"Number of Trades: {info['num_trades']}")

            if info['shares_held'] > 0:
                metrics = info['position_metrics']
                print(f"\nPosition Metrics:")
                print(f"  Entry Price: ${metrics['entry_price']:.2f}")
                print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:.2f}")
                print(f"  Unrealized Return: {metrics['unrealized_return']:.2%}")
                print(f"  Stop Loss: ${metrics['stop_loss_price']:.2f}")
                print(f"  Take Profit: ${metrics['take_profit_price']:.2f}")
            print("="*60)

    def get_metrics(self) -> Dict:
        """Get comprehensive environment metrics."""
        if len(self.portfolio_values) == 0:
            return {}

        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.starting_balance) / self.starting_balance

        # Calculate Sharpe ratio
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe = np.sqrt(252) * (np.mean(returns) / (np.std(returns) + 1e-10))

        # Calculate max drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Calculate win rate
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit', 0) <= 0]
        win_rate = len(winning_trades) / max(len(self.trades), 1)

        # Position sizing statistics (CRITICAL for understanding agent's learning!)
        buy_trades = [t for t in self.trades if t.get('action') == 'BUY']
        if buy_trades:
            shares_bought = [t.get('shares', 0) for t in buy_trades]
            avg_position_size = np.mean(shares_bought)
            min_position_size = np.min(shares_bought)
            max_position_size = np.max(shares_bought)
            std_position_size = np.std(shares_bought)

            # Position size distribution (histogram)
            position_size_dist = {}
            for i in range(1, self.max_shares + 1):
                count = sum(1 for s in shares_bought if s == i)
                if count > 0:
                    position_size_dist[f'{i}_shares'] = count
        else:
            avg_position_size = 0.0
            min_position_size = 0
            max_position_size = 0
            std_position_size = 0.0
            position_size_dist = {}

        return {
            'total_return': float(total_return),
            'final_portfolio_value': float(final_value),
            'total_profit': float(self.total_profit),
            'num_trades': int(len(self.trades)),
            'win_rate': float(win_rate),
            'winning_trades': int(len(winning_trades)),
            'losing_trades': int(len(losing_trades)),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'avg_trade_profit': float(np.mean([t.get('profit', 0) for t in self.trades])) if self.trades else 0.0,
            'total_shares_traded': int(self.total_shares_traded),
            # Position sizing metrics
            'avg_position_size': float(avg_position_size),
            'min_position_size': int(min_position_size),
            'max_position_size': int(max_position_size),
            'std_position_size': float(std_position_size),
            'position_size_distribution': position_size_dist,
            'num_buy_trades': len(buy_trades)
        }