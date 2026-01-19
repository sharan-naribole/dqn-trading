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
    - Multi-buy accumulation (multiple buys up to max_shares)
    - Partial sells with FIFO lot tracking
    - Action masking to prevent invalid trades
    - Stop-loss and take-profit guardrails (based on weighted average)
    - Transaction costs
    - Configurable reward functions

    Attributes:
        data (pd.DataFrame): Market data with features
        config (Dict): Configuration parameters
        action_masker (ActionMasker): Handles action space and masking
        guardrails (TradingGuardrails): Risk management rules
        balance (float): Current cash balance
        shares_held (int): Total shares currently held
        lots (List[Dict]): Position lots with {shares, entry_price}
        entry_price (float): Weighted average entry price
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
                - trading.max_shares: Maximum total shares that can be held
                - trading.share_increments: List of share quantities for buy/sell
                  (e.g., [10, 50, 100])
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

        # Support both share_increments (new) and share_step (legacy)
        if 'share_increments' in config['trading']:
            self.share_increments = config['trading']['share_increments']
        elif 'share_step' in config['trading']:
            # Legacy: convert share_step to share_increments
            step = config['trading']['share_step']
            self.share_increments = list(range(step, self.max_shares + 1, step))
        else:
            # Default: single share increments
            self.share_increments = [1]

        self.starting_balance = config['trading']['starting_balance']

        # Initialize components
        self.action_masker = ActionMasker(self.max_shares, self.share_increments)
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

        # Multi-buy position tracking with lots
        self.lots = []  # List of {shares, entry_price} dicts
        self.entry_price = None  # Weighted average (for compatibility)

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
                - 0: HOLD
                - 1 to N: BUY actions (one for each share_increment)
                - N+1 to 2N: SELL actions (one for each share_increment)
                - 2N+1: SELL_ALL

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

        # Override action if guardrails trigger (always sell all on guardrail)
        if should_exit and self.shares_held > 0:
            action = self.action_masker.SELL_ALL  # Guardrails always exit entire position
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

        Supports:
        - Multiple buys (accumulation up to max_shares)
        - Partial sells (FIFO lot tracking)
        - Sell all

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

        elif self.action_masker.is_buy_action(action):
            # Buy shares (may be adding to existing position)
            shares_to_buy = self.action_masker.action_to_shares(action)

            new_balance, cost, success = self.guardrails.execute_buy(
                shares_to_buy, current_price, self.balance
            )

            if success:
                self.balance = new_balance
                self._add_lot(shares_to_buy, current_price)  # Add to position
                self.total_shares_traded += shares_to_buy

                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.balance,
                    'total_shares_held': self.shares_held,
                    'weighted_avg_entry': self.entry_price
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
                # Failed to buy (insufficient balance or would exceed max_shares)
                reward = -1

        elif self.action_masker.is_sell_action(action):
            # Sell shares (partial or all)
            if self.shares_held > 0:
                # Determine shares to sell
                if action == self.action_masker.SELL_ALL:
                    shares_to_sell = self.shares_held
                else:
                    shares_to_sell = abs(self.action_masker.action_to_shares(action))

                new_balance, proceeds, success = self.guardrails.execute_sell(
                    shares_to_sell, current_price, self.balance
                )

                if success:
                    # Calculate profit using FIFO lot tracking
                    realized_profit, avg_entry_sold = self._remove_shares_fifo(
                        shares_to_sell, current_price
                    )

                    self.balance = new_balance
                    self.total_profit += realized_profit

                    profit_pct = (current_price - avg_entry_sold) / avg_entry_sold if avg_entry_sold > 0 else 0

                    # Record trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'SELL_ALL' if action == self.action_masker.SELL_ALL else f'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'proceeds': proceeds,
                        'profit': realized_profit,
                        'profit_pct': profit_pct,
                        'balance': self.balance,
                        'total_shares_held': self.shares_held,
                        'avg_entry_sold': avg_entry_sold
                    })

                    # Calculate net profit after ALL transaction costs (buy + sell)
                    buy_transaction_cost_per_share = self.config.get('trading', {}).get('buy_transaction_cost_per_share', 0.01)
                    sell_transaction_cost_per_share = self.config.get('trading', {}).get('sell_transaction_cost_per_share', 0.01)
                    total_transaction_cost = (buy_transaction_cost_per_share + sell_transaction_cost_per_share) * shares_to_sell
                    net_profit = realized_profit - total_transaction_cost

                    # Reward based on net profit as percentage of position value sold
                    position_value_sold = avg_entry_sold * shares_to_sell
                    reward = (net_profit / position_value_sold) * 100 if position_value_sold > 0 else 0
                else:
                    # Failed to sell
                    reward = -1
            else:
                # No shares to sell
                reward = -1

        self.balance_history.append(self.balance)
        return reward

    def _calculate_weighted_avg_entry(self) -> Optional[float]:
        """
        Calculate weighted average entry price from lots.

        Returns:
            Weighted average entry price, or None if no position
        """
        if not self.lots or self.shares_held == 0:
            return None

        total_cost = sum(lot['shares'] * lot['entry_price'] for lot in self.lots)
        return total_cost / self.shares_held

    def _add_lot(self, shares: int, price: float):
        """
        Add a new lot to position.

        Args:
            shares: Number of shares in lot
            price: Entry price for lot
        """
        self.lots.append({'shares': shares, 'entry_price': price})
        self.shares_held += shares
        self.entry_price = self._calculate_weighted_avg_entry()

    def _remove_shares_fifo(self, shares_to_sell: int, sell_price: float) -> Tuple[float, float]:
        """
        Remove shares using FIFO and calculate realized profit.

        Args:
            shares_to_sell: Number of shares to sell
            sell_price: Current selling price

        Returns:
            Tuple of (realized_profit, avg_entry_price_sold)
        """
        if shares_to_sell > self.shares_held:
            raise ValueError(f"Cannot sell {shares_to_sell} shares, only have {self.shares_held}")

        remaining_to_sell = shares_to_sell
        total_cost_basis = 0
        shares_sold = 0

        while remaining_to_sell > 0 and self.lots:
            lot = self.lots[0]  # FIFO: take from first lot

            if lot['shares'] <= remaining_to_sell:
                # Sell entire lot
                shares_from_lot = lot['shares']
                total_cost_basis += shares_from_lot * lot['entry_price']
                shares_sold += shares_from_lot
                remaining_to_sell -= shares_from_lot
                self.lots.pop(0)  # Remove lot
            else:
                # Partial lot sale
                shares_from_lot = remaining_to_sell
                total_cost_basis += shares_from_lot * lot['entry_price']
                shares_sold += shares_from_lot
                lot['shares'] -= shares_from_lot  # Reduce lot size
                remaining_to_sell = 0

        # Update position
        self.shares_held -= shares_sold
        self.entry_price = self._calculate_weighted_avg_entry()

        # Calculate profit
        avg_entry_sold = total_cost_basis / shares_sold
        realized_profit = (sell_price - avg_entry_sold) * shares_sold

        return realized_profit, avg_entry_sold

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