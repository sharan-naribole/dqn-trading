"""Trading guardrails for risk management."""

import numpy as np
from typing import Dict, Tuple, Optional


class TradingGuardrails:
    """Implement stop-loss and take-profit guardrails."""

    def __init__(
        self,
        stop_loss_pct: float = 5,
        take_profit_pct: float = 10,
        buy_transaction_cost_per_share: float = 0.01,
        sell_transaction_cost_per_share: float = 0.01
    ):
        """
        Initialize trading guardrails.

        Args:
            stop_loss_pct: Stop loss percentage on 0-100 scale (e.g., 5 = 5%, 20 = 20%)
            take_profit_pct: Take profit percentage on 0-100 scale (e.g., 10 = 10%, 30 = 30%)
            buy_transaction_cost_per_share: Fixed cost per share when buying (dollars, e.g., 0.01 = $0.01)
            sell_transaction_cost_per_share: Fixed cost per share when selling (dollars, e.g., 0.01 = $0.01)
        """
        # Convert 0-100 scale to 0-1 scale for internal calculations
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.buy_transaction_cost_per_share = buy_transaction_cost_per_share
        self.sell_transaction_cost_per_share = sell_transaction_cost_per_share

        # Track position details
        self.position_entry_price = None
        self.position_shares = 0
        self.position_value = 0

    def check_guardrails(
        self,
        current_price: float,
        current_position: int,
        entry_price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if guardrails should trigger.

        Args:
            current_price: Current stock price
            current_position: Current number of shares held
            entry_price: Entry price for the position

        Returns:
            Tuple of (should_exit, reason)
        """
        if current_position == 0:
            return False, "No position"

        if entry_price is None:
            entry_price = self.position_entry_price

        if entry_price is None:
            return False, "No entry price"

        # Calculate current return
        current_return = (current_price - entry_price) / entry_price

        # Check stop loss
        if current_return <= -self.stop_loss_pct:
            return True, f"Stop Loss triggered ({current_return:.2%} loss)"

        # Check take profit
        if current_return >= self.take_profit_pct:
            return True, f"Take Profit triggered ({current_return:.2%} gain)"

        return False, f"Within limits (Current: {current_return:.2%})"

    def calculate_buy_cost(self, shares: int) -> float:
        """
        Calculate transaction cost for buying shares.

        Args:
            shares: Number of shares to buy

        Returns:
            Total transaction cost
        """
        return shares * self.buy_transaction_cost_per_share

    def calculate_sell_cost(self, shares: int) -> float:
        """
        Calculate transaction cost for selling shares.

        Args:
            shares: Number of shares to sell

        Returns:
            Total transaction cost
        """
        return shares * self.sell_transaction_cost_per_share

    def execute_buy(
        self,
        shares: int,
        price: float,
        balance: float
    ) -> Tuple[float, float, bool]:
        """
        Execute buy order with guardrails.

        Args:
            shares: Number of shares to buy
            price: Current price
            balance: Available balance

        Returns:
            Tuple of (new_balance, cost, success)
        """
        # Calculate total cost including transaction fees
        trade_value = shares * price
        transaction_cost = self.calculate_buy_cost(shares)
        total_cost = trade_value + transaction_cost

        # Check if sufficient balance
        if balance < total_cost:
            return balance, 0, False

        # Update position tracking
        self.position_entry_price = price
        self.position_shares = shares
        self.position_value = trade_value

        # Return new balance and cost
        new_balance = balance - total_cost
        return new_balance, total_cost, True

    def execute_sell(
        self,
        shares: int,
        price: float,
        balance: float
    ) -> Tuple[float, float, bool]:
        """
        Execute sell order with guardrails.

        Args:
            shares: Number of shares to sell
            price: Current price
            balance: Current balance

        Returns:
            Tuple of (new_balance, proceeds, success)
        """
        # Calculate proceeds after transaction fees
        trade_value = shares * price
        transaction_cost = self.calculate_sell_cost(shares)
        net_proceeds = trade_value - transaction_cost

        # Reset position tracking
        self.position_entry_price = None
        self.position_shares = 0
        self.position_value = 0

        # Return new balance and proceeds
        new_balance = balance + net_proceeds
        return new_balance, net_proceeds, True

    def get_position_metrics(
        self,
        current_price: float
    ) -> Dict[str, float]:
        """
        Get current position metrics.

        Args:
            current_price: Current stock price

        Returns:
            Dictionary with position metrics
        """
        if self.position_shares == 0:
            return {
                'shares': 0,
                'entry_price': 0,
                'current_value': 0,
                'unrealized_pnl': 0,
                'unrealized_return': 0
            }

        current_value = self.position_shares * current_price
        unrealized_pnl = current_value - self.position_value
        unrealized_return = unrealized_pnl / self.position_value if self.position_value > 0 else 0

        return {
            'shares': self.position_shares,
            'entry_price': self.position_entry_price,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_return': unrealized_return,
            'stop_loss_price': self.position_entry_price * (1 - self.stop_loss_pct),
            'take_profit_price': self.position_entry_price * (1 + self.take_profit_pct)
        }

    def reset(self):
        """Reset guardrails state."""
        self.position_entry_price = None
        self.position_shares = 0
        self.position_value = 0

    def update_limits(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ):
        """
        Update guardrail limits.

        Args:
            stop_loss_pct: New stop loss percentage on 0-100 scale (e.g., 15 = 15%)
            take_profit_pct: New take profit percentage on 0-100 scale (e.g., 25 = 25%)
        """
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct / 100.0
        if take_profit_pct is not None:
            self.take_profit_pct = take_profit_pct / 100.0