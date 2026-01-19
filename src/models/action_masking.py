"""Action masking for invalid actions in trading with multi-buy support."""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Tuple


class ActionMasker:
    """
    Handle action masking for invalid trading actions with multi-buy/sell support.

    Action Structure:
    - Action 0: HOLD
    - Actions 1 to N: BUY actions (one for each share_increment)
    - Action N+1: BUY_MAX (buy as many shares as balance allows)
    - Actions N+2 to 2N+1: SELL actions (one for each share_increment)
    - Action 2N+2: SELL_ALL

    Example with share_increments=[10, 50, 100]:
    - 0: HOLD
    - 1: BUY_10, 2: BUY_50, 3: BUY_100
    - 4: BUY_MAX
    - 5: SELL_10, 6: SELL_50, 7: SELL_100
    - 8: SELL_ALL
    Total: 9 actions
    """

    def __init__(self, share_increments: List[int], enable_buy_max: bool = True):
        """
        Initialize action masker with share increments.

        Args:
            share_increments: List of share quantities for buy/sell actions
                             Example: [10, 50, 100]
            enable_buy_max: Whether to enable BUY_MAX action (default: True)
        """
        self.share_increments = sorted(share_increments)  # Ensure sorted
        self.enable_buy_max = enable_buy_max

        # Validate increments
        if not share_increments:
            raise ValueError("share_increments cannot be empty")
        if any(inc <= 0 for inc in share_increments):
            raise ValueError("All share_increments must be positive")

        # Calculate action space
        n_increments = len(share_increments)
        self.n_buy_actions = n_increments
        self.n_sell_actions = n_increments

        if enable_buy_max:
            self.n_actions = 1 + n_increments + 1 + n_increments + 1  # HOLD + BUYs + BUY_MAX + SELLs + SELL_ALL
        else:
            self.n_actions = 1 + n_increments + n_increments + 1  # HOLD + BUYs + SELLs + SELL_ALL

        # Action indices
        self.HOLD = 0
        self.BUY_ACTIONS = list(range(1, 1 + n_increments))

        if enable_buy_max:
            self.BUY_MAX = 1 + n_increments
            self.SELL_ACTIONS = list(range(self.BUY_MAX + 1, self.BUY_MAX + 1 + n_increments))
            self.SELL_ALL = self.BUY_MAX + 1 + n_increments
        else:
            self.BUY_MAX = None
            self.SELL_ACTIONS = list(range(1 + n_increments, 1 + 2*n_increments))
            self.SELL_ALL = 1 + 2*n_increments

        # Mapping from action index to share quantity
        self.action_to_shares_map = {}
        self.action_to_shares_map[self.HOLD] = 0

        for i, shares in enumerate(share_increments):
            buy_action = self.BUY_ACTIONS[i]
            sell_action = self.SELL_ACTIONS[i]
            self.action_to_shares_map[buy_action] = shares
            self.action_to_shares_map[sell_action] = shares

        if enable_buy_max:
            self.action_to_shares_map[self.BUY_MAX] = -2  # Special: buy max
        self.action_to_shares_map[self.SELL_ALL] = -1  # Special: sell all

    def get_action_mask(
        self,
        current_position: int,
        current_balance: float,
        current_price: float,
        min_balance: float = 0
    ) -> np.ndarray:
        """
        Get mask for valid actions based on current state.

        Args:
            current_position: Current number of shares held
            current_balance: Current cash balance
            current_price: Current stock price
            min_balance: Minimum cash balance to maintain

        Returns:
            Boolean mask array where True = valid action
        """
        mask = np.zeros(self.n_actions, dtype=bool)

        # HOLD is always valid
        mask[self.HOLD] = True

        # BUY actions - valid if can afford
        for i, shares in enumerate(self.share_increments):
            buy_action = self.BUY_ACTIONS[i]
            required_balance = shares * current_price
            if current_balance - required_balance >= min_balance:
                mask[buy_action] = True

        # BUY_MAX - valid if enabled and can afford at least 1 share
        if self.enable_buy_max and current_balance - current_price >= min_balance:
            mask[self.BUY_MAX] = True

        # SELL actions - valid only if have enough shares to sell
        if current_position > 0:
            for i, shares in enumerate(self.share_increments):
                sell_action = self.SELL_ACTIONS[i]

                # Can only sell if we have at least that many shares
                if shares <= current_position:
                    mask[sell_action] = True

            # SELL_ALL is always valid when holding any shares
            mask[self.SELL_ALL] = True

        # Ensure at least one action is valid
        if not mask.any():
            mask[self.HOLD] = True

        return mask

    def apply_action_mask(
        self,
        q_values: np.ndarray,
        action_mask: np.ndarray,
        mask_value: float = -1e9
    ) -> np.ndarray:
        """
        Apply action mask to Q-values.

        Args:
            q_values: Q-values from DQN
            action_mask: Boolean mask of valid actions
            mask_value: Value to set for invalid actions

        Returns:
            Masked Q-values
        """
        masked_q_values = q_values.copy()
        masked_q_values[~action_mask] = mask_value
        return masked_q_values

    def get_valid_actions(
        self,
        current_position: int,
        current_balance: float,
        current_price: float
    ) -> List[int]:
        """
        Get list of valid action indices.

        Args:
            current_position: Current number of shares held
            current_balance: Current cash balance
            current_price: Current stock price

        Returns:
            List of valid action indices
        """
        mask = self.get_action_mask(current_position, current_balance, current_price)
        return np.where(mask)[0].tolist()

    def action_to_shares(self, action: int) -> int:
        """
        Convert action index to number of shares.

        Args:
            action: Action index

        Returns:
            Number of shares
            - Positive: shares to buy
            - Negative: shares to sell (-1 means sell all, -2 means buy max)
            - Zero: hold
        """
        if action == self.HOLD:
            return 0
        elif action in self.BUY_ACTIONS:
            return self.action_to_shares_map[action]
        elif self.enable_buy_max and action == self.BUY_MAX:
            return -2  # Special: buy max
        elif action in self.SELL_ACTIONS:
            return -self.action_to_shares_map[action]  # Negative for sells
        elif action == self.SELL_ALL:
            return -1  # Special: sell all
        else:
            raise ValueError(f"Invalid action: {action}")

    def is_buy_action(self, action: int) -> bool:
        """Check if action is a buy action."""
        if self.enable_buy_max:
            return action in self.BUY_ACTIONS or action == self.BUY_MAX
        else:
            return action in self.BUY_ACTIONS

    def is_sell_action(self, action: int) -> bool:
        """Check if action is a sell action (including SELL_ALL)."""
        return action in self.SELL_ACTIONS or action == self.SELL_ALL

    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for action.

        Args:
            action: Action index

        Returns:
            Action name string
        """
        if action == self.HOLD:
            return "HOLD"
        elif self.enable_buy_max and action == self.BUY_MAX:
            return "BUY_MAX"
        elif action == self.SELL_ALL:
            return "SELL_ALL"
        elif action in self.BUY_ACTIONS:
            shares = self.action_to_shares_map[action]
            return f"BUY_{shares}"
        elif action in self.SELL_ACTIONS:
            shares = self.action_to_shares_map[action]
            return f"SELL_{shares}"
        else:
            return f"UNKNOWN_{action}"

    def get_random_valid_action(
        self,
        current_position: int,
        current_balance: float,
        current_price: float,
        prefer_trading: bool = False
    ) -> int:
        """
        Get a random valid action.

        Args:
            current_position: Current number of shares held
            current_balance: Current cash balance
            current_price: Current stock price
            prefer_trading: If True, prefer buy/sell over hold

        Returns:
            Random valid action index
        """
        valid_actions = self.get_valid_actions(
            current_position, current_balance, current_price
        )

        if prefer_trading and len(valid_actions) > 1:
            # Filter out HOLD if other actions available
            trading_actions = [a for a in valid_actions if a != self.HOLD]
            if trading_actions:
                valid_actions = trading_actions

        return np.random.choice(valid_actions)

    def create_tf_mask_layer(self) -> tf.keras.layers.Layer:
        """
        Create TensorFlow layer for action masking.

        Returns:
            Custom Keras layer for masking
        """
        class MaskingLayer(tf.keras.layers.Layer):
            def __init__(self, mask_value=-1e9, **kwargs):
                super().__init__(**kwargs)
                self.mask_value = mask_value

            def call(self, inputs):
                q_values, mask = inputs
                # Apply mask
                return tf.where(
                    mask,
                    q_values,
                    tf.ones_like(q_values) * self.mask_value
                )

        return MaskingLayer()

    def get_action_distribution(self, actions: List[int]) -> Dict[str, int]:
        """
        Get distribution of actions.

        Args:
            actions: List of action indices

        Returns:
            Dictionary with action counts
        """
        distribution = {
            'HOLD': 0,
            'BUY': 0,
            'SELL': 0,
            'SELL_ALL': 0
        }

        for action in actions:
            if action == self.HOLD:
                distribution['HOLD'] += 1
            elif action == self.SELL_ALL:
                distribution['SELL_ALL'] += 1
            elif action in self.BUY_ACTIONS:
                distribution['BUY'] += 1
            elif action in self.SELL_ACTIONS:
                distribution['SELL'] += 1

        return distribution

    def validate_action(
        self,
        action: int,
        current_position: int,
        current_balance: float,
        current_price: float
    ) -> bool:
        """
        Validate if an action is legal in current state.

        Args:
            action: Action to validate
            current_position: Current number of shares held
            current_balance: Current cash balance
            current_price: Current stock price

        Returns:
            True if action is valid
        """
        mask = self.get_action_mask(current_position, current_balance, current_price)
        return mask[action]

    def get_action_summary(self) -> str:
        """Get a summary of the action space."""
        lines = [
            f"Action Space Summary:",
            f"  Total actions: {self.n_actions}",
            f"  Share increments: {self.share_increments}",
            f"  BUY_MAX enabled: {self.enable_buy_max}",
            f"",
            f"Action Mapping:",
            f"  {self.HOLD}: HOLD"
        ]

        for i, shares in enumerate(self.share_increments):
            buy_action = self.BUY_ACTIONS[i]
            lines.append(f"  {buy_action}: BUY_{shares}")

        if self.enable_buy_max:
            lines.append(f"  {self.BUY_MAX}: BUY_MAX")

        for i, shares in enumerate(self.share_increments):
            sell_action = self.SELL_ACTIONS[i]
            lines.append(f"  {sell_action}: SELL_{shares}")

        lines.append(f"  {self.SELL_ALL}: SELL_ALL")

        return "\n".join(lines)
