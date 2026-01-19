"""Unit tests for action masking functionality."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.action_masking import ActionMasker


class TestActionMasker:
    """Test suite for ActionMasker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.max_shares = 10
        self.masker = ActionMasker(self.max_shares)

    def test_initialization(self):
        """Test ActionMasker initialization."""
        assert self.masker.max_shares == 10
        assert self.masker.n_actions == 12  # Hold + Buy1-10 + Sell
        assert self.masker.HOLD == 0
        assert self.masker.SELL == 11
        assert len(self.masker.BUY_ACTIONS) == 10

    def test_hold_always_valid(self):
        """Test that HOLD action is always valid."""
        # No position, no money
        mask = self.masker.get_action_mask(0, 0.0, 100.0)
        assert mask[self.masker.HOLD] == True

        # With position
        mask = self.masker.get_action_mask(5, 1000.0, 100.0)
        assert mask[self.masker.HOLD] == True

    def test_buy_invalid_when_holding(self):
        """Test that BUY actions are invalid when already holding."""
        mask = self.masker.get_action_mask(
            current_position=5,  # Holding 5 shares
            current_balance=10000.0,
            current_price=100.0
        )

        # All buy actions should be invalid
        for buy_action in self.masker.BUY_ACTIONS:
            assert mask[buy_action] == False

    def test_buy_valid_when_not_holding(self):
        """Test that BUY actions are valid when not holding and have balance."""
        mask = self.masker.get_action_mask(
            current_position=0,  # Not holding
            current_balance=10000.0,
            current_price=100.0
        )

        # All buy actions within budget should be valid
        for shares in range(1, 11):
            required_balance = shares * 100.0
            if required_balance <= 10000.0:
                assert mask[shares] == True

    def test_buy_limited_by_balance(self):
        """Test that BUY actions are limited by available balance."""
        mask = self.masker.get_action_mask(
            current_position=0,
            current_balance=250.0,  # Only enough for 2 shares
            current_price=100.0
        )

        # Can buy 1-2 shares
        assert mask[1] == True
        assert mask[2] == True

        # Cannot buy 3+ shares
        for shares in range(3, 11):
            assert mask[shares] == False

    def test_sell_only_valid_when_holding(self):
        """Test that SELL action is only valid when holding shares."""
        # Not holding
        mask = self.masker.get_action_mask(0, 10000.0, 100.0)
        assert mask[self.masker.SELL] == False

        # Holding shares
        mask = self.masker.get_action_mask(5, 10000.0, 100.0)
        assert mask[self.masker.SELL] == True

    def test_action_to_shares(self):
        """Test action to shares conversion."""
        assert self.masker.action_to_shares(self.masker.HOLD) == 0
        assert self.masker.action_to_shares(self.masker.SELL) == 0
        assert self.masker.action_to_shares(1) == 1
        assert self.masker.action_to_shares(5) == 5
        assert self.masker.action_to_shares(10) == 10

        # Invalid action
        with pytest.raises(ValueError):
            self.masker.action_to_shares(99)

    def test_get_action_name(self):
        """Test action name retrieval."""
        assert self.masker.get_action_name(self.masker.HOLD) == "HOLD"
        assert self.masker.get_action_name(self.masker.SELL) == "SELL"
        assert self.masker.get_action_name(1) == "BUY_1"
        assert self.masker.get_action_name(5) == "BUY_5"
        assert self.masker.get_action_name(99) == "UNKNOWN_99"

    def test_validate_action(self):
        """Test action validation."""
        # Valid actions
        assert self.masker.validate_action(self.masker.HOLD, 0, 10000.0, 100.0) == True
        assert self.masker.validate_action(1, 0, 10000.0, 100.0) == True
        assert self.masker.validate_action(self.masker.SELL, 5, 10000.0, 100.0) == True

        # Invalid actions
        assert self.masker.validate_action(1, 5, 10000.0, 100.0) == False  # Can't buy when holding
        assert self.masker.validate_action(self.masker.SELL, 0, 10000.0, 100.0) == False  # Can't sell when not holding
        assert self.masker.validate_action(10, 0, 500.0, 100.0) == False  # Insufficient balance

    def test_edge_case_zero_balance(self):
        """Test edge case with zero balance."""
        mask = self.masker.get_action_mask(0, 0.0, 100.0)

        # Can only hold
        assert mask[self.masker.HOLD] == True
        for buy_action in self.masker.BUY_ACTIONS:
            assert mask[buy_action] == False
        assert mask[self.masker.SELL] == False

    def test_edge_case_high_price(self):
        """Test edge case with very high stock price."""
        mask = self.masker.get_action_mask(0, 1000.0, 10000.0)

        # Can only hold (price too high)
        assert mask[self.masker.HOLD] == True
        for buy_action in self.masker.BUY_ACTIONS:
            assert mask[buy_action] == False

    def test_get_valid_actions(self):
        """Test getting list of valid actions."""
        valid_actions = self.masker.get_valid_actions(0, 10000.0, 100.0)

        # Should include HOLD and all buy actions within budget
        assert self.masker.HOLD in valid_actions
        assert 1 in valid_actions  # Can buy 1 share
        assert 10 in valid_actions  # Can buy up to 100 shares (10,000/100)
        assert self.masker.SELL not in valid_actions  # Not holding

    def test_action_distribution(self):
        """Test action distribution calculation."""
        actions = [0, 0, 1, 2, 3, 11]
        dist = self.masker.get_action_distribution(actions)

        assert dist['HOLD'] == 2
        assert dist['BUY'] == 3
        assert dist['SELL'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
