"""Experience replay buffer for DQN training."""

import numpy as np
from typing import Tuple, Optional
import random


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(
        self,
        capacity: int = 10000,
        state_shape: Tuple[int, ...] = None,
        n_actions: int = None
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of state arrays
            n_actions: Number of possible actions
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.position = 0
        self.size = 0

        # Pre-allocate arrays for efficiency
        if state_shape:
            self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
            self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        else:
            self.states = []
            self.next_states = []

        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Action masks for next states
        if n_actions:
            self.next_action_masks = np.zeros((capacity, n_actions), dtype=bool)
        else:
            self.next_action_masks = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            next_action_mask: Valid actions mask for next state
        """
        if self.state_shape:
            self.states[self.position] = state
            self.next_states[self.position] = next_state
        else:
            if len(self.states) < self.capacity:
                self.states.append(state)
                self.next_states.append(next_state)
            else:
                self.states[self.position] = state
                self.next_states[self.position] = next_state

        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done

        if next_action_mask is not None:
            if self.n_actions:
                self.next_action_masks[self.position] = next_action_mask
            else:
                if len(self.next_action_masks) < self.capacity:
                    self.next_action_masks.append(next_action_mask)
                else:
                    self.next_action_masks[self.position] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, next_action_masks)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer. Have {self.size}, requested {batch_size}")

        indices = np.random.choice(self.size, batch_size, replace=False)

        if self.state_shape:
            states = self.states[indices]
            next_states = self.next_states[indices]
        else:
            states = np.array([self.states[i] for i in indices])
            next_states = np.array([self.next_states[i] for i in indices])

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        if self.n_actions:
            next_action_masks = self.next_action_masks[indices]
        elif self.next_action_masks:
            next_action_masks = np.array([self.next_action_masks[i] for i in indices])
        else:
            next_action_masks = None

        return states, actions, rewards, next_states, dones, next_action_masks

    def sample_prioritized(
        self,
        batch_size: int,
        priorities: Optional[np.ndarray] = None,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> Tuple:
        """
        Sample with prioritized experience replay.

        Args:
            batch_size: Number of transitions to sample
            priorities: Priority values for each transition
            alpha: Priority exponent
            beta: Importance sampling exponent

        Returns:
            Tuple of (batch_data, weights, indices)
        """
        if priorities is None:
            # Use uniform sampling if no priorities provided
            return self.sample(batch_size) + (None, None)

        # Calculate sampling probabilities
        priorities = priorities[:self.size]
        probs = priorities ** alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Get batch data
        batch_data = self.sample(batch_size)

        return batch_data + (weights, indices)

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.size >= self.capacity

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0
            }

        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'mean_reward': np.mean(self.rewards[:self.size]),
            'std_reward': np.std(self.rewards[:self.size]),
            'done_ratio': np.mean(self.dones[:self.size])
        }

    def save(self, filepath: str):
        """
        Save buffer to file.

        Args:
            filepath: Path to save buffer
        """
        data = {
            'states': self.states[:self.size] if self.state_shape else self.states,
            'next_states': self.next_states[:self.size] if self.state_shape else self.next_states,
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size],
            'next_action_masks': self.next_action_masks[:self.size] if self.n_actions else self.next_action_masks,
            'position': self.position,
            'size': self.size,
            'capacity': self.capacity
        }
        np.savez_compressed(filepath, **data)

    def load(self, filepath: str):
        """
        Load buffer from file.

        Args:
            filepath: Path to load buffer from
        """
        data = np.load(filepath, allow_pickle=True)

        self.size = int(data['size'])
        self.position = int(data['position'])

        if self.state_shape:
            self.states[:self.size] = data['states']
            self.next_states[:self.size] = data['next_states']
        else:
            self.states = data['states'].tolist()
            self.next_states = data['next_states'].tolist()

        self.actions[:self.size] = data['actions']
        self.rewards[:self.size] = data['rewards']
        self.dones[:self.size] = data['dones']

        if 'next_action_masks' in data:
            if self.n_actions:
                self.next_action_masks[:self.size] = data['next_action_masks']
            else:
                self.next_action_masks = data['next_action_masks'].tolist()