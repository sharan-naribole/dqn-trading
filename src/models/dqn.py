"""
Double DQN Model with Configurable Architecture

Implements Deep Q-Network with support for:
- Standard DQN architecture
- Dueling DQN architecture (separate value and advantage streams)
- Configurable layer sizes and activation functions
- Optional dropout and batch normalization

Author: DQN Trading System
License: MIT
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
from tensorflow.keras import layers, Model


class DQNNetwork(Model):
    """
    Configurable DQN network supporting standard and dueling architectures.

    The network can be configured to use either:
    1. Standard architecture: Direct Q-value output
    2. Dueling architecture: Separate value and advantage streams

    Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        n_actions: int,
        network_config: Optional[Dict] = None,
        name: str = 'dqn_network'
    ):
        """
        Initialize DQN network with configurable architecture.

        Parameters:
            state_shape (Tuple[int, ...]): Shape of state input (window_size, n_features)
            n_actions (int): Number of possible actions
            network_config (Optional[Dict]): Network configuration containing:
                - architecture (str): 'standard' or 'dueling'
                - shared_layers (List[int]): Sizes for shared layers [256, 128]
                - value_layers (List[int]): Value stream layers (dueling only)
                - advantage_layers (List[int]): Advantage stream layers (dueling only)
                - activation (str): Activation function ('relu', 'tanh', 'elu')
                - dropout_rate (float): Dropout rate (0.0-0.5)
                - batch_norm (bool): Whether to use batch normalization
            name (str): Network name for TensorFlow

        Example:
            >>> config = {
            ...     'architecture': 'dueling',
            ...     'shared_layers': [256, 128],
            ...     'value_layers': [128],
            ...     'advantage_layers': [128]
            ... }
            >>> network = DQNNetwork((30, 25), 7, config)
        """
        super().__init__(name=name)

        # Default config if not provided
        if network_config is None:
            network_config = {
                'architecture': 'dueling',
                'shared_layers': [256, 128],
                'value_layers': [128],
                'advantage_layers': [128],
                'activation': 'relu',
                'dropout_rate': 0.0,
                'batch_norm': False
            }

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.config = network_config
        self.architecture = network_config.get('architecture', 'dueling')

        # Input processing
        self.flatten = layers.Flatten()

        # Build shared layers (used by both architectures)
        self.shared_layers = []
        for i, units in enumerate(network_config.get('shared_layers', [256, 128])):
            layer_list = []

            # Dense layer
            layer_list.append(
                layers.Dense(units, activation=network_config.get('activation', 'relu'),
                           name=f'shared_{i}')
            )

            # Optional batch normalization
            if network_config.get('batch_norm', False):
                layer_list.append(layers.BatchNormalization(name=f'bn_shared_{i}'))

            # Optional dropout
            dropout_rate = network_config.get('dropout_rate', 0.0)
            if dropout_rate > 0:
                layer_list.append(layers.Dropout(dropout_rate, name=f'dropout_shared_{i}'))

            self.shared_layers.append(layer_list)

        if self.architecture == 'dueling':
            # Build value stream
            self.value_layers = []
            for i, units in enumerate(network_config.get('value_layers', [128])):
                layer_list = []
                layer_list.append(
                    layers.Dense(units, activation=network_config.get('activation', 'relu'),
                               name=f'value_{i}')
                )
                if network_config.get('batch_norm', False):
                    layer_list.append(layers.BatchNormalization(name=f'bn_value_{i}'))
                if network_config.get('dropout_rate', 0.0) > 0:
                    layer_list.append(layers.Dropout(network_config['dropout_rate'],
                                                    name=f'dropout_value_{i}'))
                self.value_layers.append(layer_list)

            # Build advantage stream
            self.advantage_layers = []
            for i, units in enumerate(network_config.get('advantage_layers', [128])):
                layer_list = []
                layer_list.append(
                    layers.Dense(units, activation=network_config.get('activation', 'relu'),
                               name=f'advantage_{i}')
                )
                if network_config.get('batch_norm', False):
                    layer_list.append(layers.BatchNormalization(name=f'bn_advantage_{i}'))
                if network_config.get('dropout_rate', 0.0) > 0:
                    layer_list.append(layers.Dropout(network_config['dropout_rate'],
                                                    name=f'dropout_advantage_{i}'))
                self.advantage_layers.append(layer_list)

            # Output layers for dueling
            self.value_output = layers.Dense(1, name='value_output')
            self.advantage_output = layers.Dense(n_actions, name='advantage_output')

        else:  # Standard architecture
            # Direct Q-value output
            self.q_output = layers.Dense(n_actions, name='q_output')

    def call(self, inputs, training=False):
        """
        Forward pass through the network.

        Args:
            inputs: State tensor
            training: Whether in training mode (for dropout/batch norm)

        Returns:
            Q-values for each action
        """
        # Flatten input if needed
        x = self.flatten(inputs)

        # Pass through shared layers
        for layer_list in self.shared_layers:
            for layer in layer_list:
                x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)

        if self.architecture == 'dueling':
            # Split into value and advantage streams
            value = x
            advantage = x

            # Process value stream
            for layer_list in self.value_layers:
                for layer in layer_list:
                    value = layer(value, training=training) if hasattr(layer, 'training') else layer(value)
            value = self.value_output(value)

            # Process advantage stream
            for layer_list in self.advantage_layers:
                for layer in layer_list:
                    advantage = layer(advantage, training=training) if hasattr(layer, 'training') else layer(advantage)
            advantage = self.advantage_output(advantage)

            # Combine using dueling formula: Q = V + A - mean(A)
            q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        else:  # Standard architecture
            q_values = self.q_output(x)

        return q_values


class DoubleDQN:
    """Double DQN agent with configurable network architecture."""

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        n_actions: int,
        config: Dict,
        learning_rate: float = 0.001
    ):
        """
        Initialize Double DQN agent.

        Args:
            state_shape: Shape of state input
            n_actions: Number of possible actions
            config: Full configuration dict (must contain 'network' section)
            learning_rate: Learning rate for optimizer
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        # Extract network config
        network_config = config.get('network', {})

        # Create behavior and target networks with same architecture
        self.q_network = DQNNetwork(state_shape, n_actions, network_config, 'behavior_network')
        self.target_network = DQNNetwork(state_shape, n_actions, network_config, 'target_network')

        # Initialize optimizer based on config
        optimizer_type = config.get('training', {}).get('optimizer', 'adam').lower()
        if optimizer_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Initialize both networks
        dummy_input = tf.zeros((1,) + state_shape)
        self.q_network(dummy_input)
        self.target_network(dummy_input)

        # Copy weights to target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from behavior network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())

    def get_action(self, state: np.ndarray, valid_actions: np.ndarray, epsilon: float = 0) -> int:
        """
        Select action using epsilon-greedy policy with action masking.

        Args:
            state: Current state
            valid_actions: Binary mask of valid actions
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions)[0]
            return np.random.choice(valid_indices)
        else:
            # Greedy action from Q-network
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
            q_values = self.q_network(state_tensor, training=False)

            # Apply action mask (set invalid actions to very negative value)
            masked_q_values = q_values.numpy()[0] * valid_actions - (1 - valid_actions) * 1e9

            return np.argmax(masked_q_values)

    @tf.function
    def train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
        next_valid_actions: tf.Tensor,
        gamma: float = 0.99
    ):
        """
        Perform one training step using Double DQN algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            next_valid_actions: Batch of valid action masks for next states
            gamma: Discount factor

        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q_values = self.q_network(states, training=True)

            # Select Q-values for actions taken
            actions_one_hot = tf.one_hot(actions, self.n_actions)
            current_q_values = tf.reduce_sum(current_q_values * actions_one_hot, axis=1)

            # Double DQN: Use behavior network to select action, target network to evaluate
            next_q_values_behavior = self.q_network(next_states, training=False)

            # Apply action masking to behavior network's Q-values
            masked_next_q = next_q_values_behavior * next_valid_actions - (1 - next_valid_actions) * 1e9
            next_actions = tf.argmax(masked_next_q, axis=1)

            # Get Q-values from target network
            next_q_values_target = self.target_network(next_states, training=False)
            next_actions_one_hot = tf.one_hot(next_actions, self.n_actions)
            next_q_values = tf.reduce_sum(next_q_values_target * next_actions_one_hot, axis=1)

            # Compute targets
            targets = rewards + gamma * next_q_values * (1 - dones)

            # Compute loss
            loss = tf.keras.losses.MSE(targets, current_q_values)

        # Update weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss

    def get_config(self) -> Dict:
        """Get network configuration for saving."""
        return {
            'state_shape': self.state_shape,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'network_config': self.q_network.config
        }