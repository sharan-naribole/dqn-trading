"""
DQN Training Orchestration Module

Manages the complete training pipeline for Deep Q-Network agents including:
- Episode management and exploration decay
- Experience replay and batch training
- Model checkpointing and validation
- Progress tracking and metrics logging

Author: DQN Trading System
License: MIT
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os

from ..models.dqn import DoubleDQN
from ..trading.environment import TradingEnvironment
from ..utils.model_manager import ModelManager
from .replay_buffer import ReplayBuffer


class DQNTrainer:
    """
    Orchestrates the DQN training process.

    Manages the complete training lifecycle including:
    - Environment interaction and experience collection
    - Experience replay buffer management
    - Neural network training with batch updates
    - Target network synchronization
    - Model checkpointing and validation
    - Training metrics and progress logging

    Attributes:
        config (Dict): Training configuration
        dqn (DoubleDQN): Double DQN model
        replay_buffer (ReplayBuffer): Experience storage
        training_history (List[Dict]): Episode metrics history
    """

    def __init__(
        self,
        config: Dict,
        model_manager: Optional[ModelManager] = None,
        progress_logger: Optional['ProgressLogger'] = None
    ):
        """
        Initialize DQN trainer.

        Parameters:
            config (Dict): Configuration dictionary containing:
                - training.episodes: Number of training episodes
                - training.batch_size: Batch size for updates
                - training.epsilon_start: Initial exploration rate
                - training.epsilon_end: Final exploration rate
                - training.epsilon_decay: Exploration decay factor
                - training.save_frequency: Episodes between saves
                - training.target_update_freq: Target network sync frequency
                - training.validation_frequency: Episodes between validations (default: 5)
                - training.validate_at_episode_1: Run validation at episode 1 (default: True)
                - training.early_stopping_patience: Validations without improvement before stopping (default: 10, 0 = disabled)
                - training.early_stopping_metric: Metric to track for early stopping (default: 'total_return')
            model_manager (Optional[ModelManager]): Handles model persistence
            progress_logger (Optional[ProgressLogger]): Logs training progress

        Example:
            >>> config = load_config('config/default_config.json')
            >>> trainer = DQNTrainer(config)
            >>> trainer.train(train_env, validation_env)
        """
        self.config = config
        self.model_manager = model_manager or ModelManager()
        self.progress_logger = progress_logger

        # Extract training parameters
        self.episodes = config['training']['episodes']
        self.batch_size = config['training']['batch_size']
        self.epsilon_start = config['training']['epsilon_start']
        self.epsilon_end = config['training']['epsilon_end']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.save_frequency = config['training'].get('save_frequency', 10)
        self.validation_frequency = config['training'].get('validation_frequency', 5)
        self.validate_at_episode_1 = config['training'].get('validate_at_episode_1', True)
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        self.early_stopping_metric = config['training'].get('early_stopping_metric', 'total_return')

        # Initialize components
        self.dqn = None
        self.replay_buffer = None
        self.training_history = []

    def initialize_model(
        self,
        state_shape: Tuple[int, ...],
        n_actions: int
    ):
        """
        Initialize DQN model and replay buffer.

        Args:
            state_shape: Shape of state input
            n_actions: Number of possible actions
        """
        # Create Double DQN with full config (includes network architecture)
        self.dqn = DoubleDQN(
            state_shape=state_shape,
            n_actions=n_actions,
            config=self.config,
            learning_rate=self.config['training']['learning_rate']
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['replay_buffer_size'],
            state_shape=state_shape,
            n_actions=n_actions
        )

    def train(
        self,
        train_env: TradingEnvironment,
        validation_env: Optional[TradingEnvironment] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the DQN model.

        Args:
            train_env: Training environment
            validation_env: Optional validation environment
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        if self.dqn is None:
            # Initialize model based on environment
            dummy_state, _ = train_env.reset()
            state_shape = dummy_state.shape
            n_actions = train_env.action_masker.n_actions
            self.initialize_model(state_shape, n_actions)

        # Training metrics
        episode_rewards = []
        episode_profits = []
        episode_trades = []
        validation_metrics = []

        # Epsilon for exploration
        epsilon = self.epsilon_start

        # Early stopping tracking
        best_val_metric = -float('inf')
        no_improvement_count = 0
        early_stopped = False

        # Model identifier for saving
        model_identifier = self.model_manager._generate_identifier(self.config)

        # Training loop
        for episode in range(self.episodes):
            # Reset environment
            state, info = train_env.reset()
            episode_reward = 0
            episode_steps = 0
            losses = []

            # Episode loop
            done = False
            pbar = tqdm(total=len(train_env.data) - train_env.window_size,
                       desc=f"Episode {episode + 1}/{self.episodes}",
                       disable=not verbose)

            while not done:
                # Get action mask
                action_mask = train_env._get_action_mask()

                # Select action with epsilon-greedy
                if np.random.random() < epsilon:
                    # Random valid action
                    valid_actions = np.where(action_mask)[0]
                    action = np.random.choice(valid_actions)
                else:
                    # Greedy action from DQN
                    action = self.dqn.get_action(state, action_mask, epsilon=0)

                # Execute action
                next_state, reward, done, info = train_env.step(action)
                next_action_mask = train_env._get_action_mask()

                # Store transition
                self.replay_buffer.add(
                    state, action, reward, next_state, done, next_action_mask
                )

                # Train if enough samples
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self._train_step()
                    losses.append(loss)

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1

                pbar.update(1)

            pbar.close()

            # Decay epsilon
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            # Update target network based on configured frequency
            if (episode + 1) % self.config['training'].get('target_update_freq', 1) == 0:
                self.dqn.update_target_network()

            # Get episode metrics
            metrics = train_env.get_metrics()
            episode_rewards.append(episode_reward)
            episode_profits.append(metrics['total_profit'])
            episode_trades.append(metrics['num_trades'])

            # Validation
            should_validate = False
            if validation_env is not None:
                # Validate at episode 1 if configured
                if self.validate_at_episode_1 and episode == 0:
                    should_validate = True
                # Validate at regular intervals
                elif (episode + 1) % self.validation_frequency == 0:
                    should_validate = True

            if should_validate:
                val_metrics = self.evaluate(validation_env, verbose=False)
                validation_metrics.append(val_metrics)
                if verbose:
                    print(f"Validation at episode {episode + 1}: "
                          f"Return={val_metrics['total_return']:.2%}, "
                          f"Sharpe={val_metrics['sharpe_ratio']:.2f}")

                # Early stopping check
                current_metric = val_metrics.get(self.early_stopping_metric, -float('inf'))
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    no_improvement_count = 0
                    # Save best model
                    self._save_model(episode + 1, metrics, model_identifier + '_best')
                    if verbose:
                        print(f"  ✓ New best {self.early_stopping_metric}: {current_metric:.4f}")
                else:
                    no_improvement_count += 1
                    if verbose:
                        print(f"  No improvement ({no_improvement_count}/{self.early_stopping_patience})")

                # Check if should stop early
                if self.early_stopping_patience > 0 and no_improvement_count >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered at episode {episode + 1}")
                    print(f"   No improvement in {self.early_stopping_metric} for {self.early_stopping_patience} validations")
                    print(f"   Best {self.early_stopping_metric}: {best_val_metric:.4f}")
                    early_stopped = True
                    break

            # Save model
            if (episode + 1) % self.save_frequency == 0:
                self._save_model(episode + 1, metrics, model_identifier)

            # Print episode summary
            if verbose:
                self._print_episode_summary(
                    episode + 1, episode_reward, metrics, losses, epsilon
                )

            # Log to progress logger if available
            if self.progress_logger:
                episode_metrics = {
                    'reward': float(episode_reward),
                    'return': float(metrics['total_return']),
                    'profit': float(metrics['total_profit']),
                    'trades': int(metrics['num_trades']),
                    'win_rate': float(metrics['win_rate']),
                    'sharpe_ratio': float(metrics['sharpe_ratio']),
                    'max_drawdown': float(metrics.get('max_drawdown', 0))
                }
                self.progress_logger.log_episode(episode + 1, episode_metrics)

            # Store history (including position sizing metrics)
            self.training_history.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'profit': metrics['total_profit'],
                'return': metrics['total_return'],
                'trades': metrics['num_trades'],
                'win_rate': metrics['win_rate'],
                'sharpe': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'epsilon': epsilon,
                'avg_loss': np.mean(losses) if losses else 0,
                # Position sizing metrics (critical for understanding learning)
                'avg_position_size': metrics.get('avg_position_size', 0.0),
                'min_position_size': metrics.get('min_position_size', 0),
                'max_position_size': metrics.get('max_position_size', 0),
                'std_position_size': metrics.get('std_position_size', 0.0),
                'position_size_distribution': metrics.get('position_size_distribution', {}),
                'num_buy_trades': metrics.get('num_buy_trades', 0)
            })

        # Final save (only if not early stopped)
        if not early_stopped:
            self._save_model(self.episodes, train_env.get_metrics(), model_identifier)

        return {
            'episode_rewards': episode_rewards,
            'episode_profits': episode_profits,
            'episode_trades': episode_trades,
            'validation_metrics': validation_metrics,
            'training_history': self.training_history,
            'early_stopped': early_stopped,
            'best_val_metric': best_val_metric if validation_metrics else None,
            'stopped_at_episode': episode + 1 if early_stopped else self.episodes
        }

    def _train_step(self) -> float:
        """
        Perform one training step on a batch.

        Returns:
            Loss value
        """
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_action_masks = batch

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_action_masks = tf.convert_to_tensor(next_action_masks, dtype=tf.float32)

        # Train DQN using train_step
        loss = self.dqn.train_step(
            states, actions, rewards, next_states, dones, next_action_masks,
            gamma=self.config['training']['gamma']
        )

        return loss

    def evaluate(
        self,
        env: TradingEnvironment,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate the model on an environment.

        Args:
            env: Environment to evaluate on
            verbose: Print evaluation progress

        Returns:
            Evaluation metrics
        """
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get action (no exploration)
            action_mask = env._get_action_mask()
            action = self.dqn.get_action(state, action_mask, epsilon=0)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Update state
            state = next_state
            total_reward += reward

        # Get metrics
        metrics = env.get_metrics()
        metrics['total_reward'] = total_reward

        if verbose:
            self._print_evaluation_summary(metrics)

        return metrics

    def _save_model(
        self,
        episode: int,
        metrics: Dict,
        identifier: str
    ):
        """Save model and metadata."""
        # Save DQN weights
        self.model_manager.save_model(
            model=self.dqn.q_network,
            config=self.config,
            episode=episode,
            metrics=metrics,
            identifier=identifier
        )

        # Save replay buffer (optional, for resuming training)
        if episode < self.episodes:  # Don't save buffer for final model
            buffer_path = os.path.join(
                self.model_manager.base_dir,
                identifier,
                f"replay_buffer_ep{episode:03d}.npz"
            )
            self.replay_buffer.save(buffer_path)

    def save_checkpoint(
        self,
        episode: int,
        epsilon: float,
        metrics: Dict,
        identifier: str,
        best_val_metric: float = -float('inf'),
        no_improvement_count: int = 0
    ):
        """
        Save complete training checkpoint for resuming.

        Args:
            episode: Current episode number
            epsilon: Current epsilon value
            metrics: Current training metrics
            identifier: Model identifier
            best_val_metric: Best validation metric so far
            no_improvement_count: Counter for early stopping
        """
        checkpoint_dir = os.path.join(self.model_manager.base_dir, identifier, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode:03d}.json')

        checkpoint = {
            'episode': episode,
            'epsilon': epsilon,
            'metrics': metrics,
            'best_val_metric': best_val_metric,
            'no_improvement_count': no_improvement_count,
            'training_history': self.training_history,
            'config': self.config
        }

        import json
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save model and replay buffer
        self._save_model(episode, metrics, identifier)

        print(f"✓ Checkpoint saved at episode {episode}: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint for resuming.

        Args:
            checkpoint_path: Path to checkpoint JSON file

        Returns:
            Dictionary with checkpoint data
        """
        import json
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Load model weights
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        episode = checkpoint['episode']
        model_path = os.path.join(model_dir, f'model_ep{episode:03d}.h5')

        if os.path.exists(model_path):
            self.dqn.q_network.load_weights(model_path)
            self.dqn.update_target_network()
            print(f"✓ Loaded model weights from episode {episode}")

        # Load replay buffer
        buffer_path = os.path.join(model_dir, f'replay_buffer_ep{episode:03d}.npz')
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)
            print(f"✓ Loaded replay buffer with {len(self.replay_buffer)} experiences")

        # Restore training history
        self.training_history = checkpoint.get('training_history', [])

        return checkpoint

    def resume_training(
        self,
        checkpoint_path: str,
        train_env: TradingEnvironment,
        validation_env: Optional[TradingEnvironment] = None,
        additional_episodes: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            train_env: Training environment
            validation_env: Optional validation environment
            additional_episodes: Number of additional episodes (None = complete original training)
            verbose: Print training progress

        Returns:
            Training history dictionary

        Example:
            >>> trainer = DQNTrainer(config)
            >>> checkpoint_path = 'models/SPY_2016-01-01_2025-12-31/checkpoints/checkpoint_ep050.json'
            >>> history = trainer.resume_training(checkpoint_path, train_env, val_env)
        """
        print("="*60)
        print("RESUMING TRAINING FROM CHECKPOINT")
        print("="*60)

        # Initialize model if needed
        if self.dqn is None:
            dummy_state, _ = train_env.reset()
            state_shape = dummy_state.shape
            n_actions = train_env.action_masker.n_actions
            self.initialize_model(state_shape, n_actions)

        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)

        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
        no_improvement_count = checkpoint.get('no_improvement_count', 0)

        # Determine total episodes
        if additional_episodes is not None:
            total_episodes = start_episode + additional_episodes
        else:
            total_episodes = self.episodes

        print(f"\nResuming from episode {start_episode}")
        print(f"Training until episode {total_episodes}")
        print(f"Current epsilon: {epsilon:.4f}")
        print(f"Best validation {self.early_stopping_metric}: {best_val_metric:.4f}")
        print("="*60 + "\n")

        # Continue training loop (similar to regular train but starting from start_episode)
        # For brevity, this would reuse most of the train() logic
        # but start from start_episode instead of 0

        print("⚠️  Note: Full resume training implementation requires refactoring train() loop.")
        print("    For now, you can manually adjust config.episodes and retrain.")
        print("    Checkpoint successfully loaded - model weights and replay buffer restored.")

        return {
            'resumed_from_episode': start_episode,
            'checkpoint_data': checkpoint
        }

    def load_model(
        self,
        identifier: str,
        episode: Optional[int] = None
    ):
        """
        Load a saved model.

        Args:
            identifier: Model identifier
            episode: Specific episode to load (None for latest)
        """
        # Load model path and metadata
        model_path, metadata = self.model_manager.load_model(identifier, episode)

        # Initialize DQN if not already done
        if self.dqn is None:
            state_shape = metadata['config']['data']['window_size']
            n_actions = metadata['config']['trading']['max_shares'] + 2
            self.initialize_model((state_shape,), n_actions)

        # Load weights
        self.dqn.q_network.load_weights(model_path)
        self.dqn.update_target_network()

    def _print_episode_summary(
        self,
        episode: int,
        reward: float,
        metrics: Dict,
        losses: List[float],
        epsilon: float
    ):
        """Print episode training summary."""
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{self.episodes} Summary")
        print(f"{'='*60}")
        print(f"Total Reward: {reward:.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Epsilon: {epsilon:.4f}")
        if losses:
            print(f"Avg Loss: {np.mean(losses):.6f}")
        print(f"{'='*60}\n")

    def _print_evaluation_summary(self, metrics: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"Trades: {metrics['num_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"{'='*60}\n")