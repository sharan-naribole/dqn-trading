"""Quick test script to verify the system components."""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.data.collector import DataCollector
from src.data.splitter import DataSplitter
from src.features.engineer import FeatureEngineer
from src.features.normalizer import RollingNormalizer
from src.trading.environment import TradingEnvironment
from src.training.trainer import DQNTrainer


def test_components():
    """Test basic functionality of all components."""

    print("Testing DQN Trading System Components\n" + "="*50)

    # 1. Test Configuration
    print("\n1. Testing Configuration Loader...")
    try:
        # Test loading data config
        data_config = ConfigLoader('config/dry_run/data_config.json')
        print(f"✓ Data config loaded")
        print(f"  - Ticker: {data_config.config['ticker']}")
        print(f"  - Date range: {data_config.config['start_date']} to {data_config.config['end_date']}")

        # Test loading and merging with trading config
        import json
        with open('config/dry_run/trading_dry_run.json', 'r') as f:
            trading_config = json.load(f)
        config_dict = {**data_config.config, **trading_config}
        print(f"✓ Trading config loaded: {trading_config['strategy_name']}")
        print(f"  - Episodes: {trading_config['training']['episodes']}")

        # Create a config object for the rest of the tests
        class MergedConfig:
            def __init__(self, cfg):
                self.config = cfg
        config = MergedConfig(config_dict)
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

    # 2. Test Data Collection
    print("\n2. Testing Data Collection...")
    try:
        collector = DataCollector(config.config)
        spy_data, vix_data = collector.collect_data()
        print(f"✓ Data collected:")
        print(f"  - SPY records: {len(spy_data)}")
        print(f"  - VIX records: {len(vix_data)}")
    except Exception as e:
        print(f"✗ Data collection error: {e}")
        return False

    # 3. Test Feature Engineering
    print("\n3. Testing Feature Engineering...")
    try:
        combined_data = collector.combine_data(spy_data, vix_data)
        engineer = FeatureEngineer(config.config)
        featured_data = engineer.create_features(combined_data, verbose=False)
        feature_names = engineer.get_feature_names(featured_data)
        print(f"✓ Features created: {len(feature_names)} features")
    except Exception as e:
        print(f"✗ Feature engineering error: {e}")
        return False

    # 4. Test Data Splitting (identify train/val/test periods)
    print("\n4. Testing Data Splitting...")
    try:
        splitter = DataSplitter(config.config)
        splits = splitter.split_data(featured_data, verbose=False)
        print(f"✓ Data split into periods:")
        print(f"  - Train: {len(splits['train'])} records")
        print(f"  - Validation: {len(splits['validation'])} periods")
        print(f"  - Test: {len(splits['test'])} records")
    except Exception as e:
        print(f"✗ Data splitting error: {e}")
        return False

    # 5. Test Normalization (backward-only rolling on continuous timeline)
    print("\n5. Testing Normalization...")
    try:
        normalizer = RollingNormalizer(config.config)
        # Reconstruct chronological timeline for stateful normalization
        all_data = pd.concat([
            splits['train'],
            *splits['validation'],
            splits['test']
        ]).sort_index()
        # Apply backward-only rolling normalization (no lookahead bias)
        # Each point normalized using ONLY previous 30 days (center=False)
        all_normalized = normalizer.fit_transform(all_data, feature_names)
        # Extract normalized train split
        train_normalized = all_normalized.loc[splits['train'].index]
        print(f"✓ Data normalized (backward-only continuous timeline):")
        print(f"  - Train normalized: {train_normalized.shape}")
        print(f"  - Method: Rolling Z-score, window=30, center=False")
        print(f"  - No lookahead bias: Each point uses only past data")
        # Use train_normalized for subsequent tests
        splits['train'] = train_normalized
    except Exception as e:
        print(f"✗ Normalization error: {e}")
        return False

    # 6. Test Trading Environment
    print("\n6. Testing Trading Environment...")
    try:
        env = TradingEnvironment(
            splits['train'][:100],  # Use small subset for testing
            feature_names,
            config.config,
            mode='train'
        )
        state, info = env.reset()
        print(f"✓ Environment created:")
        print(f"  - State shape: {state.shape}")
        print(f"  - Action space: {env.action_masker.n_actions} actions")
        print(f"  - Starting balance: ${info['balance']:.2f}")
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False

    # 7. Test DQN Model
    print("\n7. Testing DQN Model...")
    try:
        import numpy as np
        from src.models.dqn import DoubleDQN
        dqn = DoubleDQN(
            state_shape=state.shape,
            n_actions=env.action_masker.n_actions,
            config=config.config,
            learning_rate=config.config['training']['learning_rate']
        )

        # Test forward pass
        q_values = dqn.q_network(state[np.newaxis, ...])
        print(f"✓ DQN model created:")
        print(f"  - Q-values shape: {q_values.shape}")
        print(f"  - Q-network parameters: {len(dqn.q_network.get_weights())}")
    except Exception as e:
        print(f"✗ DQN model error: {e}")
        return False

    # 8. Test Action Masking
    print("\n8. Testing Action Masking...")
    try:
        action_mask = env._get_action_mask()
        valid_actions = env.action_masker.get_valid_actions(
            env.shares_held, env.balance, env._get_current_price()
        )
        print(f"✓ Action masking working:")
        print(f"  - Valid actions: {valid_actions}")
        print(f"  - Action mask shape: {action_mask.shape}")
    except Exception as e:
        print(f"✗ Action masking error: {e}")
        return False

    # 9. Test One Training Step
    print("\n9. Testing Training Step...")
    try:
        trainer = DQNTrainer(config.config)
        trainer.initialize_model(state.shape, env.action_masker.n_actions)

        # Collect some experiences
        for _ in range(5):
            action = env.action_masker.get_random_valid_action(
                env.shares_held, env.balance, env._get_current_price()
            )
            next_state, reward, done, info = env.step(action)
            if done:
                break

        print(f"✓ Training components initialized")
        print(f"  - Replay buffer size: {len(trainer.replay_buffer)}")
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

    print("\n" + "="*50)
    print("✓ All components tested successfully!")
    return True


if __name__ == "__main__":
    success = test_components()
    if success:
        print("\nSystem is ready for training!")
        print("\nNext steps:")
        print("1. Adjust configuration in config/default_config.json")
        print("2. Run training with the main notebook")
        print("3. Monitor training progress and metrics")
    else:
        print("\n✗ Some components failed. Please check the errors above.")