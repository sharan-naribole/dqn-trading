# Configuration Parameters Reference

Complete documentation of all configuration parameters for the DQN Trading System.

## Table of Contents

1. [Configuration Structure](#configuration-structure)
2. [Data Configuration Parameters](#data-configuration-parameters)
3. [Trading Configuration Parameters](#trading-configuration-parameters)
4. [Network Architecture Parameters](#network-architecture-parameters)
5. [Training Parameters](#training-parameters)
6. [Complete Examples](#complete-examples)

---

## Configuration Structure

The system uses a **two-part configuration structure** for each project:

```
config/
├── dry_run/                      # Quick test project
│   ├── data_config.json         # Data settings
│   └── trading_dry_run.json     # Trading strategy
└── default_run/                  # Full training project
    ├── data_config.json         # Data settings
    └── trading_baseline.json    # Trading strategy
```

### In the Main Notebook:
```python
PROJECT_FOLDER = 'dry_run'    # Select project folder
TEST_MODE = False             # False: train, True: load existing models
```

---

## Data Configuration Parameters

Located in `{PROJECT_FOLDER}/data_config.json`

### Top-Level Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `ticker` | string | Yes | Stock ticker symbol | "SPY" |
| `start_date` | string | Yes | Data start date (YYYY-MM-DD) | "2024-01-01" |
| `end_date` | string | Yes | Data end date (YYYY-MM-DD) | "2025-12-31" |

### Data Processing (`data`)

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `window_size` | int | 5 | Lookback window for state | 1-30 |
| `normalization_window` | int | 30 | Rolling normalization window | 5-100 |

### Technical Indicators (`data.indicators`)

| Parameter | Type | Default | Description | Typical Range |
|-----------|------|---------|-------------|---------------|
| `bollinger_period` | int | 20 | Bollinger Bands period | 10-30 |
| `bollinger_std` | float | 2 | Standard deviations | 1-3 |
| `ema_short` | int | 8 | Short EMA period | 5-15 |
| `ema_medium` | int | 21 | Medium EMA period | 15-30 |
| `sma_short` | int | 50 | Short SMA period | 20-50 |
| `sma_long` | int | 200 | Long SMA period | 100-200 |
| `adx_period` | int | 14 | ADX period | 10-20 |
| `rsi_period` | int | 14 | RSI period | 10-20 |

### Validation Settings (`validation`)

| Parameter | Type | Default | Description | Options |
|-----------|------|---------|-------------|---------|
| `n_periods` | int | 3 | Number of validation periods | 1-5 |
| `period_unit` | string | "year" | Unit for periods | "year", "month", "week" |
| `random_seed` | int | 42 | Random seed for reproducibility | Any integer |

### Test Settings (`test`)

| Parameter | Type | Default | Description | Options |
|-----------|------|---------|-------------|---------|
| `period_duration` | int | 1 | Test period duration | 1-12 |
| `period_unit` | string | "year" | Unit for test period | "year", "month" |

### Output Settings (`output`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | string | "models" | Model storage directory |
| `results_dir` | string | "results" | Results storage directory |
| `data_dir` | string | "data" | Data cache directory |

---

## Trading Configuration Parameters

Located in `{PROJECT_FOLDER}/trading_*.json`

### Metadata

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `experiment_name` | string | Yes | Unique experiment identifier | "baseline" |
| `strategy_name` | string | Yes | Display name for plots | "Baseline (20% SL/TP)" |
| `description` | string | No | Strategy description | "Standard risk management" |

### Trading Parameters (`trading`)

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `max_shares` | int | 10 | Maximum shares per trade | 1-1000 |
| `share_step` | int | 1 | Share quantity increment (action space reduction) | 1-100 |
| `starting_balance` | float | 100000 | Initial cash balance | >0 |
| `idle_reward` | float | -0.001 | Penalty for holding | -1 to 0 |
| `buy_reward_per_share` | float | 0.0 | Bonus per share bought | 0-1 |
| `buy_transaction_cost_per_share` | float | 0.01 | Cost per share bought | 0-1 |
| `sell_transaction_cost_per_share` | float | 0.01 | Cost per share sold | 0-1 |
| `stop_loss_pct` | float | 20 | Stop-loss percentage | 0-100 |
| `take_profit_pct` | float | 20 | Take-profit percentage | 0-10000 |

#### Action Space and share_step

The `share_step` parameter controls the granularity of buy actions, dramatically reducing the action space:

**Formula:** `n_actions = (max_shares / share_step) + 2`

**Examples:**
- `max_shares=100, share_step=1`: 102 actions (Hold, Buy1-100, Sell)
- `max_shares=100, share_step=10`: 12 actions (Hold, Buy10, Buy20, ..., Buy100, Sell)
- `max_shares=100, share_step=25`: 6 actions (Hold, Buy25, Buy50, Buy75, Buy100, Sell)

**When to use larger step sizes:**
- Large max_shares values (>50) → use step=10 or higher
- Faster training desired → larger steps reduce complexity
- Fine-grained control needed → use step=1

#### Disabling Stop-Loss and Take-Profit

To disable guardrails and allow the DQN agent full control over trading decisions:

- **No Stop-Loss**: Set `stop_loss_pct: 100` (position must go to $0 to trigger)
- **No Take-Profit**: Set `take_profit_pct: 10000` (100x gain required to trigger)

Example configuration for no guardrails:
```json
"trading": {
  "stop_loss_pct": 100,    // Effectively disabled (stock must go to $0)
  "take_profit_pct": 10000  // Effectively disabled (100x gain needed)
}
```

---

## Network Architecture Parameters

Located in `trading_*.json` under `network` key.

### Architecture Settings

| Parameter | Type | Default | Description | Options |
|-----------|------|---------|-------------|---------|
| `architecture` | string | "dueling" | Network type | "standard", "dueling" |
| `shared_layers` | list | [256, 128] | Shared layer sizes | List of integers |
| `value_layers` | list | [128] | Value stream layers (dueling) | List of integers |
| `advantage_layers` | list | [128] | Advantage stream layers (dueling) | List of integers |
| `activation` | string | "relu" | Activation function | "relu", "tanh", "elu" |
| `dropout_rate` | float | 0.0 | Dropout rate | 0.0-0.5 |
| `batch_norm` | bool | false | Use batch normalization | true/false |

---

## Training Parameters

Located in `trading_*.json` under `training` key.

### Core Training Settings

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `episodes` | int | 100 | Number of training episodes | 1-1000 |
| `batch_size` | int | 64 | Batch size for training | 16-256 |
| `replay_buffer_size` | int | 10000 | Experience replay buffer size | 1000-100000 |
| `target_update_freq` | int | 1 | Target network update frequency | 1-10 |
| `save_frequency` | int | 10 | Save model every N episodes | 1-100 |

### Exploration Settings

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `epsilon_start` | float | 1.0 | Initial exploration rate | 0.5-1.0 |
| `epsilon_end` | float | 0.01 | Final exploration rate | 0.001-0.1 |
| `epsilon_decay` | float | 0.995 | Epsilon decay per episode | 0.9-0.999 |

### Optimization Settings

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `learning_rate` | float | 0.001 | Learning rate | 0.0001-0.01 |
| `gamma` | float | 0.99 | Discount factor | 0.9-0.999 |
| `optimizer` | string | "adam" | Optimizer type | "adam", "rmsprop", "sgd" |

### Validation Settings

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `validation_frequency` | int | 5 | Validate every N episodes | 1-50 |
| `validate_at_episode_1` | bool | true | Validate before training | true/false |
| `early_stopping_patience` | int | 10 | Episodes without improvement | 0-50 |
| `early_stopping_metric` | string | "total_return" | Metric for early stopping | "total_return", "sharpe_ratio" |

---

## Complete Examples

### Example 1: Dry Run Configuration

**`config/dry_run/data_config.json`:**
```json
{
  "_comment": "Dry run data configuration - minimal data for quick testing",
  "ticker": "SPY",
  "start_date": "2024-01-01",
  "end_date": "2025-12-31",
  "data": {
    "window_size": 1,
    "normalization_window": 5,
    "indicators": {
      "bollinger_period": 20,
      "bollinger_std": 2,
      "ema_short": 8,
      "ema_medium": 21,
      "sma_short": 50,
      "sma_long": 200,
      "adx_period": 14,
      "rsi_period": 14
    }
  },
  "validation": {
    "n_periods": 2,
    "period_unit": "month",
    "random_seed": 42
  },
  "test": {
    "period_duration": 2,
    "period_unit": "month"
  },
  "output": {
    "model_dir": "models",
    "results_dir": "results",
    "data_dir": "data"
  }
}
```

**`config/dry_run/trading_dry_run.json`:**
```json
{
  "_comment": "Quick dry run for testing - minimal settings",
  "experiment_name": "dry_run",
  "strategy_name": "Dry Run Test",
  "description": "Fast validation test - 1 episode, small network",

  "trading": {
    "max_shares": 1,
    "starting_balance": 10000,
    "idle_reward": -0.001,
    "buy_reward_per_share": 0.0,
    "buy_transaction_cost_per_share": 0.01,
    "sell_transaction_cost_per_share": 0.01,
    "stop_loss_pct": 20,
    "take_profit_pct": 20
  },

  "network": {
    "architecture": "dueling",
    "shared_layers": [128, 64],
    "value_layers": [32],
    "advantage_layers": [32],
    "activation": "relu",
    "dropout_rate": 0.0,
    "batch_norm": false
  },

  "training": {
    "episodes": 1,
    "batch_size": 32,
    "replay_buffer_size": 1000,
    "target_update_freq": 1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "optimizer": "adam",
    "save_frequency": 1,
    "validation_frequency": 1,
    "validate_at_episode_1": true,
    "early_stopping_patience": 0,
    "early_stopping_metric": "total_return"
  }
}
```

### Example 2: Full Training Configuration

**`config/default_run/trading_baseline.json`:**
```json
{
  "_comment": "Baseline strategy - standard risk management",
  "experiment_name": "baseline",
  "strategy_name": "Baseline (20% SL/TP)",
  "description": "Baseline strategy with 20% stop-loss and take-profit",

  "trading": {
    "max_shares": 10,
    "starting_balance": 100000,
    "idle_reward": -0.001,
    "buy_reward_per_share": 0.0,
    "buy_transaction_cost_per_share": 0.01,
    "sell_transaction_cost_per_share": 0.01,
    "stop_loss_pct": 20,
    "take_profit_pct": 20
  },

  "network": {
    "architecture": "dueling",
    "shared_layers": [256, 128],
    "value_layers": [128],
    "advantage_layers": [128],
    "activation": "relu",
    "dropout_rate": 0.0,
    "batch_norm": false
  },

  "training": {
    "episodes": 500,
    "batch_size": 64,
    "replay_buffer_size": 10000,
    "target_update_freq": 1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "optimizer": "adam",
    "save_frequency": 50,
    "validation_frequency": 5,
    "validate_at_episode_1": true,
    "early_stopping_patience": 10,
    "early_stopping_metric": "total_return"
  }
}
```

### Example 3: No Guardrails Configuration

**`config/default_run/trading_no_guardrails.json`:**
```json
{
  "_comment": "No guardrails strategy - no stop-loss or take-profit",
  "experiment_name": "no_guardrails",
  "strategy_name": "No Guardrails",
  "description": "Pure DQN decision-making without stop-loss or take-profit limits",

  "trading": {
    "max_shares": 10,
    "starting_balance": 100000,
    "idle_reward": -0.001,
    "buy_reward_per_share": 0.0,
    "buy_transaction_cost_per_share": 0.01,
    "sell_transaction_cost_per_share": 0.01,
    "stop_loss_pct": 100,
    "take_profit_pct": 10000
  },

  "network": {
    "architecture": "dueling",
    "shared_layers": [256, 128],
    "value_layers": [128],
    "advantage_layers": [128],
    "activation": "relu",
    "dropout_rate": 0.0,
    "batch_norm": false
  },

  "training": {
    "episodes": 500,
    "batch_size": 64,
    "replay_buffer_size": 10000,
    "target_update_freq": 1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "optimizer": "adam",
    "save_frequency": 50,
    "validation_frequency": 5,
    "validate_at_episode_1": true,
    "early_stopping_patience": 10,
    "early_stopping_metric": "total_return"
  }
}
```

---

## Usage in Main Notebook

```python
# Select project folder
PROJECT_FOLDER = 'dry_run'    # or 'default_run'

# Choose mode
TEST_MODE = False   # False: Train from scratch
                   # True: Load existing models (must exist)

# The notebook will:
1. Load data_config.json from PROJECT_FOLDER
2. Find all trading_*.json files in PROJECT_FOLDER
3. Train/test each strategy
4. Compare all strategies on validation and test sets
5. Save results to organized folders
```

## Model Saving Frequency

Models are saved based on `save_frequency`:
- `save_frequency: 1` → Save every episode
- `save_frequency: 10` → Save every 10 episodes
- Final model always saved as `final_model.h5`

## Storage Locations

```
models/{PROJECT_FOLDER}/{experiment_name}/     # TensorFlow models
results/{PROJECT_FOLDER}/run_{timestamp}/      # Results and plots
logs/{PROJECT_FOLDER}/{run_name}/             # Training logs
data/                                         # Cached market data (shared)
```