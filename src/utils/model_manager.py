"""Model management utilities for saving and loading DQN models."""

import os
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import tensorflow as tf
import numpy as np


class ModelManager:
    """Manage model saving, loading, and metadata."""

    def __init__(self, base_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any],
        episode: int,
        metrics: Dict[str, float] = None,
        identifier: str = None
    ) -> str:
        """
        Save model with metadata.

        Args:
            model: Keras model to save
            config: Configuration dictionary
            episode: Episode number
            metrics: Performance metrics (optional)
            identifier: Model identifier (optional, generated if not provided)

        Returns:
            Path to saved model directory
        """
        if identifier is None:
            identifier = self._generate_identifier(config)

        # Create model directory
        model_dir = os.path.join(self.base_dir, identifier)
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights (for subclassed models)
        model_filename = f"model_episode_{episode:03d}.h5"
        model_path = os.path.join(model_dir, model_filename)
        model.save_weights(model_path)

        # Update metadata
        metadata = self._load_metadata(model_dir) or {
            'config': config,
            'created_at': datetime.now().isoformat(),
            'models': []
        }

        # Add model entry
        model_entry = {
            'episode': episode,
            'filename': model_filename,
            'saved_at': datetime.now().isoformat(),
            'metrics': metrics or {}
        }

        # Update or add model entry
        existing_entry_idx = None
        for i, entry in enumerate(metadata['models']):
            if entry['episode'] == episode:
                existing_entry_idx = i
                break

        if existing_entry_idx is not None:
            metadata['models'][existing_entry_idx] = model_entry
        else:
            metadata['models'].append(model_entry)

        # Sort models by episode
        metadata['models'].sort(key=lambda x: x['episode'])

        # Save metadata
        self._save_metadata(model_dir, metadata)

        return model_dir

    def load_model(
        self,
        identifier: str,
        episode: int = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Load model and its metadata.

        Args:
            identifier: Model identifier
            episode: Specific episode to load (None for latest)

        Returns:
            Tuple of (model_path, metadata)
        """
        model_dir = os.path.join(self.base_dir, identifier)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        metadata = self._load_metadata(model_dir)
        if metadata is None or len(metadata['models']) == 0:
            raise ValueError(f"No models found in {model_dir}")

        # Find model to load
        if episode is None:
            # Load latest model
            model_entry = metadata['models'][-1]
        else:
            # Find specific episode
            model_entry = None
            for entry in metadata['models']:
                if entry['episode'] == episode:
                    model_entry = entry
                    break

            if model_entry is None:
                available_episodes = [m['episode'] for m in metadata['models']]
                raise ValueError(
                    f"Episode {episode} not found. Available: {available_episodes}"
                )

        # Load model weights
        # Note: For subclassed models, we need to recreate the model architecture
        # and then load weights. This should be handled by the caller.
        model_path = os.path.join(model_dir, model_entry['filename'])

        # Return the path and metadata for the caller to load properly
        return model_path, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dictionaries
        """
        models = []

        for identifier in os.listdir(self.base_dir):
            model_dir = os.path.join(self.base_dir, identifier)
            if os.path.isdir(model_dir):
                metadata = self._load_metadata(model_dir)
                if metadata:
                    models.append({
                        'identifier': identifier,
                        'created_at': metadata.get('created_at'),
                        'num_episodes': len(metadata['models']),
                        'latest_episode': metadata['models'][-1]['episode']
                        if metadata['models'] else None,
                        'config': metadata.get('config')
                    })

        return models

    def get_best_model(
        self,
        identifier: str,
        metric: str = 'total_return'
    ) -> Tuple[int, Dict[str, float]]:
        """
        Find best model episode based on metric.

        Args:
            identifier: Model identifier
            metric: Metric to use for selection

        Returns:
            Tuple of (best_episode, metrics)
        """
        model_dir = os.path.join(self.base_dir, identifier)
        metadata = self._load_metadata(model_dir)

        if not metadata or not metadata['models']:
            raise ValueError(f"No models found for {identifier}")

        best_episode = None
        best_value = -float('inf')
        best_metrics = None

        for model_entry in metadata['models']:
            if metric in model_entry.get('metrics', {}):
                value = model_entry['metrics'][metric]
                if value > best_value:
                    best_value = value
                    best_episode = model_entry['episode']
                    best_metrics = model_entry['metrics']

        if best_episode is None:
            raise ValueError(f"No models with metric '{metric}' found")

        return best_episode, best_metrics

    def save_replay_buffer(
        self,
        buffer: Any,
        identifier: str,
        episode: int
    ) -> None:
        """
        Save replay buffer to disk.

        Args:
            buffer: Replay buffer object
            identifier: Model identifier
            episode: Episode number
        """
        model_dir = os.path.join(self.base_dir, identifier)
        os.makedirs(model_dir, exist_ok=True)

        buffer_path = os.path.join(model_dir, f"replay_buffer_ep{episode:03d}.pkl")
        with open(buffer_path, 'wb') as f:
            pickle.dump(buffer, f)

    def load_replay_buffer(
        self,
        identifier: str,
        episode: int
    ) -> Any:
        """
        Load replay buffer from disk.

        Args:
            identifier: Model identifier
            episode: Episode number

        Returns:
            Replay buffer object
        """
        model_dir = os.path.join(self.base_dir, identifier)
        buffer_path = os.path.join(model_dir, f"replay_buffer_ep{episode:03d}.pkl")

        if not os.path.exists(buffer_path):
            raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")

        with open(buffer_path, 'rb') as f:
            return pickle.load(f)

    def _generate_identifier(self, config: Dict[str, Any]) -> str:
        """Generate model identifier from configuration."""
        # Use experiment_name as primary identifier to ensure unique per-strategy models
        experiment_name = config.get('experiment_name', 'unnamed')
        ticker = config['ticker']
        start = config['start_date']
        end = config['end_date']
        episodes = config['training']['episodes']

        # Include project_folder if present (for organized directory structure)
        project_folder = config.get('project_folder', None)

        # Use experiment_name to guarantee uniqueness across strategies
        base_identifier = f"{experiment_name}_{ticker}_{start}_{end}_ep{episodes}"

        if project_folder:
            return f"{project_folder}/{base_identifier}"
        return base_identifier

    def _load_metadata(self, model_dir: str) -> Optional[Dict[str, Any]]:
        """Load metadata from model directory."""
        metadata_path = os.path.join(model_dir, 'metadata.json')

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _save_metadata(self, model_dir: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to model directory."""
        import numpy as np

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        metadata_clean = convert_types(metadata)
        metadata_path = os.path.join(model_dir, 'metadata.json')

        with open(metadata_path, 'w') as f:
            json.dump(metadata_clean, f, indent=2, default=str)

    def cleanup_old_models(
        self,
        identifier: str,
        keep_last: int = 5,
        keep_best: bool = True,
        metric: str = 'total_return'
    ) -> None:
        """
        Clean up old model files, keeping only recent and best models.

        Args:
            identifier: Model identifier
            keep_last: Number of recent models to keep
            keep_best: Whether to keep the best performing model
            metric: Metric to use for determining best model
        """
        model_dir = os.path.join(self.base_dir, identifier)
        metadata = self._load_metadata(model_dir)

        if not metadata or len(metadata['models']) <= keep_last:
            return

        models_to_keep = set()

        # Keep the last N models
        for model in metadata['models'][-keep_last:]:
            models_to_keep.add(model['filename'])

        # Keep the best model
        if keep_best:
            try:
                best_episode, _ = self.get_best_model(identifier, metric)
                for model in metadata['models']:
                    if model['episode'] == best_episode:
                        models_to_keep.add(model['filename'])
                        break
            except ValueError:
                pass  # No models with the metric

        # Remove old model files
        for model in metadata['models']:
            if model['filename'] not in models_to_keep:
                model_path = os.path.join(model_dir, model['filename'])
                if os.path.exists(model_path):
                    os.remove(model_path)

        # Update metadata
        metadata['models'] = [
            m for m in metadata['models']
            if m['filename'] in models_to_keep
        ]
        self._save_metadata(model_dir, metadata)