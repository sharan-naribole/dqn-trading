"""Progress logging system for monitoring training runs."""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ProgressLogger:
    """Log training progress to files for monitoring."""

    def __init__(self, run_name: Optional[str] = None, log_dir: str = "logs"):
        """
        Initialize progress logger.

        Args:
            run_name: Name for this run (uses timestamp if not provided)
            log_dir: Directory to store log files
        """
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.run_name = run_name
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create different log files
        self.progress_file = os.path.join(self.log_dir, 'progress.json')
        self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
        self.status_file = os.path.join(self.log_dir, 'status.txt')
        self.summary_file = os.path.join(self.log_dir, 'summary.json')

        # Initialize progress tracking
        self.start_time = datetime.now()
        self.current_stage = None
        self.stages_completed = []
        self.metrics_history = []

        # Write initial status
        self._update_status("Initialized", f"Run: {run_name}")

    def start_stage(self, stage_name: str, details: str = ""):
        """
        Log the start of a new stage.

        Args:
            stage_name: Name of the stage
            details: Additional details
        """
        self.current_stage = stage_name
        timestamp = datetime.now()

        # Update progress file
        progress = self._load_progress()
        progress['current_stage'] = stage_name
        progress['stage_start_time'] = timestamp.isoformat()
        progress['stages'][stage_name] = {
            'status': 'running',
            'start_time': timestamp.isoformat(),
            'details': details
        }
        self._save_progress(progress)

        # Update status file
        self._update_status(f"Starting: {stage_name}", details)

    def complete_stage(self, stage_name: str, metrics: Optional[Dict] = None):
        """
        Log the completion of a stage.

        Args:
            stage_name: Name of the stage
            metrics: Stage metrics/results
        """
        timestamp = datetime.now()
        self.stages_completed.append(stage_name)

        # Update progress file
        progress = self._load_progress()
        if stage_name in progress['stages']:
            progress['stages'][stage_name]['status'] = 'completed'
            progress['stages'][stage_name]['end_time'] = timestamp.isoformat()
            if metrics:
                progress['stages'][stage_name]['metrics'] = metrics

            # Calculate duration
            start_time = datetime.fromisoformat(progress['stages'][stage_name]['start_time'])
            duration = (timestamp - start_time).total_seconds()
            progress['stages'][stage_name]['duration_seconds'] = duration

        progress['stages_completed'] = self.stages_completed
        self._save_progress(progress)

        # Update status file
        self._update_status(f"Completed: {stage_name}",
                           f"Duration: {duration:.1f}s" if 'duration' in locals() else "")

    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """
        Log metrics for a training episode.

        Args:
            episode: Episode number
            metrics: Episode metrics
        """
        # Add timestamp and episode to metrics
        metrics_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            **metrics
        }

        # Append to history
        self.metrics_history.append(metrics_with_meta)

        # Save to CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.metrics_file, index=False)

        # Update progress file with latest metrics
        progress = self._load_progress()
        progress['latest_episode'] = episode
        progress['latest_metrics'] = metrics
        progress['total_episodes_completed'] = len(self.metrics_history)
        self._save_progress(progress)

        # Update status with key metrics
        status_msg = f"Episode {episode}: Return={metrics.get('return', 0):.2%}, "
        status_msg += f"Trades={metrics.get('trades', 0)}, "
        status_msg += f"Win Rate={metrics.get('win_rate', 0):.2%}"
        self._update_status("Training Progress", status_msg)

    def log_validation(self, period: int, metrics: Dict[str, float]):
        """
        Log validation results.

        Args:
            period: Validation period number
            metrics: Validation metrics
        """
        progress = self._load_progress()
        if 'validation' not in progress:
            progress['validation'] = []

        progress['validation'].append({
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })

        self._save_progress(progress)

        # Update status
        self._update_status(f"Validation Period {period}",
                           f"Return: {metrics.get('total_return', 0):.2%}")

    def log_test_results(self, metrics: Dict[str, float], comparison: Optional[Dict] = None):
        """
        Log final test results.

        Args:
            metrics: Test metrics
            comparison: Strategy comparison (e.g., vs buy & hold)
        """
        progress = self._load_progress()
        progress['test_results'] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'comparison': comparison
        }

        # Calculate total runtime
        total_runtime = (datetime.now() - self.start_time).total_seconds()
        progress['total_runtime_seconds'] = total_runtime
        progress['status'] = 'completed'

        self._save_progress(progress)

        # Create final summary
        self._create_summary(progress)

        # Update status
        status_msg = f"Test Complete: Return={metrics.get('total_return', 0):.2%}, "
        status_msg += f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}"
        self._update_status("Run Complete", status_msg)

    def log_error(self, stage: str, error: str):
        """
        Log an error.

        Args:
            stage: Stage where error occurred
            error: Error message
        """
        progress = self._load_progress()
        progress['status'] = 'failed'
        progress['error'] = {
            'stage': stage,
            'message': error,
            'timestamp': datetime.now().isoformat()
        }
        self._save_progress(progress)

        self._update_status(f"Error in {stage}", error)

    def get_current_progress(self) -> Dict:
        """Get current progress state."""
        return self._load_progress()

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        if os.path.exists(self.metrics_file):
            return pd.read_csv(self.metrics_file)
        return pd.DataFrame()

    def _load_progress(self) -> Dict:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)

        # Initialize if doesn't exist
        return {
            'run_name': self.run_name,
            'start_time': self.start_time.isoformat(),
            'status': 'running',
            'stages': {},
            'stages_completed': [],
            'latest_episode': 0,
            'latest_metrics': {}
        }

    def _save_progress(self, progress: Dict):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

    def _update_status(self, status: str, details: str = ""):
        """Update status file with latest information."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status_line = f"[{timestamp}] {status}"
        if details:
            status_line += f" - {details}"

        # Append to status file
        with open(self.status_file, 'a') as f:
            f.write(status_line + "\n")

        # Also print to console for immediate feedback
        print(status_line)

    def _create_summary(self, progress: Dict):
        """Create final summary file."""
        summary = {
            'run_name': self.run_name,
            'start_time': progress['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_runtime_seconds': progress.get('total_runtime_seconds', 0),
            'status': progress.get('status', 'unknown'),
            'stages_completed': progress.get('stages_completed', []),
            'episodes_trained': progress.get('total_episodes_completed', 0),
            'final_test_return': None,
            'final_sharpe_ratio': None,
            'vs_buy_hold': None
        }

        # Extract key metrics if available
        if 'test_results' in progress:
            test_metrics = progress['test_results'].get('metrics', {})
            summary['final_test_return'] = test_metrics.get('total_return', 0)
            summary['final_sharpe_ratio'] = test_metrics.get('sharpe_ratio', 0)

            if progress['test_results'].get('comparison'):
                comparison = progress['test_results']['comparison']
                summary['vs_buy_hold'] = comparison.get('outperformance', 0)

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def print_summary(self):
        """Print a summary of the current run."""
        progress = self._load_progress()

        print("\n" + "="*60)
        print(f"RUN SUMMARY: {self.run_name}")
        print("="*60)

        # Runtime
        if 'total_runtime_seconds' in progress:
            runtime = progress['total_runtime_seconds']
            print(f"Total Runtime: {runtime/60:.1f} minutes")

        # Stages
        print(f"\nStages Completed: {len(progress.get('stages_completed', []))}")
        for stage in progress.get('stages_completed', []):
            stage_info = progress['stages'].get(stage, {})
            duration = stage_info.get('duration_seconds', 0)
            print(f"  • {stage}: {duration:.1f}s")

        # Training metrics
        if 'latest_metrics' in progress:
            print(f"\nLatest Training Metrics:")
            for key, value in progress['latest_metrics'].items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.4f}")

        # Test results
        if 'test_results' in progress:
            test_metrics = progress['test_results'].get('metrics', {})
            print(f"\nTest Results:")
            print(f"  • Total Return: {test_metrics.get('total_return', 0):.2%}")
            print(f"  • Sharpe Ratio: {test_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  • Max Drawdown: {test_metrics.get('max_drawdown', 0):.2%}")
            print(f"  • Win Rate: {test_metrics.get('win_rate', 0):.2%}")

        print("="*60 + "\n")