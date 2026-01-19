"""Standalone script to monitor ongoing training runs."""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.progress_logger import ProgressLogger


def monitor_run(run_name=None, refresh_interval=5):
    """
    Monitor a training run in real-time.

    Args:
        run_name: Name of the run to monitor (latest if None)
        refresh_interval: Seconds between refreshes
    """
    # Find run to monitor
    log_dir = "logs"

    if run_name is None:
        # Get latest run
        if not os.path.exists(log_dir):
            print("No logs directory found. No runs to monitor.")
            return

        runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not runs:
            print("No runs found in logs directory.")
            return

        run_name = sorted(runs)[-1]
        print(f"Monitoring latest run: {run_name}")

    # Create logger instance
    logger = ProgressLogger(run_name=run_name, log_dir="logs")

    print(f"\n{'='*60}")
    print(f"MONITORING: {run_name}")
    print(f"{'='*60}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")

    last_episode = 0

    try:
        while True:
            # Clear screen (works on Unix/Linux/Mac)
            os.system('clear' if os.name == 'posix' else 'cls')

            # Load current progress
            progress = logger.get_current_progress()

            # Display header
            print(f"\n{'='*60}")
            print(f"🔄 LIVE TRAINING MONITOR - {run_name}")
            print(f"{'='*60}")

            # Display runtime
            if 'start_time' in progress:
                start_time = datetime.fromisoformat(progress['start_time'])
                runtime = (datetime.now() - start_time).total_seconds()
                print(f"⏱️  Runtime: {runtime/60:.1f} minutes")

            # Display current status
            status = progress.get('status', 'unknown')
            status_icon = {
                'running': '🟢',
                'completed': '✅',
                'failed': '❌',
                'unknown': '❓'
            }.get(status, '❓')
            print(f"{status_icon} Status: {status}")

            # Display current stage
            if 'current_stage' in progress:
                print(f"📍 Current Stage: {progress['current_stage']}")

            # Display stages progress
            print(f"\n📊 Pipeline Progress:")
            stages = progress.get('stages', {})
            for stage_name, stage_info in stages.items():
                status = stage_info.get('status', 'pending')
                status_icon = '✅' if status == 'completed' else '🔄' if status == 'running' else '⏳'

                duration_str = ""
                if 'duration_seconds' in stage_info:
                    duration = stage_info['duration_seconds']
                    duration_str = f" ({duration:.1f}s)"

                metrics_str = ""
                if 'metrics' in stage_info and isinstance(stage_info['metrics'], str):
                    metrics_str = f" - {stage_info['metrics']}"

                print(f"  {status_icon} {stage_name}{duration_str}{metrics_str}")

            # Display training metrics if available
            if 'latest_episode' in progress and progress['latest_episode'] > 0:
                print(f"\n📈 Training Metrics (Episode {progress['latest_episode']}):")

                metrics = progress.get('latest_metrics', {})
                if metrics:
                    print(f"  • Return: {metrics.get('return', 0):.2%}")
                    print(f"  • Profit: ${metrics.get('profit', 0):.2f}")
                    print(f"  • Trades: {metrics.get('trades', 0)}")
                    print(f"  • Win Rate: {metrics.get('win_rate', 0):.2%}")
                    print(f"  • Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"  • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

                # Show episode progress
                current_ep = progress['latest_episode']
                if current_ep > last_episode:
                    print(f"  📊 New episodes: {current_ep - last_episode}")
                    last_episode = current_ep

            # Display validation results if available
            if 'validation' in progress and progress['validation']:
                print(f"\n🔍 Validation Results:")
                for val in progress['validation'][-3:]:  # Show last 3
                    period = val.get('period', 0)
                    metrics = val.get('metrics', {})
                    print(f"  Period {period}: Return={metrics.get('total_return', 0):.2%}")

            # Display test results if available
            if 'test_results' in progress:
                test_metrics = progress['test_results'].get('metrics', {})
                print(f"\n🎯 Test Results:")
                print(f"  • Total Return: {test_metrics.get('total_return', 0):.2%}")
                print(f"  • Sharpe Ratio: {test_metrics.get('sharpe_ratio', 0):.2f}")

                if progress['test_results'].get('comparison'):
                    comparison = progress['test_results']['comparison']
                    print(f"\n📊 Strategy Comparison:")
                    print(f"  • DQN Return: {comparison.get('dqn_return', 0):.2%}")
                    print(f"  • Buy & Hold: {comparison.get('buy_hold_return', 0):.2%}")
                    print(f"  • Outperformance: {comparison.get('outperformance', 0):.2%}")

            # Display error if any
            if 'error' in progress:
                error = progress['error']
                print(f"\n❌ ERROR in {error.get('stage', 'unknown')}:")
                print(f"  {error.get('message', 'Unknown error')}")

            # Show last update time
            print(f"\n🔄 Last refresh: {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Next refresh in {refresh_interval} seconds...")

            # Check if run is complete
            if status == 'completed':
                print(f"\n{'='*60}")
                print("✅ Training Complete!")
                print(f"{'='*60}")

                # Show summary
                logger.print_summary()
                break

            elif status == 'failed':
                print(f"\n{'='*60}")
                print("❌ Training Failed!")
                print(f"{'='*60}")
                break

            # Wait for next refresh
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped. Run is still in progress.")
        print(f"To resume monitoring, run: python monitor_training.py --run {run_name}")


def list_runs():
    """List all available runs."""
    log_dir = "logs"

    if not os.path.exists(log_dir):
        print("No logs directory found.")
        return

    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    if not runs:
        print("No runs found.")
        return

    print(f"\n{'='*60}")
    print("Available Training Runs:")
    print(f"{'='*60}")

    for run_name in sorted(runs):
        progress_file = os.path.join(log_dir, run_name, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            status = progress.get('status', 'unknown')
            episodes = progress.get('latest_episode', 0)

            status_icon = {
                'running': '🟢',
                'completed': '✅',
                'failed': '❌',
                'unknown': '❓'
            }.get(status, '❓')

            print(f"{status_icon} {run_name} - Status: {status}, Episodes: {episodes}")

    print(f"{'='*60}")
    print("\nTo monitor a run: python monitor_training.py --run <run_name>")
    print("To monitor latest: python monitor_training.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor DQN trading training runs")
    parser.add_argument('--run', type=str, help='Name of run to monitor (latest if not specified)')
    parser.add_argument('--list', action='store_true', help='List all available runs')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh interval in seconds')

    args = parser.parse_args()

    if args.list:
        list_runs()
    else:
        monitor_run(args.run, args.refresh)