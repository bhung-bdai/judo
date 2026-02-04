# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Plot optimization results from multiple hyperparameter optimization runs.

This script loads all optimization_results.json files from the results/hyperparameter_optimization
folder and creates plots showing:
1. Best score over time (line plot)
2. Current trial scores (scatter plot)

Usage:
    python scripts/plot_optimization_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_optimization_results(results_dir: str = "results/hyperparameter_optimization") -> dict:
    """Load all optimization results from the results directory.

    Args:
        results_dir: Path to the hyperparameter optimization results directory

    Returns:
        Dictionary mapping timestamp folder names to optimization results
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}

    all_results = {}

    # Find all optimization_results.json files
    for json_file in results_path.glob("*/optimization_results.json"):
        timestamp_folder = json_file.parent.name

        try:
            with open(json_file, "r") as f:
                results = json.load(f)
                all_results[timestamp_folder] = results
                print(f"Loaded: {timestamp_folder}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\nTotal optimization runs found: {len(all_results)}")
    return all_results


def plot_optimization_results(all_results: dict, save_dir: str = "results/hyperparameter_optimization"):
    """Plot optimization results for all runs.

    Args:
        all_results: Dictionary mapping timestamps to optimization results
        save_dir: Directory to save the plots
    """
    if not all_results:
        print("No results to plot")
        return

    # Create output directory for plots
    save_path = Path(save_dir) / "summary_plots"
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a figure for each optimization run
    for timestamp, results in all_results.items():
        fig, ax = plt.subplots(figsize=(12, 7))

        # Extract data from best_score_over_time
        best_score_data = results.get("best_score_over_time", [])

        if not best_score_data:
            print(f"No score data found for {timestamp}")
            continue

        # Convert elapsed time to hours
        times_hours = [entry["elapsed_time"] / 3600 for entry in best_score_data]
        best_scores = [entry["best_score"] for entry in best_score_data]
        current_scores = [entry["current_trial_score"] for entry in best_score_data]

        # Plot current trial scores as scatter points
        ax.scatter(times_hours, current_scores, c="lightblue", s=50, alpha=0.6, label="Trial Score", zorder=2)

        # Plot best score over time as a line
        ax.plot(times_hours, best_scores, "r-", linewidth=2.5, label="Best Score So Far", zorder=3)

        # Get metadata
        task_name = results.get("task_name", "Unknown")
        metric = results.get("metric", "score")
        n_trials = results.get("n_trials", len(best_score_data))
        total_time_hours = results.get("total_optimization_time", 0) / 3600
        best_score = results.get("best_score", max(best_scores) if best_scores else 0)

        # Format title with key information
        title = f"Optimization: {task_name}\n"
        title += f"Metric: {metric} | Best: {best_score:.3f} | "
        title += f"Trials: {n_trials} | Time: {total_time_hours:.2f}h"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xlabel("Elapsed Time (hours)", fontsize=12)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=11)

        # Add timestamp to plot
        ax.text(
            0.02,
            0.98,
            f"Run: {timestamp}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        # Save individual plot
        plot_filename = save_path / f"optimization_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {plot_filename}")

        plt.close()

    # Create a combined plot if there are multiple runs
    if len(all_results) > 1:
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        # Collect all data for computing mean
        all_times_data = []
        all_best_scores_data = []
        all_current_scores_data = []

        for (timestamp, results), color in zip(all_results.items(), colors, strict=False):
            best_score_data = results.get("best_score_over_time", [])
            if not best_score_data:
                continue

            times_hours = [entry["elapsed_time"] / 3600 for entry in best_score_data]
            best_scores = [entry["best_score"] for entry in best_score_data]
            current_scores = [entry["current_trial_score"] for entry in best_score_data]

            all_times_data.append(times_hours)
            all_best_scores_data.append(best_scores)
            all_current_scores_data.append(current_scores)

            # Plot individual trial scores with transparency
            ax.scatter(times_hours, current_scores, c=[color], s=30, alpha=0.2, zorder=2)

            # Plot individual best score lines with transparency
            ax.plot(times_hours, best_scores, linewidth=1.5, alpha=0.3, color=color, zorder=3)

        # Compute and plot mean best score over time
        if all_times_data:
            # Find common time grid (use the run with most data points as reference)
            max_len = max(len(times) for times in all_times_data)
            ref_idx = [len(times) for times in all_times_data].index(max_len)
            ref_times = all_times_data[ref_idx]

            # Interpolate all runs to common time grid using zero-order hold
            interpolated_best_scores = []
            for times, scores in zip(all_times_data, all_best_scores_data, strict=False):
                if len(times) > 1:
                    # Zero-order hold: use searchsorted to find indices
                    indices = np.searchsorted(times, ref_times, side="right") - 1
                    indices = np.clip(indices, 0, len(scores) - 1)
                    interp_scores = np.array([scores[i] for i in indices])
                    interpolated_best_scores.append(interp_scores)

            # Compute mean and std
            if interpolated_best_scores:
                mean_best_scores = np.mean(interpolated_best_scores, axis=0)
                std_best_scores = np.std(interpolated_best_scores, axis=0)

                # Plot mean best score as thick line
                ax.plot(
                    ref_times,
                    mean_best_scores,
                    "r-",
                    linewidth=3,
                    label=f"Mean Best Score (n={len(all_results)})",
                    zorder=5,
                )

                # Plot standard deviation as shaded region
                ax.fill_between(
                    ref_times,
                    mean_best_scores - std_best_scores,
                    mean_best_scores + std_best_scores,
                    color="red",
                    alpha=0.2,
                    zorder=4,
                    label="±1 Std Dev",
                )

        ax.set_title("All Optimization Runs - Mean Best Score Over Time", fontsize=16, fontweight="bold")
        ax.set_xlabel("Elapsed Time (hours)", fontsize=13)
        ax.set_ylabel("Score", fontsize=13)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=11)

        plt.tight_layout()

        # Save combined plot
        combined_filename = save_path / "all_optimizations_combined.png"
        plt.savefig(combined_filename, dpi=150, bbox_inches="tight")
        print(f"\nSaved combined plot: {combined_filename}")

        plt.close()

    print(f"\nAll plots saved to: {save_path}")


def print_summary_statistics(all_results: dict):
    """Print summary statistics for all optimization runs.

    Args:
        all_results: Dictionary mapping timestamps to optimization results
    """
    if not all_results:
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for timestamp, results in all_results.items():
        print(f"\nRun: {timestamp}")
        print(f"  Task: {results.get('task_name', 'Unknown')}")
        print(f"  Metric: {results.get('metric', 'Unknown')}")
        print(f"  Best Score: {results.get('best_score', 0):.4f}")
        print(f"  Total Trials: {results.get('n_trials', 0)}")
        print(f"  Total Time: {results.get('total_optimization_time', 0) / 3600:.2f} hours")

        # Print best parameters
        best_params = results.get("best_params", {})
        if best_params:
            print("  Best Parameters:")
            for param, value in best_params.items():
                print(f"    {param}: {value:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    # Load all optimization results
    all_results = load_all_optimization_results()

    if all_results:
        # Print summary statistics
        print_summary_statistics(all_results)

        # Create plots
        plot_optimization_results(all_results)

        print("\nDone!")
    else:
        print("No optimization results found.")
