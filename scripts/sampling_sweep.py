# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Hyperparameter optimization for spot_box_push task using Bayesian Optimization.

This script provides utilities for optimizing task configuration parameters
(e.g., reward weights) using Bayesian Optimization with Optuna.

Example usage:
    python scripts/sampling_sweep.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from performance_benchmark import benchmark_single_task_and_optimizer


def objective_function(
    params: dict[str, float],
    task_name: str = "spot_box_push",
    optimizer_name: str = "cem",
    num_episodes: int = 10,
    episode_length_s: float = 30.0,
    metric: str = "success_rate",
    verbose: bool = False,
) -> float:
    """Objective function for Bayesian optimization.

    Args:
        params: Dictionary of task config parameters to evaluate
        task_name: Name of the task to benchmark
        optimizer_name: Name of the optimizer to use
        num_episodes: Number of episodes to run
        episode_length_s: Maximum length of each episode (in seconds)
        metric: Metric to optimize. Options:
            - "success_rate": Fraction of successful episodes (default)
            - "avg_reward": Average reward across all episodes
            - "avg_episode_length": Average episode length (shorter is better for this metric)
        verbose: Whether to print detailed information

    Returns:
        Scalar value to minimize (negative for metrics we want to maximize)
    """
    print(f"\nEvaluating params: {params}")

    # Run benchmark
    results = benchmark_single_task_and_optimizer(
        task_name=task_name,
        optimizer_name=optimizer_name,
        num_episodes=num_episodes,
        episode_length_s=episode_length_s,
        task_config_overrides=params,
        verbose=verbose,
    )

    # Compute the metric
    if metric == "success_rate":
        success_count = sum(ep["success"] for ep in results)
        score = success_count / len(results)
        # Return negative for minimization (we want to maximize success rate)
        objective_value = -score
        print(f"  Success rate: {score:.3f} ({success_count}/{len(results)})")

    elif metric == "avg_reward":
        all_rewards = [np.mean(ep["rewards"]) for ep in results]
        score = np.mean(all_rewards)
        # Return negative for minimization (we want to maximize reward)
        objective_value = -score
        print(f"  Average reward: {score:.3f}")

    elif metric == "avg_episode_length":
        avg_length = np.mean([ep["length"] for ep in results])
        # Shorter episodes might be better (faster success)
        objective_value = avg_length
        print(f"  Average episode length: {avg_length:.3f}s")

    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"  Objective value: {objective_value:.3f}")

    return objective_value


def run_bayesian_optimization(
    param_space: dict[str, tuple[float, float]],
    task_name: str = "spot_box_push",
    optimizer_name: str = "cem",
    num_episodes: int = 10,
    episode_length_s: float = 30.0,
    n_trials: int = 50,
    n_startup_trials: int = 10,
    metric: str = "success_rate",
    random_state: int = 42,
    save_results: bool = True,
    verbose: bool = False,
) -> dict:
    """Run Bayesian optimization to find optimal task config parameters using Optuna.

    Args:
        param_space: Dictionary mapping parameter names to (min, max) tuples
            Example: {"w_orientation": (10.0, 200.0), "w_gripper_proximity": (0.1, 10.0)}
        task_name: Name of the task to optimize
        optimizer_name: Name of the optimizer to use
        num_episodes: Number of episodes per evaluation
        episode_length_s: Maximum length of each episode (in seconds)
        n_trials: Total number of optimization trials
        n_startup_trials: Number of random initialization trials before using TPE sampler
        metric: Metric to optimize ("success_rate", "avg_reward", or "avg_episode_length")
        random_state: Random seed for reproducibility
        save_results: Whether to save results to disk
        verbose: Whether to print detailed information

    Returns:
        Dictionary containing optimization results
    """
    print("=" * 80)
    print(f"Starting Bayesian Optimization for {task_name}")
    print("Parameter space:")
    for param_name, (min_val, max_val) in param_space.items():
        print(f"  {param_name}: [{min_val}, {max_val}]")
    print(f"Metric to optimize: {metric}")
    print(f"Number of trials: {n_trials}")
    print(f"Startup trials (random): {n_startup_trials}")
    print("=" * 80)

    # Create Optuna study
    # For success_rate and avg_reward, we want to maximize (direction="maximize")
    # For avg_episode_length, we want to minimize (direction="minimize")
    direction = "minimize" if metric == "avg_episode_length" else "maximize"

    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=f"{task_name}_{metric}_optimization",
    )

    # Track time for each trial
    trial_times = []
    start_time = time.time()

    # Define the objective function for Optuna
    def optuna_objective(trial: optuna.Trial) -> float:
        trial_start_time = time.time()

        # Sample parameters from the search space
        params = {}
        for param_name, (min_val, max_val) in param_space.items():
            params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        print(f"\n[Trial {trial.number + 1}/{n_trials}] Evaluating params: {params}")

        # Run benchmark
        results = benchmark_single_task_and_optimizer(
            task_name=task_name,
            optimizer_name=optimizer_name,
            num_episodes=num_episodes,
            episode_length_s=episode_length_s,
            task_config_overrides=params,
            verbose=verbose,
        )

        # Compute the metric
        if metric == "success_rate":
            success_count = sum(ep["success"] for ep in results)
            score = success_count / len(results)
            print(f"  Success rate: {score:.3f} ({success_count}/{len(results)})")
        elif metric == "avg_reward":
            all_rewards = [np.mean(ep["rewards"]) for ep in results]
            score = float(np.mean(all_rewards))
            print(f"  Average reward: {score:.3f}")
        elif metric == "avg_episode_length":
            score = float(np.mean([ep["length"] for ep in results]))
            print(f"  Average episode length: {score:.3f}s")
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Record time for this trial
        trial_end_time = time.time()
        elapsed_from_start = trial_end_time - start_time
        trial_duration = trial_end_time - trial_start_time
        trial_times.append(
            {
                "trial_number": trial.number,
                "elapsed_time": elapsed_from_start,
                "trial_duration": trial_duration,
            }
        )

        print(f"  Trial took {trial_duration:.2f}s (total elapsed: {elapsed_from_start:.2f}s)")

        return score

    # Run optimization
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)

    total_time = time.time() - start_time

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print(f"Best {metric}: {best_score:.3f}")
    print(f"Total optimization time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print("Best parameters:")
    for param_name, value in best_params.items():
        print(f"  {param_name}: {value:.4f}")
    print("=" * 80)

    # Compute best score over time
    best_score_over_time = []
    current_best = None

    for i, trial in enumerate(study.trials):
        trial_value = trial.value
        if trial_value is None:
            continue

        # Update current best
        if current_best is None:
            current_best = trial_value
        elif direction == "maximize":
            current_best = max(current_best, trial_value)
        else:
            current_best = min(current_best, trial_value)

        # Get elapsed time for this trial
        elapsed_time = trial_times[i]["elapsed_time"] if i < len(trial_times) else 0

        best_score_over_time.append(
            {
                "trial_number": trial.number,
                "elapsed_time": elapsed_time,
                "best_score": float(current_best),
                "current_trial_score": float(trial_value),
            }
        )

    # Prepare results dictionary
    optimization_results = {
        "task_name": task_name,
        "optimizer_name": optimizer_name,
        "metric": metric,
        "param_space": param_space,
        "n_trials": n_trials,
        "n_startup_trials": n_startup_trials,
        "num_episodes": num_episodes,
        "episode_length_s": episode_length_s,
        "best_params": best_params,
        "best_score": float(best_score),
        "total_optimization_time": float(total_time),
        "best_score_over_time": best_score_over_time,
        "all_trials": [
            {
                "params": trial.params,
                "value": trial.value,
                "number": trial.number,
                "elapsed_time": trial_times[i]["elapsed_time"] if i < len(trial_times) else None,
                "trial_duration": trial_times[i]["trial_duration"] if i < len(trial_times) else None,
            }
            for i, trial in enumerate(study.trials)
        ],
    }

    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / "hyperparameter_optimization" / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = results_dir / "optimization_results.json"
        with open(json_path, "w") as f:
            json.dump(optimization_results, f, indent=2)

        print(f"\nResults saved to: {json_path}")

        # Save Optuna visualizations if plotly is available
        try:
            import optuna.visualization as vis

            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(str(results_dir / "optimization_history.html"))

            # Parameter importances
            fig = vis.plot_param_importances(study)
            fig.write_html(str(results_dir / "param_importances.html"))

            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(str(results_dir / "parallel_coordinate.html"))

            # Slice plot (parameter relationships)
            fig = vis.plot_slice(study)
            fig.write_html(str(results_dir / "slice_plot.html"))

            print(f"Visualizations saved to: {results_dir}")

        except ImportError:
            print("plotly not available, skipping visualizations")

    return optimization_results


def spot_box_push_optimization():
    """Example: Optimize reward weights for spot_box_push task."""
    # Define parameter space for spot_box_push weights
    param_space = {
        # "w_orientation": (10.0, 200.0),
        "w_goal": (0.0, 500.0),
        "w_torso_proximity": (0.1, 10.0),
        "w_gripper_proximity": (0.1, 10.0),
        "w_object_velocity": (0.1, 10.0),
    }

    seeds = [42, 123, 456, 789, 101]
    for seed in seeds:
        results = run_bayesian_optimization(
            param_space=param_space,
            task_name="spot_box_push",
            optimizer_name="cem",
            num_episodes=10,  # Use fewer episodes for faster iteration during optimization
            episode_length_s=30.0,
            n_trials=50,  # Total number of evaluations
            n_startup_trials=5,  # Random initialization trials
            metric="success_rate",
            random_state=seed,
            save_results=True,
            verbose=False,
        )


def quick_test():
    """Quick test with minimal evaluations to verify everything works."""
    param_space = {
        "w_orientation": (50.0, 150.0),
        "w_gripper_proximity": (2.0, 6.0),
    }

    results = run_bayesian_optimization(
        param_space=param_space,
        task_name="spot_box_push",
        optimizer_name="cem",
        num_episodes=3,
        episode_length_s=10.0,
        n_trials=5,
        n_startup_trials=2,
        metric="success_rate",
        random_state=42,
        save_results=True,
        verbose=False,
    )

    return results


if __name__ == "__main__":
    # Run the full optimization
    spot_box_push_optimization()

    # Or run a quick test
    # quick_test()
