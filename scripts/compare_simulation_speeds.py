# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Compare simulation speeds between Spot tasks with and without low-level policy.

This script times the action update step for:
1. Regular spot_box_push (without low-level policy)
2. spot_box_push_baseline (with low-level locomotion policy)
"""

import time
from typing import Any

import numpy as np
from mujoco import mj_step

from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks
from judo.utils.mujoco import SimBackendSpot


def load_controller_for_task(
    task_name: str,
    optimizer_name: str = "mppi",
) -> tuple[Any, Any, Controller, Any, Any, ControllerConfig]:
    """Load a controller for the specified task.

    Args:
        task_name: Name of the task to load
        optimizer_name: Name of the optimizer to use

    Returns:
        Tuple of (task, optimizer, controller, task_config, optimizer_config, controller_config)
    """
    # Load task
    task_dict = get_registered_tasks()
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' is not registered.")
    task_cls, task_config_cls = task_dict[task_name]
    task_config = task_config_cls()
    task = task_cls()

    # Load optimizer
    optimizer_dict = get_registered_optimizers()
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' is not registered.")
    optimizer_cls, optimizer_config_cls = optimizer_dict[optimizer_name]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override(task_name)
    optimizer = optimizer_cls(optimizer_config, task.nu)

    # Load controller
    controller_config = ControllerConfig()
    controller_config.set_override(task_name)
    controller = Controller(
        controller_config,
        task,
        task_config,
        optimizer,
        optimizer_config,
    )

    return task, optimizer, controller, task_config, optimizer_config, controller_config


def time_update_action(
    task_name: str,
    num_updates: int = 100,
    optimizer_name: str = "mppi",
) -> dict[str, float]:
    """Time the update_action calls for a given task.

    Args:
        task_name: Name of the task to benchmark
        num_updates: Number of update_action calls to time
        optimizer_name: Name of the optimizer to use

    Returns:
        Dictionary with timing statistics
    """
    print(f"\nTiming {task_name}...")

    # Load controller
    task, optimizer, controller, task_config, optimizer_config, controller_config = load_controller_for_task(
        task_name, optimizer_name
    )

    sim_model = task.sim_model
    ctrl_model = task.model

    # Initialize custom backend for Spot tasks
    custom_sim_backend = None
    use_custom_backend = "spot" in task_name and "baseline" not in task_name

    if use_custom_backend:
        custom_sim_backend = SimBackendSpot(task_to_sim_ctrl=task.task_to_sim_ctrl)

    # Reset task
    task.reset()
    controller.reset()

    # Get initial state
    curr_state = np.concatenate([task.data.qpos, task.data.qvel])
    curr_time = 0.0

    # Simulation timestep
    sim_dt = sim_model.opt.timestep
    plan_dt = 1.0 / controller_config.control_freq

    # Warm-up: run a few iterations to avoid cold start effects
    print(f"  Running {5} warm-up iterations...")
    for i in range(5):
        controller.update_action(curr_state, curr_time)
        current_action = controller.action(curr_time)

        # Advance simulation
        task.pre_sim_step()
        if use_custom_backend and custom_sim_backend is not None:
            custom_sim_backend.sim(sim_model, task.data, current_action)
        else:
            task.data.ctrl[:] = current_action
            mj_step(sim_model, task.data)
        task.post_sim_step()

        # Update state
        curr_state = np.concatenate([task.data.qpos, task.data.qvel])
        curr_time += plan_dt
        controller.system_metadata = task.get_sim_metadata()

    # Timing: measure update_action calls
    print(f"  Timing {num_updates} update_action calls...")
    update_times = []

    for i in range(num_updates):
        start_time = time.perf_counter()
        controller.update_action(curr_state, curr_time)
        end_time = time.perf_counter()

        update_times.append(end_time - start_time)

        # Get action and advance simulation
        current_action = controller.action(curr_time)

        # Advance simulation
        task.pre_sim_step()
        if use_custom_backend and custom_sim_backend is not None:
            custom_sim_backend.sim(sim_model, task.data, current_action)
        else:
            task.data.ctrl[:] = current_action
            mj_step(sim_model, task.data)
        task.post_sim_step()

        # Update state
        curr_state = np.concatenate([task.data.qpos, task.data.qvel])
        curr_time += plan_dt
        controller.system_metadata = task.get_sim_metadata()

    # Compute statistics
    update_times = np.array(update_times)
    stats = {
        "mean": np.mean(update_times),
        "std": np.std(update_times),
        "min": np.min(update_times),
        "max": np.max(update_times),
        "median": np.median(update_times),
        "total": np.sum(update_times),
    }

    return stats


def print_comparison(stats_regular: dict[str, float], stats_baseline: dict[str, float]) -> None:
    """Print comparison of timing statistics.

    Args:
        stats_regular: Timing statistics for regular task
        stats_baseline: Timing statistics for baseline task
    """
    print("\n" + "=" * 80)
    print("TIMING COMPARISON")
    print("=" * 80)

    print("\nRegular Spot Box Push (with low-level policy):")
    print(f"  Mean time:   {stats_regular['mean'] * 1000:.2f} ms")
    print(f"  Std time:    {stats_regular['std'] * 1000:.2f} ms")
    print(f"  Median time: {stats_regular['median'] * 1000:.2f} ms")
    print(f"  Min time:    {stats_regular['min'] * 1000:.2f} ms")
    print(f"  Max time:    {stats_regular['max'] * 1000:.2f} ms")
    print(f"  Total time:  {stats_regular['total']:.2f} s")

    print("\nSpot Box Push Baseline (without low-level locomotion policy):")
    print(f"  Mean time:   {stats_baseline['mean'] * 1000:.2f} ms")
    print(f"  Std time:    {stats_baseline['std'] * 1000:.2f} ms")
    print(f"  Median time: {stats_baseline['median'] * 1000:.2f} ms")
    print(f"  Min time:    {stats_baseline['min'] * 1000:.2f} ms")
    print(f"  Max time:    {stats_baseline['max'] * 1000:.2f} ms")
    print(f"  Total time:  {stats_baseline['total']:.2f} s")

    print("\nSpeedup Analysis:")
    speedup = stats_regular["mean"] / stats_baseline["mean"]
    if speedup > 1.0:
        print(f"  Regular task is {speedup:.2f}x SLOWER than baseline")
        print(f"  Baseline task is {1 / speedup:.2f}x FASTER than regular")
    else:
        print(f"  Regular task is {1 / speedup:.2f}x FASTER than baseline")
        print(f"  Baseline task is {speedup:.2f}x SLOWER than regular")

    print(
        f"\n  Absolute time difference: {abs(stats_regular['mean'] - stats_baseline['mean']) * 1000:.2f} ms per update"
    )
    print("=" * 80)


def main() -> None:
    """Main function to compare simulation speeds."""
    print("Comparing simulation speeds: Spot Box Push with/without low-level policy")
    print("=" * 80)

    # Configuration
    num_updates = 100
    optimizer_name = "cem"

    print("\nConfiguration:")
    print(f"  Number of updates to time: {num_updates}")
    print(f"  Optimizer: {optimizer_name}")

    # Time regular spot_box_push (without low-level policy)
    stats_regular = time_update_action(
        task_name="spot_box_push",
        num_updates=num_updates,
        optimizer_name=optimizer_name,
    )

    # Time spot_box_push_baseline (with low-level policy)
    stats_baseline = time_update_action(
        task_name="spot_box_push_baseline",
        num_updates=num_updates,
        optimizer_name=optimizer_name,
    )

    # Print comparison
    print_comparison(stats_regular, stats_baseline)


if __name__ == "__main__":
    main()
