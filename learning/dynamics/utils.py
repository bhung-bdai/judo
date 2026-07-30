# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Utility functions for the dynamics learning."""
from typing import Generator, Any

import torch
from pathlib import Path


def flatten_last_two_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Flattens the last two dimensions of a tensor."""
    return tensor.reshape(*tensor.shape[:-2], -1)


def unflatten_last_two_dims(tensor: torch.Tensor, dim_1: int, dim_2: int) -> torch.Tensor:
    """Unflattens the last two dimensions of a tensor."""
    return tensor.reshape(*tensor.shape[:-1], dim_1, dim_2)


def expand_action_horizon(
    action_horizon: torch.Tensor,
    prediction_horizon_steps: int,
) -> torch.Tensor:
    """Repeat decimated actions so the horizon matches prediction steps.

    Example: 25 action steps -> 50 prediction steps with repeat factor 2.
    """
    action_horizon_steps = action_horizon.shape[1]
    if action_horizon_steps == prediction_horizon_steps:
        return action_horizon
    if prediction_horizon_steps % action_horizon_steps != 0:
        raise ValueError(
            f"prediction_horizon_steps ({prediction_horizon_steps}) must be a multiple of "
            f"action_horizon_steps ({action_horizon_steps})"
        )
    return action_horizon.repeat_interleave(
        prediction_horizon_steps // action_horizon_steps,
        dim=1,
    )


def loop_dataloader(dataloader: torch.utils.data.DataLoader) -> Generator[Any, None, None]:
    """Create an infinite loop generator over the dataloader.

    Args:
        dataloader: DataLoader to loop over

    Yields:
        Batches from the dataloader indefinitely
    """
    while True:
        for batch in dataloader:
            yield batch


def get_judo_path() -> Path:
    """Get the path to the judo directory."""
    return Path(__file__).parent.parent.parent