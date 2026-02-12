# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Import mujoco first to ensure libmujoco.so is loaded into the process
# before the pybind module tries to resolve it as a dynamic dependency.
import mujoco as _mujoco  # noqa: F401

from mujoco_extensions import policy_rollout

__all__ = [
    "policy_rollout",
]
