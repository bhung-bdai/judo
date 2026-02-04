# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_box.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
DEFAULT_BOX_HEIGHT = 0.254
# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
USE_LEGS = False


@dataclass
class SpotBoxUprightConfig(SpotBaseConfig):
    """Config for the spot box uprighting task."""

    w_goal: float = 0.0
    w_object_orientation: float = 100.0
    orientation_error_smoothing_width: float = 1.0


class SpotBoxUpright(SpotBase):
    """Task getting Spot to upright a box."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["box_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["box_joint"])
        self.object_z_axis_idx = get_sensor_indices(self.model, "object_z_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBoxUprightConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot box uprighting task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        object_linear_velocity = qvel[..., self.object_vel_idx[0:3]]

        object_z_axis = sensors[..., self.object_z_axis_idx]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (body_height <= config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array([0.0, 0.0, DEFAULT_BOX_HEIGHT])[None, None], axis=-1
        ).mean(-1)

        # Compute orientation reward (box y-axis should be vertical, so z-component should be 1 or -1)
        orientation_error = np.abs(object_z_axis[..., 2] - 1.0)  # 0 to 1
        orientation_error_smooth = np.exp(orientation_error / config.orientation_error_smoothing_width)  # 1 to e
        object_orientation_reward = -config.w_object_orientation * orientation_error_smooth.mean(axis=-1)

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object,
            axis=-1,
        ).mean(-1)

        # Compute squared l2 norm of the object velocity.
        object_linear_velocity_penalty = -config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_penalty.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_penalty
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        # Sample object position in annulus
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

        # Object starts on its side (45 degree roll)
        object_pose = np.array([*object_pos, DEFAULT_BOX_HEIGHT, np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])

        # Place robot at random x and y
        robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(
        self, model: MjModel, data: MjData, config: SpotBoxUprightConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if the box is upright (y-axis vertical)."""
        # Get box y-axis sensor data for orientation check
        object_z_axis = data.sensordata[self.object_z_axis_idx]

        # Check if y-axis is vertical (z-component should be close to 1 or -1)
        orientation_error = np.abs(object_z_axis[2] - 1.0)
        orientation_success = orientation_error <= 0.1  # Small tolerance

        return bool(orientation_success)

    def failure(
        self, model: MjModel, data: MjData, config: SpotBoxUprightConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[..., self.body_pose_idx[2]]
        return body_height <= config.spot_fallen_threshold
