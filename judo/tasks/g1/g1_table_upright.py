# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.g1.g1_base import DEFAULT_JOINT_POSITIONS, STANDING_HEIGHT, G1Base, G1BaseConfig
from judo.utils.indexing import get_pos_indices, get_sensor_indices

XML_PATH = str(MODEL_PATH / "xml/g1/g1_table.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
TABLE_HEIGHT = 0.6  # Height of the table top from the ground (0.765 + 0.015)
# Annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0

# Arm usage configuration for table manipulation
USE_LEFT_ARM = True
USE_RIGHT_ARM = True
USE_LEFT_WRIST = False
USE_RIGHT_WRIST = False


@dataclass
class G1TableUprightConfig(G1BaseConfig):
    """Config for the G1 table upright task."""

    goal_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, TABLE_HEIGHT]))
    w_goal: float = 0.0
    w_orientation: float = 15.0
    w_pelvis_proximity: float = 0.1
    w_hand_proximity: float = 10.0
    w_controls: float = 3


class G1TableUpright(G1Base):
    """Task getting G1 to upright a table from a rotated position.

    Uses both arms for manipulation by default (USE_LEFT_ARM=True, USE_RIGHT_ARM=True).
    """

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(
            model_path=model_path,
            use_left_arm=USE_LEFT_ARM,
            use_right_arm=USE_RIGHT_ARM,
            use_left_wrist=USE_LEFT_WRIST,
            use_right_wrist=USE_RIGHT_WRIST,
        )
        self.body_pose_idx = get_pos_indices(self.model, "floating_base_joint")
        self.object_pose_idx = get_pos_indices(self.model, ["table_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "table_y_axis")
        self.left_palm_to_object_idx = get_sensor_indices(self.model, "trace_left_palm")
        self.right_palm_to_object_idx = get_sensor_indices(self.model, "trace_right_palm")

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return super().nu

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: G1TableUprightConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the G1 table upright task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]

        object_y_axis = sensors[..., self.object_y_axis_idx]
        left_palm_to_object = sensors[..., self.left_palm_to_object_idx]
        right_palm_to_object = sensors[..., self.right_palm_to_object_idx]

        # Check if any state in the rollout has G1 fallen
        g1_fallen_reward = -config.fall_penalty * (body_height <= config.fall_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal (XY only for table pushing)
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos[..., :2] - np.array(config.goal_position[:2])[None, None], axis=-1
        ).mean(-1)

        # Table orientation reward - penalize if table is not upright
        table_orientation_reward = -config.w_orientation * np.abs(np.dot(object_y_axis, Z_AXIS) - 1.0).sum(axis=-1)

        # Compute l2 distance from pelvis pos. to object pos.
        pelvis_proximity_reward = config.w_pelvis_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        # Compute l2 distance from hands to object - encourage getting hands close to table
        # Use the minimum distance from either hand
        left_hand_dist = np.linalg.norm(left_palm_to_object, axis=-1)
        right_hand_dist = np.linalg.norm(right_palm_to_object, axis=-1)
        min_hand_dist = np.minimum(left_hand_dist, right_hand_dist)
        hand_proximity_reward = -config.w_hand_proximity * min_hand_dist.mean(-1)

        # Control cost: penalize velocity commands (absolute) and arm position deviations (from default)
        base_vel = controls[..., :3]
        vel_cost = np.linalg.norm(base_vel, axis=-1).mean(-1)

        if self.use_left_arm or self.use_right_arm:
            # Penalize arm positions as deviation from default
            arm_commands = controls[..., 3:]
            arm_defaults = self.default_command[3:]
            arm_deviation = arm_commands - arm_defaults
            arm_cost = np.linalg.norm(arm_deviation, axis=-1).mean(-1)
            controls_reward = -config.w_controls * (vel_cost + arm_cost)
        else:
            controls_reward = -config.w_controls * vel_cost

        assert g1_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert table_orientation_reward.shape == (batch_size,)
        assert pelvis_proximity_reward.shape == (batch_size,)
        assert hand_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            g1_fallen_reward
            + goal_reward
            + table_orientation_reward
            + pelvis_proximity_reward
            + hand_proximity_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        # Table positioned at the origin
        object_pos = np.array([0.0, 0.0])

        # Rotate table 90 degrees on roll axis
        # Roll rotation of 90 degrees: quaternion = [cos(45°), sin(45°), 0, 0]
        roll_angle = np.pi / 2  # 90 degrees
        object_orientation = np.array([np.cos(roll_angle / 2), 0, -np.sin(roll_angle / 2), 0])

        reset_object_pose = np.array([*object_pos, TABLE_HEIGHT, *object_orientation])

        # G1 reset pose: base position + orientation + joint positions + object pose
        return np.array(
            [
                -1.5,
                -0.5,
                STANDING_HEIGHT,  # Standing height
                1,
                0,
                0,
                0,  # Quaternion (upright orientation)
                *DEFAULT_JOINT_POSITIONS,  # Joint positions
                *reset_object_pose,  # Object pose
            ]
        )
