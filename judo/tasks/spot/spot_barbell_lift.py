# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotBarbellLift task - lift and move a barbell to a goal.

Adapted from starfish/dexterity/tasks/spot_barbell_lift.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT, WR1_T_GRASP
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_barbell" / "robot.xml")


@dataclass
class SpotBarbellLiftConfig(SpotBaseConfig):
    """Configuration for the SpotBarbellLift task."""

    robot_proximity_radius: float = 0.75
    w_goal: float = 50.0
    w_goal_height: float = 400.0
    w_torso_proximity: float = 50.0
    w_gripper_proximity: float = 250.0
    w_gripper_barbell_orientation: float = 50.0
    w_barbell_orientation: float = 100.0
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 1.0]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotBarbellLift(SpotBase[SpotBarbellLiftConfig]):
    """Task getting Spot to lift and move a barbell to a desired goal location."""

    name: str = "spot_barbell_lift"
    config_t: type[SpotBarbellLiftConfig] = SpotBarbellLiftConfig  # type: ignore[assignment]
    config: SpotBarbellLiftConfig

    def __init__(
        self,
        config: SpotBarbellLiftConfig | None = None,
    ) -> None:
        """Initialize the SpotBarbellLift task."""
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("barbell_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.gripper_x_axis_idx = self.get_sensor_start_index("gripper_x_axis")
        self.gripper_y_axis_idx = self.get_sensor_start_index("gripper_y_axis")
        self.barbell_x_axis_idx = self.get_sensor_start_index("object_x_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the barbell lift task."""
        batch_size = states.shape[0]
        reward = np.zeros((batch_size,))
        qpos = states[..., : self.model.nq]

        W_p_torso = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        W_p_gripper = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        W_p_barbell = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx : self.gripper_x_axis_idx + 3]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx : self.gripper_y_axis_idx + 3]
        barbell_x_axis = sensors[..., self.barbell_x_axis_idx : self.barbell_x_axis_idx + 3]

        W_p_gripper_tip = W_p_gripper + WR1_T_GRASP * gripper_x_axis

        # Goal reward
        reward += -self.config.w_goal * np.linalg.norm(
            W_p_barbell - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        # Goal height reward
        goal_height_error = np.abs(W_p_barbell[..., 2] - self.config.goal_position[2]).mean(axis=-1)
        reward += -self.config.w_goal_height * goal_height_error

        # Torso proximity reward
        torso_t_barbell = W_p_barbell[..., :2] - W_p_torso[..., :2]
        torso_t_barbell_norm = (
            self.config.robot_proximity_radius
            - np.clip(np.linalg.norm(torso_t_barbell, axis=-1), 0, self.config.robot_proximity_radius)
        ) / self.config.robot_proximity_radius
        reward += -self.config.w_torso_proximity * torso_t_barbell_norm.mean(axis=-1)

        # Gripper proximity reward
        reward += -self.config.w_gripper_proximity * np.linalg.norm(
            W_p_gripper_tip - W_p_barbell, axis=-1
        ).mean(axis=-1)

        # Barbell orientation reward
        orientation_error = 1.0 - np.linalg.norm(barbell_x_axis[..., :2], axis=-1)
        reward += -self.config.w_barbell_orientation * orientation_error.mean(axis=-1)

        # Gripper-object orientation alignment
        gripper_y_axis_norm = gripper_y_axis / (np.linalg.norm(gripper_y_axis, axis=-1, keepdims=True) + 1e-8)
        barbell_x_axis_norm = barbell_x_axis / (np.linalg.norm(barbell_x_axis, axis=-1, keepdims=True) + 1e-8)
        dot_product = np.clip(np.sum(gripper_y_axis_norm * barbell_x_axis_norm, axis=-1), -1.0, 1.0)
        reward += -self.config.w_gripper_barbell_orientation * (1.0 - np.abs(dot_product)).mean(axis=-1)

        # Fall penalty
        body_height = qpos[..., self.body_pose_idx + 2]
        reward += -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Controls penalty
        reward += -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        return reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the barbell lift task."""
        standing_pose = np.array([0, 0, STANDING_HEIGHT])
        robot_radius = 1.0

        object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        while np.linalg.norm(object_xyz[:2]) < robot_radius:
            object_xyz = (np.random.rand(3) * 2.0 - 1.0) * 3.0
        object_xyz[2] = 0.1

        theta = np.random.uniform(0, 2 * np.pi)
        object_orientation = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

        return np.array(
            [*standing_pose, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_xyz, *object_orientation]
        )
