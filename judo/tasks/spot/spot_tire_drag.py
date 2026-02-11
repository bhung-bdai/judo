# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireDrag task - drag a tire to a goal location.

Adapted from starfish/dexterity/tasks/spot_tire_drag.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    TIRE_HALF_WIDTH,
    TIRE_RADIUS,
)
from judo.tasks.spot.spot_utils import quat_mul
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire" / "robot.xml")


@dataclass
class SpotTireDragConfig(SpotBaseConfig):
    """Configuration for the SpotTireDrag task."""

    w_goal: float = 60.0
    w_gripper_proximity: float = 10.0
    w_torso_proximity: float = 5.0
    fall_penalty: float = 10_000.0
    gripper_too_inside_tire_penalty: float = 150.0
    gripper_not_above_tire_penalty: float = 100.0
    w_controls: float = 2.0
    goal_position: np.ndarray = np_1d_field(
        np.array([1.0, 1.5, 0.18]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotTireDrag(SpotBase[SpotTireDragConfig]):
    """Task getting Spot to drag a tire to a desired goal location."""

    name: str = "spot_tire_drag"
    config_t: type[SpotTireDragConfig] = SpotTireDragConfig  # type: ignore[assignment]
    config: SpotTireDragConfig

    def __init__(
        self,
        config: SpotTireDragConfig | None = None,
    ) -> None:
        """Initialize the SpotTireDrag task."""
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("tire_rubber_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire drag task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        W_p_tire = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        W_p_torso = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        W_p_gripper = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        W_p_object_goal = self.config.goal_position
        W_p_tire_goal = W_p_object_goal - W_p_tire
        W_u_tire_goal = W_p_tire_goal / (np.linalg.norm(W_p_tire_goal, axis=-1, keepdims=True) + 1e-8)

        # Gripper proximity
        W_p_gripper_des = W_p_tire + (TIRE_RADIUS - 0.05) * W_u_tire_goal
        W_p_gripper_des[..., 2] = TIRE_HALF_WIDTH + 0.1
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            W_p_gripper - W_p_gripper_des, axis=-1
        ).mean(axis=-1)

        # Torso proximity
        W_p_torso_des = W_p_tire + 0.9 * W_u_tire_goal
        W_p_torso_des[..., 2] = STANDING_HEIGHT
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(
            W_p_torso - W_p_torso_des, axis=-1
        ).mean(axis=-1)

        # Goal distance
        goal_reward = -self.config.w_goal * np.linalg.norm(W_p_tire - W_p_object_goal, axis=-1).mean(-1)

        # Penalties
        gripper_distance_from_tire = np.linalg.norm(W_p_gripper - W_p_tire, axis=-1)
        gripper_inside_tire_reward = -self.config.gripper_too_inside_tire_penalty * (
            gripper_distance_from_tire < (TIRE_RADIUS * 0.5)
        ).mean(axis=-1)

        gripper_height = W_p_gripper[..., 2]
        gripper_not_above_tire = gripper_height < 2 * TIRE_HALF_WIDTH + 0.05
        gripper_too_far_from_tire = gripper_distance_from_tire > TIRE_RADIUS
        gripper_not_above_tire_reward = -self.config.gripper_not_above_tire_penalty * (
            np.logical_and(gripper_not_above_tire, gripper_too_far_from_tire)
        ).mean(axis=-1)

        body_height = qpos[..., self.body_pose_idx + 2]
        spot_fallen_reward = -self.config.fall_penalty * (
            body_height <= self.config.spot_fallen_threshold
        ).any(axis=-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert goal_reward.shape == (batch_size,)
        return (
            goal_reward
            + gripper_proximity_reward
            + torso_proximity_reward
            + gripper_inside_tire_reward
            + gripper_not_above_tire_reward
            + spot_fallen_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the tire drag task."""
        for _ in range(100):
            tire_pose = np.zeros(7)
            tire_pose[0] = np.random.uniform(-2, 2)
            tire_pose[1] = np.random.uniform(-2, 2)
            tire_pose[2] = TIRE_HALF_WIDTH
            if np.random.random() < 0.5:
                tire_pose[3:] = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]
            else:
                tire_pose[3:] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0]
            random_yaw = np.random.uniform(0, 2 * np.pi)
            random_yaw_quat = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])
            tire_pose[3:] = quat_mul(random_yaw_quat, tire_pose[3:])

            robot_pose = np.zeros(7)
            robot_pose[0] = np.random.uniform(-2, 2)
            robot_pose[1] = np.random.uniform(-2, 2)
            robot_pose[2] = STANDING_HEIGHT
            random_yaw_robot = np.random.uniform(0, 2 * np.pi)
            robot_pose[3:] = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])

            if np.linalg.norm(robot_pose[:3] - tire_pose[:3]) > 2:
                return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])

        tire_pose = np.array([2.0, 0.0, TIRE_HALF_WIDTH, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
        return np.array([-1.75, 0, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])
