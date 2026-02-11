# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireNavigate task - navigate Spot with tire obstacle avoidance.

Adapted from starfish/dexterity/tasks/spot_tire_navigate.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT, TIRE_HALF_WIDTH
from judo.tasks.spot.spot_utils import quat_mul
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire" / "robot.xml")


@dataclass
class SpotTireNavigateConfig(SpotBaseConfig):
    """Configuration for the SpotTireNavigate task."""

    w_goal_xy: float = 50.0
    w_goal_z: float = 5.0
    w_avoid: float = 25.0
    avoid_radius: float = 1.1
    w_facing: float = 50.0
    w_obj_velocity: float = 1000.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.0]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotTireNavigate(SpotBase[SpotTireNavigateConfig]):
    """Spot navigate task with a tire obstacle in the scene."""

    name: str = "spot_tire_navigate"
    config_t: type[SpotTireNavigateConfig] = SpotTireNavigateConfig  # type: ignore[assignment]
    config: SpotTireNavigateConfig

    def __init__(
        self,
        config: SpotTireNavigateConfig | None = None,
    ) -> None:
        """Initialize the SpotTireNavigate task."""
        super().__init__(model_path=XML_PATH, use_arm=False, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("tire_rubber_joint")
        self.body_x_axis_idx = self.get_sensor_start_index("body_x_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire navigate task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq : self.model.nq + self.model.nv]

        p_WB = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        p_W_goal = self.config.goal_position

        # Distance to target (separate XY, Z)
        delta_xy = p_WB[..., :2] - p_W_goal[:2]
        dist_xy = np.linalg.norm(delta_xy, axis=-1)
        dz = np.abs(p_WB[..., 2] - p_W_goal[2])

        goal_reward = (
            -self.config.w_goal_xy * dist_xy.mean(axis=-1)
            - self.config.w_goal_z * dz.mean(axis=-1)
        )

        # Robot facing the target goal
        robot_x_axis_w = sensors[..., self.body_x_axis_idx : self.body_x_axis_idx + 3]
        dir_to_goal = p_W_goal - p_WB
        u_to_goal = dir_to_goal / np.maximum(np.linalg.norm(dir_to_goal, axis=-1, keepdims=True), 1e-6)
        facing_reward = (
            +self.config.w_facing
            * dist_xy[..., 0]
            * np.sum(robot_x_axis_w * u_to_goal, axis=-1).mean(axis=-1)
        )

        # Obstacle avoidance
        p_obj = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        d = np.linalg.norm(p_WB[..., :2] - p_obj[..., :2], axis=-1)
        total_penalty = np.maximum(0.0, self.config.avoid_radius - d)
        avoid_penalty = -self.config.w_avoid * np.mean(total_penalty, axis=-1)

        # Object velocity penalty
        # Tire free joint velocity starts at nv offset for the tire joint
        tire_vel_start = self.model.jnt_dofadr[self.model.joint("tire_rubber_joint").id]
        v_obj = qvel[..., tire_vel_start : tire_vel_start + 6]
        vel_magnitude = np.linalg.norm(v_obj, axis=-1)
        obj_velocity_penalty = -self.config.w_obj_velocity * np.mean(vel_magnitude, axis=-1)

        assert facing_reward.shape == (batch_size,)
        assert avoid_penalty.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert obj_velocity_penalty.shape == (batch_size,)

        return goal_reward + avoid_penalty + facing_reward + obj_velocity_penalty

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the tire navigate task."""
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
