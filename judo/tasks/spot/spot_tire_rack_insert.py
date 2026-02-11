# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireRackInsert task - insert a tire into a rack.

Adapted from starfish/dexterity/tasks/spot_tire_rack_insert.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    TIRE_RADIUS,
)

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire_rack" / "robot.xml")

DEFAULT_TORSO_POSITION = np.array([2.5, 0, STANDING_HEIGHT])


@dataclass
class SpotTireRackInsertConfig(SpotBaseConfig):
    """Configuration for the SpotTireRackInsert task."""

    w_tire_alignment: float = 50.0
    w_tire_upright: float = 20.0
    w_tire_goal: float = 150.0
    w_torso_proximity: float = 5.0
    w_torso_orientation: float = 5.0
    w_gripper_proximity: float = 20.0
    fall_penalty: float = 10_000.0
    w_controls: float = 1.0
    w_object_velocity: float = 10.0
    tire_fallen_threshold: float = TIRE_RADIUS - 0.05


class SpotTireRackInsert(SpotBase[SpotTireRackInsertConfig]):
    """Task getting Spot to insert a tire into a rack."""

    name: str = "spot_tire_rack_insert"
    config_t: type[SpotTireRackInsertConfig] = SpotTireRackInsertConfig  # type: ignore[assignment]
    config: SpotTireRackInsertConfig

    def __init__(
        self,
        config: SpotTireRackInsertConfig | None = None,
    ) -> None:
        """Initialize the SpotTireRackInsert task."""
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("tire_rubber_joint")
        self.rack_pose_idx = self.get_joint_position_start_index("tire_rack_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.body_x_axis_idx = self.get_sensor_start_index("body_x_axis")
        self.tire_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.rack_x_axis_idx = self.get_sensor_start_index("rack_x_axis")
        self.object_vel_idx = self.model.jnt_dofadr[self.model.joint("tire_rubber_joint").id]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire rack insert task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        p_WB = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        p_WG = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        p_WT = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        p_WR = qpos[..., self.rack_pose_idx : self.rack_pose_idx + 3]

        Bx_W = sensors[..., self.body_x_axis_idx : self.body_x_axis_idx + 3]
        Ty_W = sensors[..., self.tire_y_axis_idx : self.tire_y_axis_idx + 3]
        Rx_W = sensors[..., self.rack_x_axis_idx : self.rack_x_axis_idx + 3]

        # Body behind tire
        Tprimez_W = np.array([0, 0, 1])
        candidate_1_Tprimex_W = np.cross(Ty_W, Tprimez_W)
        candidate_2_Tprimex_W = -candidate_1_Tprimex_W
        p_TR_W = -p_WT + p_WR
        Tprimex_W = np.where(
            np.repeat(
                (np.einsum("ijk,ijk->ij", p_TR_W, candidate_1_Tprimex_W) > 0)[:, :, np.newaxis], repeats=3, axis=-1
            ),
            candidate_1_Tprimex_W,
            candidate_2_Tprimex_W,
        )

        p_WBdes = p_WT - Tprimex_W * (TIRE_RADIUS + 1.0)
        p_WBdes[..., 2] = STANDING_HEIGHT
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(p_WB - p_WBdes, axis=-1).mean(axis=-1)

        # Body orientation
        dot_product_orient = np.abs(np.einsum("ijk,ijk->ij", Bx_W, Ty_W))
        torso_orientation_reward = -self.config.w_torso_orientation * dot_product_orient.mean(axis=-1)

        # Gripper proximity
        p_TP1_W = -Tprimex_W * TIRE_RADIUS
        p_TP2_W = np.array([0, 0, -TIRE_RADIUS])
        p_TGdes_W = p_TP1_W * 1 / 2 + p_TP2_W * (3**0.5) / 2
        p_WGdes = p_WT + p_TGdes_W
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(p_WG - p_WGdes, axis=-1).mean(
            axis=-1
        )

        # Tire alignment
        alignment_error = np.abs(np.einsum("ijk,ijk->ij", Ty_W, Rx_W))
        tire_alignment_reward = -self.config.w_tire_alignment * alignment_error.mean(axis=-1)

        # Tire upright
        upright_error = np.abs(Ty_W[..., 2])
        tire_upright_reward = -self.config.w_tire_upright * upright_error.mean(axis=-1)

        # Tire goal
        p_Rx = p_WR[0, 0, 0]
        p_Ry = p_WR[0, 0, 1]
        p_WTdes = np.array([p_Rx, p_Ry, TIRE_RADIUS])
        tire_goal_reward = -self.config.w_tire_goal * np.linalg.norm(p_WT - p_WTdes, axis=-1).mean(axis=-1)

        # Fall penalties
        body_height = qpos[..., self.body_pose_idx + 2]
        spot_fallen_reward = -self.config.fall_penalty * (
            body_height <= self.config.spot_fallen_threshold
        ).any(axis=-1)

        tire_height = qpos[..., self.object_pose_idx + 2]
        tire_fallen_reward = -self.config.fall_penalty * (
            tire_height <= self.config.tire_fallen_threshold
        ).any(axis=-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        object_velocity = qvel[..., self.object_vel_idx : self.object_vel_idx + 3]
        object_velocity_reward = -self.config.w_object_velocity * np.exp(
            np.linalg.norm(object_velocity, axis=-1)
        ).mean(axis=-1)

        assert torso_proximity_reward.shape == (batch_size,)
        return (
            torso_proximity_reward + torso_orientation_reward + gripper_proximity_reward
            + tire_alignment_reward + tire_upright_reward + tire_goal_reward
            + spot_fallen_reward + tire_fallen_reward + controls_reward + object_velocity_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the tire rack insert task."""
        # Randomize rack pose
        rack_pose = (np.random.rand(7) - 0.5) * 2.0
        rack_pose[2] = TIRE_RADIUS
        yaw_rad = np.deg2rad(np.random.uniform(-20, 20))
        rack_pose[3:] = [np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)]
        robot_radius = 1.0
        while np.linalg.norm(rack_pose[:3] - DEFAULT_TORSO_POSITION) < robot_radius:
            rack_pose = (np.random.rand(7) - 0.5) * 2.0
            rack_pose[2] = TIRE_RADIUS
            yaw_rad = np.deg2rad(np.random.uniform(-20, 20))
            rack_pose[3:] = [np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)]

        # Tire pose in between robot and rack
        robot_xy = DEFAULT_TORSO_POSITION[:2]
        rack_xy = rack_pose[:2]
        rack_to_robot_dist = np.linalg.norm(robot_xy - rack_xy)
        tire_dist = np.random.uniform(0.5, max(0.6, rack_to_robot_dist - robot_radius))
        tire_angle = yaw_rad
        tire_offset = np.array([tire_dist * np.cos(tire_angle), tire_dist * np.sin(tire_angle)])
        tire_xy = rack_xy + tire_offset
        tire_z = TIRE_RADIUS
        tire_quat = rack_pose[3:]

        tire_pose = np.zeros(7)
        tire_pose[:2] = tire_xy
        tire_pose[2] = tire_z
        tire_pose[3:] = tire_quat

        return np.array(
            [*DEFAULT_TORSO_POSITION, 0, 0, 0, 1, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose, *rack_pose]
        )
