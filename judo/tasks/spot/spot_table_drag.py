# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTableDrag task - drag a table to a goal location.

Adapted from starfish/dexterity/tasks/spot_table_drag.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT, TABLE_HEIGHT, TABLE_WIDTH, WR1_T_GRASP
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml" / "spot_table_short" / "robot.xml")


@dataclass
class SpotTableDragConfig(SpotBaseConfig):
    """Configuration for the SpotTableDrag task."""

    robot_proximity_radius: float = 1.0
    w_goal: float = 500.0
    w_torso_proximity: float = 400.0
    w_gripper_proximity: float = 250.0
    w_gripper_orientation: float = 1000.0
    w_grasp_cone: float = 250.0
    w_gripper_open_threshold: float = 0.2
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, TABLE_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotTableDrag(SpotBase[SpotTableDragConfig]):
    """Task for Spot to drag a table to a desired goal location."""

    name: str = "spot_table_drag"
    config_t: type[SpotTableDragConfig] = SpotTableDragConfig  # type: ignore[assignment]
    config: SpotTableDragConfig

    def __init__(
        self,
        config: SpotTableDragConfig | None = None,
    ) -> None:
        """Initialize the SpotTableDrag task."""
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("table_short_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.gripper_x_axis_idx = self.get_sensor_start_index("gripper_x_axis")
        self.gripper_y_axis_idx = self.get_sensor_start_index("gripper_y_axis")
        self.finger_x_axis_idx = self.get_sensor_start_index("finger_x_axis")
        self.table_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.table_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.table_z_axis_idx = self.get_sensor_start_index("object_z_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the table drag task."""
        batch_size = states.shape[0]
        reward = np.zeros((batch_size,))
        qpos = states[..., : self.model.nq]

        W_p_torso = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        W_p_gripper = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        W_p_table = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx : self.gripper_x_axis_idx + 3]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx : self.gripper_y_axis_idx + 3]
        table_x_axis = sensors[..., self.table_x_axis_idx : self.table_x_axis_idx + 3]
        table_z_axis = sensors[..., self.table_z_axis_idx : self.table_z_axis_idx + 3]

        W_p_gripper_tip = W_p_gripper + WR1_T_GRASP * gripper_x_axis

        # Contact points on table edges
        contact_point_list = []
        for i in [-1, 1]:
            contact_point_list.append(
                W_p_table + table_x_axis * i * (TABLE_WIDTH / 2) + table_z_axis * TABLE_HEIGHT
            )
        contact_points = np.stack(contact_point_list, axis=-2)

        # Goal reward (XY only)
        reward += -self.config.w_goal * np.linalg.norm(
            W_p_table[..., :2] - self.config.goal_position[:2], axis=-1
        ).mean(-1)

        # Torso proximity
        torso_t_table = W_p_table[..., :2] - W_p_torso[..., :2]
        torso_t_table_norm = (
            self.config.robot_proximity_radius
            - np.clip(np.linalg.norm(torso_t_table, axis=-1), 0, self.config.robot_proximity_radius)
        ) / self.config.robot_proximity_radius
        reward += -self.config.w_torso_proximity * torso_t_table_norm.mean(axis=-1)

        # Gripper proximity to contact points
        gripper_to_contacts = W_p_gripper_tip[..., None, :] - contact_points
        gripper_to_contacts_dist = np.linalg.norm(gripper_to_contacts, axis=-1)
        contact_index = gripper_to_contacts_dist.mean(axis=-2).argmin(axis=-1)
        row_indices = np.arange(batch_size)
        gripper_t_contact_dist = gripper_to_contacts_dist[row_indices, :, contact_index].mean(axis=-1)
        reward += -self.config.w_gripper_proximity * gripper_t_contact_dist

        # Gripper too close to table center penalty
        gripper_t_table = W_p_table[..., :2] - W_p_gripper_tip[..., :2]
        gripper_t_table_norm = (
            TABLE_WIDTH / 2 - np.clip(np.linalg.norm(gripper_t_table, axis=-1), 0, TABLE_WIDTH / 2)
        ) / (TABLE_WIDTH / 2)
        reward += -100.0 * self.config.w_gripper_proximity * gripper_t_table_norm.mean(axis=-1)

        # Gripper orientation: align with contact direction
        closest_contact_point = contact_points[row_indices, :, contact_index, :]
        contact_direction = W_p_table + np.array([0, 0, TABLE_HEIGHT]) - closest_contact_point
        contact_direction_norm = contact_direction / (np.linalg.norm(contact_direction, axis=-1, keepdims=True) + 1e-8)
        gripper_x_axis_norm = gripper_x_axis / (np.linalg.norm(gripper_x_axis, axis=-1, keepdims=True) + 1e-8)
        dot_product = np.clip(np.sum(gripper_x_axis_norm * contact_direction_norm, axis=-1), -1.0, 1.0)
        reward += -self.config.w_gripper_orientation * (1.0 - dot_product).mean(axis=-1)

        # Gripper orientation: align with ground plane
        gripper_z_axis = np.cross(gripper_x_axis, gripper_y_axis)
        reward += -self.config.w_gripper_orientation * (1.0 - np.abs(gripper_z_axis[..., 2])).mean(axis=-1)

        # Fall penalty
        body_height = qpos[..., self.body_pose_idx + 2]
        reward += -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Controls
        reward += -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        return reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the table drag task."""
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
