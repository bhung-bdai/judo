# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotHandNavigate task - navigate Spot's end effector to a goal.

Adapted from starfish/dexterity/tasks/spot_hand_navigate.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from judo.utils.fields import np_1d_field


@dataclass
class SpotHandNavigateConfig(SpotBaseConfig):
    """Configuration for the SpotHandNavigate task."""

    w_goal: float = 60.0
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.5, 0.0, 0.5]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotHandNavigate(SpotBase[SpotHandNavigateConfig]):
    """Task getting Spot to navigate its end effector to a desired goal location."""

    name: str = "spot_hand_navigate"
    config_t: type[SpotHandNavigateConfig] = SpotHandNavigateConfig  # type: ignore[assignment]
    config: SpotHandNavigateConfig

    def __init__(
        self,
        config: SpotHandNavigateConfig | None = None,
    ) -> None:
        """Initialize the SpotHandNavigate task."""
        super().__init__(use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the hand navigate task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        spot_fallen_reward = -self.config.fall_penalty * (
            body_height <= self.config.spot_fallen_threshold
        ).any(axis=-1)

        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        goal_reward = -self.config.w_goal * np.linalg.norm(
            gripper_pos - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return spot_fallen_reward + goal_reward + controls_reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the hand navigate task."""
        return np.array(
            [*np.random.randn(2), STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos]
        )
