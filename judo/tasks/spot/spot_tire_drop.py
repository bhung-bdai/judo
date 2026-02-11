# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotTireDrop task - drop a tire from upright to lying flat.

Adapted from starfish/dexterity/tasks/spot_tire_drop.py.
"""

from dataclasses import dataclass

import numpy as np

from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT, TIRE_RADIUS
from judo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig

XML_PATH = str(MODEL_PATH / "xml" / "spot_tire" / "robot.xml")


@dataclass
class SpotTireDropConfig(SpotTireUprightConfig):
    """Configuration for the SpotTireDrop task."""

    w_tire_orientation: float = -200.0  # Negative = reward dropping


class SpotTireDrop(SpotTireUpright):
    """Task getting Spot to drop a tire to lie flat on the ground.

    Inherits reward from SpotTireUpright but with negative orientation weight.
    """

    name: str = "spot_tire_drop"
    config_t: type[SpotTireDropConfig] = SpotTireDropConfig  # type: ignore[assignment]
    config: SpotTireDropConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: SpotTireDropConfig | None = None,
    ) -> None:
        """Initialize the SpotTireDrop task."""
        super().__init__(config=config or SpotTireDropConfig())
        self.use_legs = False
        self.set_command_values()

    @property
    def reset_pose(self) -> np.ndarray:
        """Tire starts upright for drop task."""
        for _ in range(100):
            tire_pose = np.zeros(7)
            tire_pose[0] = np.random.uniform(-2, 2)
            tire_pose[1] = np.random.uniform(-2, 2)
            tire_pose[2] = TIRE_RADIUS  # Standing upright

            random_yaw = np.random.uniform(0, 2 * np.pi)
            tire_pose[3:] = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])

            robot_pose = np.zeros(7)
            robot_pose[0] = np.random.uniform(-2, 2)
            robot_pose[1] = np.random.uniform(-2, 2)
            robot_pose[2] = STANDING_HEIGHT
            random_yaw_robot = np.random.uniform(0, 2 * np.pi)
            robot_pose[3:] = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])

            if np.linalg.norm(robot_pose[:3] - tire_pose[:3]) > 2:
                return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])

        tire_pose = np.array([0, 0, TIRE_RADIUS, 1, 0, 0, 0])
        return np.array([-1.75, 0, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_pose])
