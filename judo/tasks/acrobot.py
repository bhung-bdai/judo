# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm

XML_PATH = str(MODEL_PATH / "xml/acrobot.xml")


@dataclass
class AcrobotConfig(TaskConfig):
    """Reward configuration for the acrobot task."""

    w_vertical_hip: float = 30.0
    w_vertical_knee: float = 20.0
    w_velocity: float = 0.001
    w_control: float = 0.001


class Acrobot(Task[AcrobotConfig]):
    """Defines the acrobot balancing task."""

    name: str = "acrobot"
    config_t: type[AcrobotConfig] = AcrobotConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the acrobot task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.hip_pos_adr = self.get_joint_position_start_index("hip")
        self.knee_pos_adr = self.get_joint_position_start_index("knee")
        self.vel_inds = [self.hip_pos_adr, self.knee_pos_adr]

        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the acrobot reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The acrobot reward has four terms:
            * `vertical_rew`, penalizing the distance between the pole angle and vertical.
            * `velocity_rew` penalizing squared linear and angular velocity.
            * `control_rew` penalizing any actuation.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        hip_pos = qpos[..., self.hip_pos_adr]
        knee_pos = qpos[..., self.knee_pos_adr]
        vel = qvel[..., self.vel_inds]

        vertical_knee_rew = -self.config.w_vertical_knee * quadratic_norm(np.cos(hip_pos + knee_pos) - 1)
        vertical_hip_rew = -self.config.w_vertical_hip * quadratic_norm(np.cos(hip_pos) - 1)
        velocity_rew = -self.config.w_velocity * quadratic_norm(vel).sum(-1)
        control_rew = -self.config.w_control * quadratic_norm(controls).sum(-1)

        assert vertical_knee_rew.shape == (batch_size,)
        assert vertical_hip_rew.shape == (batch_size,)
        assert velocity_rew.shape == (batch_size,)
        assert control_rew.shape == (batch_size,)

        return vertical_knee_rew + vertical_hip_rew + velocity_rew + control_rew

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        self.data.qpos = np.array([np.pi, 0.5])
        self.data.qvel = 1e-12 * np.random.randn(2)
        mujoco.mj_forward(self.model, self.data)
