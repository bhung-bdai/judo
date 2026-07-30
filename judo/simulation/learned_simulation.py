

from omegaconf import DictConfig
from pathlib import Path

import numpy as np
from mujoco import mj_forward

import torch
from judo.simulation.base import Simulation

from learning.dynamics.utils import get_judo_path
from learning.dynamics.models import BaseModel
from learning.dynamics.tasks import resolve_task_name
from learning.dynamics.loader import load_dynamics_model_and_cfgs
from learning.dynamics.experiments.utils import TrainingConfig


class LearnedSimulation(Simulation):
    """Simulation class for learned simulations."""

    def __init__(
        self,
        init_task: str = "cylinder_push",
        task_registration_cfg: DictConfig | None = None,
        experiment_dir: str | Path = "cylinder_push_rnn/20260723_151618",
    ) -> None:
        """Initialize the learned simulation with a learned model."""
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)
        self.model, self.model_cfg, self.training_cfg, _ = load_dynamics_model_and_cfgs(
            experiment_dir, load_supplementary_configs=True, randomize_seed=True, overwrite_batch=1,
        )
        assert self.training_cfg is not None
        self.task_space = resolve_task_name(self.training_cfg.task)
        self.device = self.training_cfg.device

    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward.

        Args:
            command: Control command for this timestep.
        """
        if self.paused:
            return
        self.current_qpos = self.task.data.qpos.copy()
        self.current_qvel = self.task.data.qvel.copy()
        command = self.task.task_to_sim_ctrl(command)[np.newaxis, np.newaxis, :]
        current_state = torch.from_numpy(np.concatenate([self.current_qpos, self.current_qvel], axis=0))[np.newaxis, :].to(dtype=torch.float32, device=self.device)
        action_horizon_command = torch.from_numpy(
            np.repeat(command, self.training_cfg.action_horizon_steps, axis=1)
        ).to(dtype=torch.float32, device=self.device)
        predicted_states = self.model.rollout_sequential(
            self.task_space,
            current_state,
            action_horizon_command,
            self.training_cfg.prediction_horizon_steps,
            teacher_forcing=None,
        )
        print(f"Predicted states shape: {predicted_states.shape}")
        print(f"Predicted states: {predicted_states}")
        next_state = predicted_states[0, 1, :].detach().cpu().numpy()
        self.task.pre_sim_step()
        self.task.data.qpos[:] = next_state[:self.task.model.nq]
        self.task.data.qvel[:] = next_state[self.task.model.nq:]
        # Just run forward kinematics, model should be doing the rest
        mj_forward(self.task.sim_model, self.task.data)
        self.task.post_sim_step()
