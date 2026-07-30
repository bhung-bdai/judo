# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo rollout backend for parallel trajectory simulation."""

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from judo.utils.rollout_backend import RolloutBackend
from learning.dynamics.loader import load_dynamics_model_and_cfgs
from learning.dynamics.tasks import resolve_task_name


class WorldModelRolloutBackend(RolloutBackend):
    """Backend for conducting multithreaded rollouts using a world model.

    Uses a world model to rollout the trajectory for .
    """

    def __init__(
        self,
        model: str | Path,
        num_threads: int,
        experiment_dir: str | Path,
    ) -> None:
        """Initialize the rollout backend.

        Args:
            model: MuJoCo model for the scene.
            num_threads: Number of parallel rollout threads.
        """
        self.num_threads = num_threads
        self.model, self.model_cfg, self.training_cfg, _ = load_dynamics_model_and_cfgs(
            experiment_dir, load_supplementary_configs=True, randomize_seed=True, overwrite_batch=1,
        )
        self.task_space = resolve_task_name(self.training_cfg.task)
        assert self.training_cfg is not None
        self.model.eval()

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Conduct parallel rollouts.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, nu).
            last_policy_output: Unused. Accepted for interface compatibility.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Always None for this backend.
        """
        # TODO: include an option for batched rollouts
        x0_torch = torch.from_numpy(x0).to(dtype=torch.float32, device=self.model.device).unsqueeze(0)
        x0_torch = x0_torch.repeat(self.num_threads, 1)
        controls = controls[:, ::self.training_cfg.action_decimation_factor, :]
        controls_torch = torch.from_numpy(controls).to(dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            predicted_states = self.model.rollout_sequential(
                self.task_space,
                x0_torch,
                controls_torch,
                prediction_horizon_steps=self.training_cfg.prediction_horizon_steps,
                teacher_forcing=None,
            )

        # if x0.ndim == 1:
        #     x0 = np.tile(x0, (self.num_threads, 1))

        # nq = self._models[0].nq
        # nv = self._models[0].nv
        # nu = self._models[0].nu

        # # Prepend time to batched x0
        # full_states = np.concatenate([time.time() * np.ones((len(self._models), 1)), x0], axis=-1)

        # assert full_states.shape[-1] == nq + nv + 1
        # assert full_states.ndim == 2
        # assert controls.ndim == 3
        # assert controls.shape[-1] == model.
        # assert controls.shape[0] == full_states.shape[0]

        # _states, _sensors = self._rollout_obj.rollout(self._models, self._datas, full_states, controls)

        # out_states = np.array(_states)[..., 1:]  # Remove time from state
        # out_sensors = np.array(_sensors)
        # TODO: change this to output sensor data later on.
        # print("Out shapes: ", predicted_states.shape)
        # print(predicted_states.shape)
        predicted_states = predicted_states.detach().cpu().numpy()
        predicted_sensors = self.task_space.generate_sensor_data(predicted_states)
        return predicted_states, predicted_sensors, None # out_states, np.zeros_like(), None

    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Recreates internal state (model/data pairs) for new thread count.

        Args:
            num_threads: New number of parallel threads.
        """
        self.num_threads = num_threads
