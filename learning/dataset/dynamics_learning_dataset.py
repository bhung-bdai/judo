

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.dataset.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig

@dataclass(frozen=True)
class DatasetKeys:
    control: str = "control"
    control_viapoints: str = "control_viapoints"
    goal_pos: str = "goal_pos"
    qpos: str = "qpos"
    qvel: str = "qvel"
    reward: str = "reward"
    rollout_controls: str = "rollout_controls"
    rollout_rewards: str = "rollout_rewards"
    rollout_states: str = "rollout_states"
    sensor: str = "sensor"
    task_timestep: str = "task_timestep"
    trajectory_length: str = "trajectory_length"


class DynamicsLearningDataset(HDF5Dataset):
    """Dynamics learning dataset.

    Each sample draws a random (episode, timestep) and returns a fixed-size
    history window plus a fixed-size future target window.
    """

    def __init__(self, cfg: HDF5DatasetConfig):
        super().__init__(cfg)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Get a single item from the dataset.
        
        Output sizes:
        - current_state: (batch_size, state_size)
        - sensor_horizon: (batch_size, horizon_ts, sensor_size)
        - action_horizon: (batch_size, horizon_ts, action_size)
        - state_targets: (batch_size, state_size)
        """
        ds = self._ensure_open()
        ep, t = self._sample_episode()
        current_state = np.concatenate((ds[DatasetKeys.qpos][ep, t, :], ds[DatasetKeys.qvel][ep, t, :]))
        current_sensor = ds[DatasetKeys.sensor][ep, t, :]
        # TODO(@bhung): flatten both of these later.
        action_horizon = self._concatenate_data_window(
            ds, ep, t, self._horizon_ts[ep], (DatasetKeys.control,), self.obs_ts_dim, pad_method="zero"
        )
        # print("action_horizon: ", action_horizon.shape)
        action_horizon = action_horizon[::self.cfg.action_decimation_factor, :]
        state_targets = self._concatenate_data_window(
            ds, ep, t + 1, self._horizon_ts[ep], (DatasetKeys.qpos, DatasetKeys.qvel), self.obs_ts_dim, pad_method="edge"
        )
        state_targets = state_targets[::self.cfg.prediction_decimation_factor, :]
        sensor_targets = self._concatenate_data_window(
            ds, ep, t + 1, self._horizon_ts[ep], (DatasetKeys.sensor,), self.obs_ts_dim, pad_method="edge"
        )
        sensor_targets = sensor_targets[::self.cfg.prediction_decimation_factor, :]
        return (
            torch.as_tensor(current_state, dtype=torch.float32),
            torch.as_tensor(current_sensor, dtype=torch.float32),
            torch.as_tensor(action_horizon, dtype=torch.float32),
            torch.as_tensor(state_targets, dtype=torch.float32),
            torch.as_tensor(sensor_targets, dtype=torch.float32),
        )


def create_dynamics_learning_dataset(cfg: HDF5DatasetConfig) -> tuple[DynamicsLearningDataset, DataLoader]:
    dataset = DynamicsLearningDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        pin_memory=True,
        drop_last=True,
    )
    return dataset, loader

