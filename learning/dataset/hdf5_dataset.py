

from dataclasses import dataclass, MISSING
from typing import Callable, Literal
import inspect
import logging

import h5py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def slice_padded(arr: np.ndarray, start: int, length: int, timestep_dim: int, pad_method: Literal["zero", "edge"]) -> np.ndarray:
    """Read a fixed-length window from a trajectory, padding at edges if needed.

    Args:
        arr: Trajectory array with shape (T, ..., feature_dim).
        start: Start index (may be negative).
        length: Number of timesteps to return.

    Returns:
        Array with shape (length, feature_dim).
    """
    T = arr.shape[timestep_dim]
    end = start + length
    chunk_start = max(0, start)
    chunk_end = min(T, end)
    chunk = arr[chunk_start:chunk_end]
    left_pad = chunk_start - start
    right_pad = end - chunk_end

    # If the chunk_start is less than start, we pad left. if the chunk end is greater than end, we pad right.
    # But we only need to pad the innermost row because that's the features, while the rest is dimensions.
    if left_pad or right_pad:
        pad_dims = [(0, 0) for _ in range(len(arr.shape))]
        pad_dims[timestep_dim] = (left_pad, right_pad)
        # TODO: edge vs constant of 0. In this case it's probably better to do 0 in the long run.
        if pad_method == "zero":
            chunk = np.pad(chunk, pad_dims, mode="constant", constant_values=0)
        elif pad_method == "edge":
            chunk = np.pad(chunk, pad_dims, mode="edge")
    return chunk


@dataclass
class HDF5DatasetConfig:
    file_path: str
    obs_timestep_dim: int = 0
    horizon: int = 1
    batch_size: int = 128
    shuffle: bool = True
    seed: int = 192
    out_keys: tuple[str, ...] | None = None
    action_decimation_factor: int = 1
    prediction_decimation_factor: int = 1

    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })


class HDF5Dataset(Dataset):
    """Random-window sampler over HDF5 trajectories.

    Each sample draws a random (episode, timestep) and returns a fixed-size
    history window plus a fixed-size future target window. Edge padding is
    applied when the window crosses episode boundaries.
    """

    def __init__(self, cfg: HDF5DatasetConfig):
        self.cfg = cfg
        self._ds: h5py.File | None = None  # lazy open for DataLoader workers

        with h5py.File(cfg.file_path, "r") as ds:
            self.trajectory_lengths = ds["trajectory_length"][:].astype(int)
            self.num_episodes = len(self.trajectory_lengths)
            self.dt = ds["task_timestep"][:].astype(float)

        self._rng = np.random.default_rng(self.cfg.seed)
        self._horizon_ts = np.divide(cfg.horizon, self.dt).astype(int) # number of steps in the horizon
        self._length = int(sum(self.trajectory_lengths[ep] - self._horizon_ts[ep] - 1 for ep in range(self.num_episodes)))

        if self.cfg.batch_size > int(np.max(self._length)):
            logging.warning(
                f"Batch size {self.cfg.batch_size} capped to dataset length {self._length}."
            )
            self.cfg.batch_size = int(np.max(self.trajectory_lengths))


    def _ensure_open(self) -> h5py.File:
        if self._ds is None:
            self._ds = h5py.File(self.cfg.file_path, "r")
        return self._ds

    def _sample_episode(self) -> tuple[int, int]:
        ep = int(self._rng.integers(0, self.num_episodes))
        t = int(self._rng.integers(0, self.trajectory_lengths[ep] - self._horizon_ts[ep] - 1))
        return ep, t

    def _concatenate_data_window(
        self,
        ds: h5py.File,
        ep: int,
        start: int,
        length: int,
        keys: tuple[str, ...],
        timestep_dim: int,
        pad_method: Literal["zero", "edge"] = "zero"
    ) -> np.ndarray:
        chunks = [slice_padded(ds[key][ep], start, length, timestep_dim, pad_method) for key in keys]
        return np.concatenate(chunks, axis=-1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Not implemented yet")

    @property
    def obs_ts_dim(self) -> int:
        return self.cfg.obs_timestep_dim - 1

def create_h5_dataloader(cfg: HDF5DatasetConfig) -> tuple[HDF5Dataset, DataLoader]:
    dataset = HDF5Dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        pin_memory=True,
        drop_last=True,
    )
    return dataset, loader


def main() -> None:
    from learning.dynamics.utils import get_judo_path

    file_path = f"{get_judo_path()}/learning/data/cylinder_push.h5"
    # test = ["qpos", "qvel"]

    cfg = HDF5DatasetConfig(file_path=file_path, batch_size=2, obs_timestep_dim=1, horizon=1)
    dataset, loader = create_h5_dataloader(cfg)

    for batch_input, batch_target in loader:
        print(batch_input.shape)
        print(batch_target.shape)
        print(batch_input[0, -1, :])
        print(batch_input[0, -2, :])
        print(batch_input[0, 0, :])
        print(batch_target[0, 0, :])
        break


if __name__ == "__main__":
    main()
