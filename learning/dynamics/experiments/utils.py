
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from learning.dataset.dynamics_learning_dataset import HDF5DatasetConfig
from learning.dynamics.tasks import Task
from learning.dynamics.utils import get_judo_path
from learning.dynamics.models.base import BaseModel

@dataclass
class TrainingConfig:
    # Task
    task: Task

    # Dataset configuration
    dataset_file_path: str
    obs_timestep_dim: int = 1

    # Training hyperparameters
    batch_size: int = 128
    gradient_steps: int = 20_000
    lr: float = 1e-3
    max_grad_norm: float = 1.0

    # Model config
    output_activation: str | None = None
    device: str = "cuda"

    # Logging and checkpointing
    save_freq: int = 5_000
    log_freq: int = 500

    # Task-specific
    horizon: float = 1.0  # in seconds
    task_timestep: float = 0.02  # in seconds
    action_decimation_factor: int = 1
    prediction_decimation_factor: int = 1
    sequential: bool = False
    teacher_forcing: bool = False

    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })


    def __post_init__(self) -> None:
        self.prediction_horizon_steps = int(self.horizon / self.task_timestep) // self.prediction_decimation_factor
        self.action_horizon_steps = int(self.horizon / self.task_timestep) // self.action_decimation_factor


@dataclass
class EvalConfig:
    # Evaluation configuration
    experiment_dir: str | Path
    eval_batch: int


def create_experiment_dir(experiment_name: str, *, timestamped: bool) -> Path:
    """Create an experiment directory under ``learning/dynamics/experiments``.

    Args:
        experiment_name: Base name for the experiment (e.g. ``cylinder_push_rnn``).
        timestamped: If True, nest the run under a timestamped subdirectory.

    Returns:
        Path to the created experiment directory.
    """
    experiment_dir = Path(f"{get_judo_path()}/learning/dynamics/experiments/{experiment_name}")
    if timestamped:
        experiment_dir = experiment_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    return experiment_dir


def serialize_training_config(training_cfg: TrainingConfig) -> dict[str, Any]:
    """Convert a training config to a JSON-serializable dictionary."""
    cfg_dict = asdict(training_cfg)
    cfg_dict["task"] = type(training_cfg.task).__name__
    return cfg_dict


def save_experiment_artifacts(
    experiment_dir: Path,
    *,
    model_config: dict[str, Any],
    training_cfg: TrainingConfig,
    dataset_cfg: HDF5DatasetConfig | None = None,
) -> None:
    """Save model, training, and dataset configs into an experiment directory."""
    with open(experiment_dir / "model_config.json", "w", encoding="utf-8") as model_config_file:
        json.dump(model_config, model_config_file, indent=4)
    with open(experiment_dir / "training_config.json", "w", encoding="utf-8") as training_config_file:
        json.dump(serialize_training_config(training_cfg), training_config_file, indent=4)
    if dataset_cfg is not None:
        with open(experiment_dir / "dataset_config.json", "w", encoding="utf-8") as dataset_config_file:
            json.dump(asdict(dataset_cfg), dataset_config_file, indent=4)


def setup_experiment(
    experiment_name: str,
    *,
    model: BaseModel,
    training_cfg: TrainingConfig,
    dataset_cfg: HDF5DatasetConfig | None = None,
    timestamped: bool = False,
) -> Path:
    """Create an experiment directory and save configs before training.

    Args:
        experiment_name: Base name for the experiment.
        model: Model whose config will be saved.
        training_cfg: Training configuration to save.
        timestamped: If True, create a new timestamped subdirectory for this run.

    Returns:
        Path to the experiment directory where checkpoints should be written.
    """
    experiment_dir = create_experiment_dir(experiment_name, timestamped=timestamped)
    save_experiment_artifacts(
        experiment_dir,
        model_config=asdict(model.cfg),
        training_cfg=training_cfg,
        dataset_cfg=dataset_cfg,
    )
    return experiment_dir


def checkpoint_path(experiment_dir: Path, model_name: str, step: int) -> Path:
    """Return the checkpoint path for a given training step."""
    return experiment_dir / "checkpoints" / f"world_model_{model_name}_{step}.pth"


def save_checkpoint(experiment_dir: Path, model: BaseModel, step: int) -> Path:
    """Save a model checkpoint into the experiment directory."""
    checkpoint_file = checkpoint_path(experiment_dir, model.name, step)
    torch.save(model.state_dict(), checkpoint_file)
    return checkpoint_file
