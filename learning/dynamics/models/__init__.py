

from learning.dynamics.models.base import BaseModel, BaseModelConfig
from learning.dynamics.models.mlp import MLP, MLPConfig
from learning.dynamics.models.rnn import RNN, RNNConfig
from learning.dynamics.utils import get_judo_path
from learning.dataset.dynamics_learning_dataset import HDF5DatasetConfig

import json
from pathlib import Path


model_registry = {
    "mlp": (MLP, MLPConfig),
    "rnn": (RNN, RNNConfig),
}


def resolve_model_name(model_name: str) -> tuple[BaseModel, BaseModelConfig]:
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not found in registry")
    return model_registry[model_name.lower()]


def load_model(
    experiment_dir: str | Path, checkpoint: str | Path = "best_model.pth", load_supplementary_configs: bool = False
    ) -> tuple[BaseModel, BaseModelConfig, dict | None, dict | None]:
    """Loads a model and its config from an experiment directory.
    
    Args:
        experiment_dir: The directory containing the experiment.
        checkpoint: The checkpoint to load. Defaults to the best model in the experiment directory "best_model.pth".

    Returns:
        A tuple containing the model and its config.
    """
    experiment_dir = Path(f"{get_judo_path()}/learning/dynamics/experiments/{experiment_dir}")
    model_cfg = json.load(open(experiment_dir / "model_config.json"))
    model_class, model_config_class = resolve_model_name(model_cfg["name"])
    model_cfg = model_config_class(**model_cfg)
    model = model_class(model_cfg)
    model.load_checkpoint(str(experiment_dir / "checkpoints" / checkpoint))
    training_cfg = None
    dataset_cfg = None

    if load_supplementary_configs:
        training_cfg = json.load(open(experiment_dir / "training_config.json"))
        dataset_cfg = json.load(open(experiment_dir / "dataset_config.json"))
    return (model, model_cfg, training_cfg, dataset_cfg)


__all__ = [
    "BaseModel", 
    "BaseModelConfig", 
    "MLP", 
    "MLPConfig", 
    "RNN", 
    "RNNConfig", 
    "resolve_model_name", 
    "load_model"
]