import random

from pathlib import Path

from learning.dynamics.models import load_model, BaseModel, BaseModelConfig
from learning.dynamics.experiments.utils import TrainingConfig, HDF5DatasetConfig

def load_dynamics_model_and_cfgs(
    experiment_dir: str | Path, 
    checkpoint: str | Path = "best_model.pth", 
    load_supplementary_configs: bool = False,
    randomize_seed: bool = False,
    overwrite_batch: int | None = None,

) -> tuple[BaseModel, BaseModelConfig, TrainingConfig | None, HDF5DatasetConfig | None]:
    model, model_cfg, training_cfg, dataset_cfg = load_model(experiment_dir, checkpoint, load_supplementary_configs)
    training_cfg = TrainingConfig.from_dict(training_cfg) if training_cfg else None
    if dataset_cfg is not None:
        if randomize_seed:
            dataset_cfg["seed"] = random.randint(0, 1000000)
        if overwrite_batch is not None:
            dataset_cfg["batch_size"] = overwrite_batch
        dataset_cfg = HDF5DatasetConfig.from_dict(dataset_cfg)
    return model, model_cfg, training_cfg, dataset_cfg