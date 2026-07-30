from dataclasses import dataclass, field

import tyro

from learning.dynamics.experiments.utils import TrainingConfig
from learning.dynamics.models import MLP, MLPConfig
from learning.dynamics.train import run_experiment
from learning.dynamics.tasks import CylinderPushTask

from learning.dynamics.utils import get_judo_path
@dataclass
class MLPTrainingConfig(TrainingConfig):
    task: CylinderPushTask = field(default_factory=CylinderPushTask)
    dataset_file_path: str ="learning/data/cylinder_push.h5"
    horizon: float = 1.0
    task_timestep: float = 0.02
    batch_size: int = 128
    action_decimation_factor: int = 2
    prediction_decimation_factor: int = 1
    lr: float = 1e-3
    sequential: bool = False
    device: str = "cuda"
    output_activation: str = "linear"
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    layer_activations: list[str] = field(default_factory=lambda: ["relu", "relu"])
    gradient_steps: int = 20_000
    save_freq: int = 500
    log_freq: int = 100
    experiment_name: str = "cylinder_push_mlp"



def main(training_cfg: MLPTrainingConfig) -> None:
    look_ahead_steps = training_cfg.prediction_horizon_steps if not training_cfg.sequential else 1
    print(f"Saving models to {get_judo_path()}/learning/models/")
    model_cfg = MLPConfig(
        sequential=training_cfg.sequential,
        input_size=training_cfg.task.robot_frame_state_size + training_cfg.task.action_size * look_ahead_steps // training_cfg.action_decimation_factor,
        output_size=training_cfg.task.world_frame_state_size * look_ahead_steps // training_cfg.prediction_decimation_factor,
        output_activation=training_cfg.output_activation,
        hidden_sizes=training_cfg.hidden_sizes,
        layer_activations=training_cfg.layer_activations,
        batch_size=training_cfg.batch_size,
        device=training_cfg.device,
    )
    model = MLP(model_cfg)
    run_experiment(model, training_cfg, training_cfg.experiment_name, timestamped=True)


if __name__ == "__main__":
    main(tyro.cli(MLPTrainingConfig))