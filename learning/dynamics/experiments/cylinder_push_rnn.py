
from dataclasses import dataclass, field

import tyro

from learning.dynamics.experiments.utils import TrainingConfig
from learning.dynamics.models import RNN, RNNConfig
from learning.dynamics.tasks import CylinderPushTask
from learning.dynamics.train import run_experiment

@dataclass
class RNNTrainingConfig(TrainingConfig):
    task: CylinderPushTask = field(default_factory=CylinderPushTask)
    dataset_file_path: str ="learning/data/cylinder_push.h5"
    horizon: float = 1.0
    task_timestep: float = 0.02
    batch_size: int = 128
    action_decimation_factor: int = 2
    prediction_decimation_factor: int = 1
    lr: float = 1e-3
    sequential: bool = True
    device: str = "cuda"
    output_activation: str = "linear"
    gru_hidden_size: int = 256
    gru_num_layers: int = 2
    gradient_steps: int = 5_000
    save_freq: int = 500
    log_freq: int = 100
    experiment_name: str = "cylinder_push_rnn"
    teacher_forcing: bool = True


def main(training_cfg: RNNTrainingConfig) -> None:
    step_input_size = training_cfg.task.robot_frame_state_size + training_cfg.task.action_size
    model_cfg = RNNConfig(
        sequential=training_cfg.sequential,
        input_size=step_input_size,
        output_size=training_cfg.task.world_frame_state_size,
        init_input_size=(
            step_input_size
            if training_cfg.sequential
            else training_cfg.task.robot_frame_state_size
            + training_cfg.task.action_size * training_cfg.action_horizon_steps
        ),
        step_input_size=step_input_size,
        batch_size=training_cfg.batch_size,
        device=training_cfg.device,
        gru_hidden_size=training_cfg.gru_hidden_size,
        gru_num_layers=training_cfg.gru_num_layers,
        output_activation=training_cfg.output_activation,
    )
    model = RNN(model_cfg)
    run_experiment(model, training_cfg, training_cfg.experiment_name, timestamped=True)


if __name__ == "__main__":
    main(tyro.cli(RNNTrainingConfig))