"""Evaluation script for a learned simulation."""


import tyro

from learning.dynamics.models import load_model
from learning.dynamics.experiments.utils import EvalConfig, TrainingConfig
from learning.dataset.dynamics_learning_dataset import HDF5Dataset, HDF5DatasetConfig, create_dynamics_learning_dataset
from learning.dynamics.utils import loop_dataloader
from learning.dynamics.tasks import resolve_task_name
from learning.dynamics.loader import load_dynamics_model_and_cfgs


def eval(eval_cfg: EvalConfig) -> None:
    """Evaluate a learned simulation."""
    model, _, training_cfg, dataset_cfg = load_dynamics_model_and_cfgs(
        eval_cfg.experiment_dir, load_supplementary_configs=True, randomize_seed=True, overwrite_batch=eval_cfg.eval_batch
    )
    model.eval()
    # dataset_cfg.batch_size = eval_cfg.eval_batch
    _, dataloader = create_dynamics_learning_dataset(dataset_cfg)
    data_batches = loop_dataloader(dataloader)
    current_state, current_sensors, action_horizon, state_targets, sensor_targets = next(data_batches)
    current_state = current_state.to(training_cfg.device)
    action_horizon = action_horizon.to(training_cfg.device)
    state_targets = state_targets.to(training_cfg.device)
    sensor_targets = sensor_targets.to(training_cfg.device)
    print(f"Current state shape: {current_state.shape}, Action horizon shape: {action_horizon.shape}, State targets shape: {state_targets.shape}")
    task = resolve_task_name(training_cfg.task)
    if training_cfg.sequential:
        predicted_states = model.rollout_sequential(
            task,
            current_state,
            current_sensors,
            action_horizon,
            state_targets.shape[1],
            teacher_forcing=None if training_cfg.teacher_forcing else state_targets
        )
    else:
        predicted_states = model.rollout_batched(
            task,
            current_state,
            action_horizon,
            state_targets.shape[1],
            teacher_forcing=None if training_cfg.teacher_forcing else state_targets
        )
    print(predicted_states.shape)
    loss = model.compute_loss(predicted_states, state_targets)
    # print(predicted_states[0], )
    print(f"Loss across this batch: {loss.item()}")

    
if __name__ == "__main__":
    tyro.cli(eval)