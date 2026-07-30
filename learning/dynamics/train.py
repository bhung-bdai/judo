# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Training script for diffusion policy with vanilla PyTorch."""

import time
from pathlib import Path
import torch
from tqdm import tqdm

from learning.dynamics.utils import get_judo_path
from learning.dataset.dynamics_learning_dataset import (
    create_dynamics_learning_dataset,
    HDF5DatasetConfig,
)
from learning.dynamics.models import BaseModel

from learning.dynamics.experiments.utils import TrainingConfig, save_checkpoint, setup_experiment
from learning.dynamics.utils import (
    loop_dataloader,
    get_judo_path,
)


def train(
    policy: BaseModel,
    training_cfg: TrainingConfig,
    dataset_cfg: HDF5DatasetConfig,
    experiment_dir: Path | None = None,
) -> None:
    """Vanilla PyTorch training loop using model's training_step() method.

    Args:
        policy: Policy model to train (handles its own optimizer)
        training_cfg: Configuration object
    """
    _, dataloader = create_dynamics_learning_dataset(dataset_cfg)
    data_batches = loop_dataloader(dataloader)
    min_loss: float = float("inf")
    min_policy: Path | None = None

    n_gradient_step = 0
    save_counter = 0
    log_counter = 0

    loss_list = torch.zeros(training_cfg.log_freq, device="cpu")
    start_time = time.time()
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_cfg.lr)

    # Create progress bar
    pbar = tqdm(range(training_cfg.gradient_steps), desc="Training", unit="step")
    policy.train()

    for _step in pbar:
        # Current_state: (batch_size, state_size)
        # Action_horizon: (batch_size, horizon_ts, action_size)
        # State_targets: (batch_size, horizon_ts, state_size)
        current_state, _, action_horizon, state_targets, _ = next(data_batches)
        current_state = current_state.to(training_cfg.device)
        # current_sensors = current_sensors.to(training_cfg.device)
        action_horizon = action_horizon.to(training_cfg.device)
        state_targets = state_targets.to(training_cfg.device)
        if training_cfg.sequential:
            predicted_states = policy.rollout_sequential(
                training_cfg.task,
                current_state, 
                action_horizon, 
                state_targets.shape[1],
                teacher_forcing=None if not training_cfg.teacher_forcing else state_targets
            )
        else:
            predicted_states = policy.rollout_batched(
                training_cfg.task,
                current_state,
                action_horizon,
                state_targets.shape[1],
                teacher_forcing=None if not training_cfg.teacher_forcing else state_targets
            )
        loss = policy.compute_loss(predicted_states, state_targets)

        # Update the policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=training_cfg.max_grad_norm)
        optimizer.step()
        loss_list[log_counter] = loss.item()

       # Increment step
        n_gradient_step += 1
        save_counter += 1
        log_counter += 1

        # Log training metrics
        if log_counter == training_cfg.log_freq:
            avg_loss = loss_list.mean().item()
            # Update progress bar with average loss
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
            log_counter = 0

        # Save checkpoint
        if save_counter == training_cfg.save_freq:
            if experiment_dir is not None:
                checkpoint_path = save_checkpoint(experiment_dir, policy, n_gradient_step)
            else:
                checkpoint_path = Path(
                    f"{get_judo_path()}/learning/dynamics/models/world_model_{policy.name}_{n_gradient_step}.pth"
                )
                torch.save(policy.state_dict(), str(checkpoint_path))
            save_counter = 0
            if loss_list.mean().item() < min_loss:
                min_loss = loss_list.mean().item()
                min_policy = checkpoint_path

        # Finish
        if n_gradient_step > training_cfg.gradient_steps:
            break

    if min_policy is not None:
        best_path = min_policy.with_name(f"best_model.pth")
        min_policy.rename(best_path)
        # TODO: replace this with a logger
        print(f"Best policy {min_policy} saved to {best_path}")

    pbar.close()
    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")


def run_experiment(model: BaseModel, training_cfg: TrainingConfig, experiment_name: str, timestamped: bool = False) -> None:
    """Create an experiment directory, save configs, and train the model.

    Args:
        model: The model to train.
        training_cfg: The training configuration.
        experiment_name: The name of the experiment.

    Returns:
        None
    """
    dataset_cfg = HDF5DatasetConfig(
        file_path=f"{get_judo_path()}/{training_cfg.dataset_file_path}",
        obs_timestep_dim=training_cfg.obs_timestep_dim,
        batch_size=training_cfg.batch_size,
        horizon=training_cfg.horizon,
        action_decimation_factor=training_cfg.action_decimation_factor,
        prediction_decimation_factor=training_cfg.prediction_decimation_factor,
    )
    experiment_dir = setup_experiment(
        experiment_name, model=model, training_cfg=training_cfg, dataset_cfg=dataset_cfg, timestamped=timestamped
    )
    train(model, training_cfg, dataset_cfg, experiment_dir=experiment_dir)
