

from dataclasses import dataclass, MISSING, field

import torch
from torch import nn
from torch.nn import functional as F

from learning.dynamics.models.base import (
    ACTIVATION_MAP,
    BaseModelConfig,
    BaseModel,
)
from learning.dynamics.tasks.base import Task
from learning.dynamics.utils import expand_action_horizon


@dataclass(kw_only=True)
class MLPConfig(BaseModelConfig):
    """Configuration for the MLP model."""
    name: str = "mlp"

    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    layer_activations: list[str] = field(default_factory=lambda: ["relu", "relu"])
    output_activation: str = "linear"
    dropouts: list[float] | None = None

    task_timestep: float = 0.02
    batch_size: int = 128
    obs_timestep_dim: int = 1


class MLP(BaseModel):
    """Basic MLP."""
    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self._dropouts = [nn.Dropout(dropout).to(self.cfg.device) for dropout in self.cfg.dropouts] if self.cfg.dropouts is not None else []
        layers = []
        if len(self.cfg.hidden_sizes) == 0:
            layers.append(nn.Linear(self.cfg.input_size, self.cfg.output_size))
            if self.cfg.output_activation not in ("linear", "none"):
                layers.append(ACTIVATION_MAP[self.cfg.output_activation])
        else:
            # Input layer
            layers.append(nn.Linear(self.cfg.input_size, self.cfg.hidden_sizes[0]))
            layers.append(ACTIVATION_MAP[self.cfg.layer_activations[0]])

            # Hidden layers
            for i in range(len(self.cfg.hidden_sizes) - 1):
                if len(self._dropouts) > 0:
                    layers.append(self._dropouts[i])
                layers.append(nn.Linear(self.cfg.hidden_sizes[i], self.cfg.hidden_sizes[i + 1]))
                layers.append(ACTIVATION_MAP[self.cfg.layer_activations[i]])

            # Output layer
            if len(self._dropouts) > 0:
                layers.append(self._dropouts[-1])
            layers.append(nn.Linear(self.cfg.hidden_sizes[-1], self.cfg.output_size))
            if self.cfg.output_activation not in ("linear", "none"):
                layers.append(ACTIVATION_MAP[self.cfg.output_activation])

        self.model = nn.Sequential(
            *layers,
        )
        self.model.to(self.cfg.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with the head activation applied to the last hidden state.

        Outputs: (batch_size, horizon_steps *output_size)
        """
        return self.model(inputs)

    def batched_forward(self, inputs: torch.Tensor, teacher_forcing: torch.Tensor | None) -> torch.Tensor:
        """Forward pass with the head activation applied to the last hidden state.

        Outputs: (batch_size, horizon_steps *output_size)
        """
        return self.model(inputs)

    def get_checkpoint(self) -> dict:
        return {
            "model": self.model.state_dict(),
        }

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def rollout_sequential(
        self,
        task: Task,
        current_state: torch.Tensor,
        action_horizon: torch.Tensor,
        prediction_horizon_steps: int,
        teacher_forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rollout the policy for a given number of steps.

        Args:
            policy: Policy model to rollout
            current_state: Current state
            action_horizon: Action horizon
            state_targets: State targets
            horizon_steps: Number of steps to rollout
            action_decimation_factor: Action decimation factor
            prediction_decimation_factor: Prediction decimation factor

        Returns:
            Predicted states (batch_size, prediction_horizon_steps, state_size)
        """
        action_horizon = expand_action_horizon(action_horizon, prediction_horizon_steps)
        predicted_states = torch.zeros(
            current_state.shape[0], prediction_horizon_steps, self.output_size, device=self.device
        )
        t = 0
        for t in range(prediction_horizon_steps):
            # Decimate the action horizon if we are recording too many at a time
            inputs = task.project_state(
                current_state, action_horizon[:, t, :].unsqueeze(1), device=self.device
            )
            predictions = self.forward(inputs)
            predicted_states[:, t, :] = task.lift_state(current_state, predictions, 1)[:, 0, :]
            if teacher_forcing is not None:
                current_state = teacher_forcing[:, t, :]
            else:
                current_state = predicted_states[:, t, :]
        return predicted_states


    def rollout_batched(
        self,
        task: Task,
        current_state: torch.Tensor,
        action_horizon: torch.Tensor,
        prediction_horizon_steps: int,
        teacher_forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rollout the policy for a given number of steps in a single batch.
        
        Args:
            policy: Policy model to rollout
            current_state: Current state
            action_horizon: Action horizon
            state_targets: State targets
            horizon_steps: Number of steps to rollout
            action_decimation_factor: Action decimation factor
            prediction_decimation_factor: Prediction decimation factor

        Returns:
            Predicted states (batch_size, prediction_horizon_steps, state_size)
        """
        inputs = task.project_state(current_state, action_horizon, device=self.device)
        predictions = self.batched_forward(inputs, teacher_forcing)
        predicted_states = task.lift_state(current_state, predictions, prediction_horizon_steps)
        return predicted_states
