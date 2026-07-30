

from dataclasses import dataclass, MISSING

import torch
from torch import nn

from learning.dynamics.models.base import BaseModelConfig, BaseModel
from learning.dynamics.tasks.base import Task
from learning.dynamics.utils import expand_action_horizon


@dataclass(kw_only=True)
class RNNConfig(BaseModelConfig):
    """Configuration for the RNN model."""
    gru_hidden_size: int
    gru_num_layers: int
    # Flattened project_state used once to initialize hidden state (batched mode).
    init_input_size: int
    # Per-timestep GRU input: projected state + action at one step.
    step_input_size: int
    name: str = "rnn"



class RNN(BaseModel):
    """RNN using GRU units with teacher forcing for batched training."""

    def __init__(self, config: RNNConfig):
        super().__init__(config)
        self.obs_to_hidden = nn.Linear(
            self.cfg.init_input_size,
            self.cfg.gru_num_layers * self.cfg.gru_hidden_size,
            device=self.cfg.device,
        )

        self.gru = nn.GRU(
            input_size=self.cfg.step_input_size,
            hidden_size=self.cfg.gru_hidden_size,
            num_layers=self.cfg.gru_num_layers,
            device=self.cfg.device,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(self.cfg.gru_hidden_size, self.cfg.output_size, device=self.cfg.device)

        self.layers = [
            self.obs_to_hidden,
            self.gru,
            self.head,
        ]

    def init_dec_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        return self.obs_to_hidden(inputs).view(
            batch_size, self.cfg.gru_num_layers, self.cfg.gru_hidden_size
        ).transpose(0, 1).contiguous()

    def _build_dec_inputs(
        self,
        current_state: torch.Tensor,
        teacher_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Shifted teacher forcing: [s0, s1, ..., s_{H-1}] -> predict [s1, ..., sH]."""
        dec_inputs = torch.zeros_like(teacher_forcing)
        dec_inputs[:, 0, :] = current_state
        if teacher_forcing.shape[1] > 1:
            dec_inputs[:, 1:, :] = teacher_forcing[:, :-1, :]
        return dec_inputs

    def batched_forward(
        self,
        inputs: torch.Tensor,
        teacher_forcing: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the full horizon in one GRU unroll."""
        dec_hidden = self.init_dec_hidden(inputs)
        dec_inputs = self._build_dec_inputs(current_state, teacher_forcing)
        dec_outs, _ = self.gru(dec_inputs, dec_hidden)
        preds = self.head(dec_outs)
        return preds.reshape(preds.shape[0], -1)

    def forward(
        self,
        inputs: torch.Tensor,
        dec_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one timestep. Returns (prediction, updated_hidden)."""
        if dec_hidden is None:
            dec_hidden = self.init_dec_hidden(inputs)
        dec_out, dec_hidden = self.gru(inputs.unsqueeze(1), dec_hidden)
        return self.head(dec_out[:, 0, :]), dec_hidden

    def get_checkpoint(self) -> dict:
        return {
            "obs_to_hidden": self.obs_to_hidden.state_dict(),
            "gru": self.gru.state_dict(),
            "head": self.head.state_dict(),
        }

    def train(self) -> None:
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()

    def rollout_sequential(
        self,
        task: Task,
        current_state: torch.Tensor,
        # current_sensors: torch.Tensor,
        action_horizon: torch.Tensor,
        prediction_horizon_steps: int,
        teacher_forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rollout the policy one step at a time, carrying GRU hidden state forward."""
        action_horizon = expand_action_horizon(action_horizon, prediction_horizon_steps)
        predicted_states = torch.zeros(
            current_state.shape[0],
            prediction_horizon_steps,
            self.output_size,
            device=self.device,
        )
        # predicted_sensors = torch.zeros(
        #     current_sensors.shape[0],
        #     prediction_horizon_steps,
        #     current_sensors.shape[1],
        #     device=self.device,
        # )
        dec_hidden = None
        for t in range(prediction_horizon_steps):
            inputs = task.project_state(
                current_state, action_horizon[:, t, :].unsqueeze(1), device=self.device
            )
            predictions, dec_hidden = self.forward(inputs, dec_hidden)
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
        """Rollout the full horizon in one batched GRU unroll."""
        inputs = task.project_state(current_state, action_horizon, device=self.device)
        predictions = self.batched_forward(inputs, teacher_forcing, current_state)
        predicted_states = task.lift_state(current_state, predictions, prediction_horizon_steps)
        return predicted_states
