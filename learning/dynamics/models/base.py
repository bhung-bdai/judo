# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass, MISSING
from typing import Any

import torch
from torch import nn

import torch.nn.functional as F


ACTIVATION_MAP = {
    "linear": nn.Identity(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "log_softmax": nn.LogSoftmax(dim=-1),
    "softplus": nn.Softplus(),
}


@dataclass(kw_only=True)
class BaseModelConfig:
    """Base configuration for the neural network models."""
    name: str
    input_size: int # Number of input features
    output_size: int # Number of output features
    batch_size: int
    output_activation: str
    sequential: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseModel(nn.Module, ABC):
    """Base class for the neural network models."""
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        assert config.batch_size > 0, "Batch size must be greater than 0!"
        self.cfg = config

    @property
    def output_size(self) -> int:
        return self.cfg.output_size

    @property
    def input_size(self) -> int:
        return self.cfg.input_size

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def device(self) -> str:
        return self.cfg.device

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError

    @abstractmethod
    def get_checkpoint(self) -> dict:
        """Get the checkpoint for the model."""
        raise NotImplementedError

    @abstractmethod
    def train(self) -> Any:
        """Set the model to training mode."""
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> Any:
        """Set the model to evaluation mode."""
        raise NotImplementedError

    def load_checkpoint(self, file_path: str) -> None:
        """Load the checkpoint for the model."""
        self.load_state_dict(torch.load(file_path))

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the MSE loss"""
        return F.mse_loss(predictions, targets)


