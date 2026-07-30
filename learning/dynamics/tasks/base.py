from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import torch

class Task(ABC):
    @staticmethod
    @abstractmethod
    def project_state(
        current_state: torch.Tensor,
        action_horizon: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Project the state to the robot frame."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def lift_state(
        current_state: torch.Tensor,
        pred_states: torch.Tensor,
        horizon_steps: int,
    ) -> torch.Tensor:
        """Lift the state to the robot frame."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_sensor_data(
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray:
        """Project the sensor to the robot frame."""
        raise NotImplementedError
