
import numpy as np
import torch

from learning.dynamics.tasks.base import Task
from learning.dynamics.utils import flatten_last_two_dims, unflatten_last_two_dims

class CylinderPushTask(Task):
    world_frame_state_size: int = 8
    robot_frame_state_size: int = 6
    world_frame_sensor_size: int = 6
    action_size: int = 2
    qpos: slice = slice(0, 4)
    qvel: slice = slice(4, 8)
    pusher_pos: slice = slice(0, 2)
    cart_pos: slice = slice(2, 4)
    pusher_vel: slice = slice(4, 6)
    cart_vel: slice = slice(6, 8)

    @staticmethod
    def project_state(current_state: torch.Tensor, action_horizon: torch.Tensor, device: str) -> torch.Tensor:
        """Project the state to the robot frame.

        State layout (8D): [robot_x, robot_y, object_x, object_y, robot_vx, robot_vy, object_vx, object_vy]
        """
        robot_xy = current_state[:, :2].to(device)
        projected_current_state = current_state[:, 2:].clone().to(device)
        projected_current_state = torch.cat(
            [
                projected_current_state[:, :2] - robot_xy,
                projected_current_state[:, 2:],
            ],
            dim=1,
        )
        projected_action_horizon = action_horizon.to(device) - robot_xy.unsqueeze(1)
        return torch.cat(
            [
                projected_current_state,
                flatten_last_two_dims(projected_action_horizon),
            ],
            dim=1,
        )

    @staticmethod
    def lift_state(current_state: torch.Tensor, pred_states: torch.Tensor, horizon_steps: int) -> torch.Tensor:
        """Lift the state to the robot frame.

        Args:
            current_state: (batch_size, state_size)
            pred_states: (batch_size, horizon_ts * state_size)

        Pred states should be (robot_xy_offset, object_xy_offset, robot_vel, object_vel) in the
        robot-position anchor frame. Velocities are world-frame and must be signed.

        Outputs: (batch_size, horizon_steps, state_size)
        """
        pred_states = unflatten_last_two_dims(pred_states, horizon_steps, CylinderPushTask.world_frame_state_size)
        robot_xy = current_state[:, :2].unsqueeze(1)
        return torch.cat(
            [
                pred_states[:, :, :2] + robot_xy,
                pred_states[:, :, 2:4] + robot_xy,
                pred_states[:, :, 4:],
            ],
            dim=-1,
        )

    @staticmethod
    def generate_sensor_data(current_state_world_frame: np.ndarray) -> np.ndarray:
        """Generate sensor data from the current state.

        Args:
            current_state_world_frame: (batch_size, horizon_steps, task_world_frame_state_size)

        Returns:
            (batch_size, horizon_steps, task_world_frame_sensor_size)
        """
        batch_size = current_state_world_frame.shape[0]
        horizon_steps = current_state_world_frame.shape[1]
        pusher_pos = current_state_world_frame[..., CylinderPushTask.pusher_pos]
        cart_pos = current_state_world_frame[..., CylinderPushTask.cart_pos]
        sensor_data_world_frame = np.concatenate(
            [
                pusher_pos,
                np.full((batch_size, horizon_steps, 1), 0.15, dtype=np.float32),
                cart_pos,
                np.full((batch_size, horizon_steps, 1), 0.15, dtype=np.float32),
            ],
            axis=-1,
        )
        return sensor_data_world_frame
