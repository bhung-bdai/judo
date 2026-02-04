# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.g1.g1_base import DEFAULT_JOINT_POSITIONS, STANDING_HEIGHT, G1Base, G1BaseConfig
from judo.utils.indexing import get_pos_indices, get_sensor_indices

XML_PATH = str(MODEL_PATH / "xml/g1/g1_box_carry.xml")

# Platform and goal positions
PLATFORM_START_POS = np.array([0.8, 0.0, 0.58])  # Top surface of starting platform
PLATFORM_GOAL_POS = np.array([-0.8, 0.0, 0.58])  # Top surface of goal platform
PLATFORM_HEIGHT = 0.58  # Height of platform surface

# Arm usage configuration - need both arms to carry
USE_LEFT_ARM = True
USE_RIGHT_ARM = True
USE_LEFT_WRIST = False
USE_RIGHT_WRIST = False


class Phase(Enum):
    """Defines the phases of the G1 box carry task."""

    PICK = 0  # Grasp and lift the box
    MOVE = 1  # Walk to the goal platform
    PLACE = 2  # Lower and release the box


@dataclass
class PickConfig:
    """Reward configuration for the pick phase of the G1 box carry task."""

    target_box_height: float = 0.9  # Target height to lift box to
    w_box_height: float = 150.0  # Weight for lifting box to target height
    w_box_orientation: float = 20.0  # Weight for keeping box upright
    w_two_hand_grasp: float = 20.0  # Weight for grasping with both hands
    w_hand_to_face: float = 30.0  # Weight for hands being close to box face centers
    grasp_distance_threshold: float = 0.15  # Distance threshold for considering "grasping"


@dataclass
class MoveConfig:
    """Reward configuration for the move phase of the G1 box carry task."""

    w_goal_distance: float = 100.0  # Weight for moving toward goal platform
    w_box_orientation: float = 20.0  # Weight for keeping box upright while moving
    w_box_height: float = 150.0  # Weight for maintaining box height while moving
    w_robot_orientation: float = 60.0  # Weight for robot turning toward goal (180 degrees)
    target_box_height: float = 0.9  # Target height to maintain while moving


@dataclass
class PlaceConfig:
    """Reward configuration for the place phase of the G1 box carry task."""

    target_box_height: float = PLATFORM_HEIGHT + 0.2  # Target height for placing (platform + box height)
    w_place_height: float = 100.0  # Weight for lowering box to platform
    w_goal_xy: float = 50.0  # Weight for xy alignment with goal
    w_robot_orientation: float = 60.0  # Weight for robot turning toward goal (180 degrees)
    w_box_orientation: float = 20.0  # Weight for keeping box upright


@dataclass
class GlobalConfig:
    """Global reward configuration for the G1 box carry task."""

    w_controls: float = 2.0  # Weight for control costs


@dataclass
class G1BoxCarryConfig(G1BaseConfig):
    """Config for the G1 box carrying task."""

    # Phase-specific reward weights
    pick_weights: PickConfig = field(default_factory=PickConfig)
    move_weights: MoveConfig = field(default_factory=MoveConfig)
    place_weights: PlaceConfig = field(default_factory=PlaceConfig)
    global_weights: GlobalConfig = field(default_factory=GlobalConfig)


class G1BoxCarry(G1Base):
    """Task getting G1 to pick up a box from a platform.

    Simplified version focusing on grasping and lifting the box to a target height.
    Uses both arms for grasping (USE_LEFT_ARM=True, USE_RIGHT_ARM=True).
    """

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(
            model_path=model_path,
            use_left_arm=USE_LEFT_ARM,
            use_right_arm=USE_RIGHT_ARM,
            use_left_wrist=USE_LEFT_WRIST,
            use_right_wrist=USE_RIGHT_WRIST,
        )
        self.body_pose_idx = get_pos_indices(self.model, "floating_base_joint")
        self.box_pose_idx = get_pos_indices(self.model, ["box_joint"])

        # Sensor indices
        self.box_pos_idx = get_sensor_indices(self.model, "trace_carry_box")
        self.left_palm_idx = get_sensor_indices(self.model, "trace_left_palm")
        self.right_palm_idx = get_sensor_indices(self.model, "trace_right_palm")
        self.platform_start_idx = get_sensor_indices(self.model, "platform_start_pos")
        self.platform_goal_idx = get_sensor_indices(self.model, "platform_goal_pos")

        # Box face sensor indices (left and right faces only)
        self.box_face_left_idx = get_sensor_indices(self.model, "box_face_left_pos")
        self.box_face_right_idx = get_sensor_indices(self.model, "box_face_right_pos")

        # Box orientation sensor
        self.box_z_axis_idx = get_sensor_indices(self.model, "box_z_axis")

        # Phase tracking
        self._data = mujoco.MjData(self.model)  # used for computing phase
        self.phase = Phase.PICK  # default phase

    def pre_rollout(self, curr_state: np.ndarray, config: G1BoxCarryConfig) -> None:
        """Computes the current phase of the task.

        Phase transitions:
        - PICK: Default phase, trying to grasp and lift box
        - MOVE: Box is lifted above target height, move toward goal
        - PLACE: Robot is near goal platform, lower and place box
        """
        # Update the data object with current state
        self._data.qpos[:] = curr_state[: self.model.nq]
        self._data.qvel[:] = curr_state[self.model.nq : self.model.nq + self.model.nv]
        mujoco.mj_forward(self.model, self._data)

        phase = Phase.PICK  # default phase

        # Get current box position and robot position
        box_z = curr_state[self.box_pose_idx[2]]
        robot_pos = curr_state[self.body_pose_idx[0:2]]  # xy position

        # Check if box is lifted (above target height)
        box_lifted = box_z > config.pick_weights.target_box_height - 0.1  # small tolerance

        # Check if robot is near goal platform
        goal_xy = PLATFORM_GOAL_POS[0:2]
        robot_to_goal_dist = np.linalg.norm(robot_pos - goal_xy)
        near_goal = robot_to_goal_dist < 1.0  # within 1m of goal

        # Phase transitions
        if box_lifted and not near_goal:
            phase = Phase.MOVE  # box is lifted, move to goal
        elif box_lifted and near_goal:
            phase = Phase.PLACE  # near goal, place the box

        self.phase = phase

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: G1BoxCarryConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the G1 box carry task.

        The reward function switches between 3 phases:
        * PICK: Grasp and lift the box from the starting platform
        * MOVE: Carry the box to the goal platform
        * PLACE: Lower and place the box on the goal platform

        There are also global rewards that are always applied:
        * Fallen: Penalty if robot falls
        * Controls: Penalty for control effort
        """
        batch_size = states.shape[0]

        # Note: states has shape (batch, T+1, nq+nv), sensors has shape (batch, T, nsensordata)
        # We need to align them by taking states[:, :-1] or states[:, 1:] depending on convention
        # For reward calculation, we typically use states[:, :-1] to match sensor timesteps
        qpos = states[:, :-1, : self.model.nq]  # (batch, T, nq)
        body_pos = qpos[..., self.body_pose_idx[0:3]]  # (batch, T, 3)
        box_pos = qpos[..., self.box_pose_idx[0:3]]  # (batch, T, 3)

        # Get sensor data
        box_sensor_pos = sensors[..., self.box_pos_idx]
        left_palm_pos = sensors[..., self.left_palm_idx]
        right_palm_pos = sensors[..., self.right_palm_idx]
        box_z_axis = sensors[..., self.box_z_axis_idx]

        # Common computations used across phases
        world_z_axis = np.array([0.0, 0.0, 1.0])
        z_alignment = np.sum(box_z_axis * world_z_axis[None, None], axis=-1)  # (batch, T)

        print(f"Phase: {self.phase}")
        # Phase-specific rewards
        if self.phase == Phase.PICK:
            # PICK phase: grasp and lift the box
            w_box_height = config.pick_weights.w_box_height
            w_box_orientation = config.pick_weights.w_box_orientation
            w_hand_to_face = config.pick_weights.w_hand_to_face

            # Box height reward - lift box to target height
            box_height_error = (box_pos[..., 2] - config.pick_weights.target_box_height) ** 2
            box_height_reward = -w_box_height * box_height_error

            # Box orientation reward
            box_orientation_reward = w_box_orientation * z_alignment

            # Hand-to-face reward - encourage hands to approach box faces
            box_face_left = sensors[..., self.box_face_left_idx]
            box_face_right = sensors[..., self.box_face_right_idx]
            left_hand_to_right_face = np.linalg.norm(left_palm_pos - box_face_right, axis=-1)
            right_hand_to_left_face = np.linalg.norm(right_palm_pos - box_face_left, axis=-1)
            hand_to_face_dist = left_hand_to_right_face + right_hand_to_left_face
            hand_to_face_reward = -w_hand_to_face * hand_to_face_dist

            rewards = (box_height_reward + box_orientation_reward + hand_to_face_reward).sum(axis=-1)

        elif self.phase == Phase.MOVE:
            # MOVE phase: walk toward goal while maintaining grasp and height
            w_goal_distance = config.move_weights.w_goal_distance
            w_box_orientation = config.move_weights.w_box_orientation
            w_box_height = config.move_weights.w_box_height
            w_robot_orientation = config.move_weights.w_robot_orientation

            # Goal distance reward - move robot toward goal
            goal_xy = PLATFORM_GOAL_POS[0:2]
            robot_to_goal_dist = np.linalg.norm(body_pos[..., 0:2] - goal_xy[None, None], axis=-1)
            goal_distance_reward = -w_goal_distance * robot_to_goal_dist

            # Box orientation reward - keep box upright
            box_orientation_reward = w_box_orientation * z_alignment

            # Two-hand grasp reward - maintain grasp

            # Box height maintenance - keep box at target height
            box_height_error = (box_pos[..., 2] - config.move_weights.target_box_height) ** 2
            box_height_reward = -w_box_height * box_height_error

            # Robot orientation reward - encourage turning around (180 degrees)
            # Get robot orientation from quaternion (w, x, y, z format in qpos)
            # The robot starts facing forward (+X), goal is at -X, so robot should turn 180 degrees
            body_quat = qpos[..., self.body_pose_idx[3:7]]  # (batch, T, 4) - quaternion [w, x, y, z]

            # Calculate forward direction (X-axis) from quaternion
            # For a quaternion [w, x, y, z], the rotated forward vector (originally [1, 0, 0]) is:
            # forward_x = 1 - 2(y^2 + z^2)
            y_q = body_quat[..., 2]
            z_q = body_quat[..., 3]
            forward_x = 1 - 2 * (y_q**2 + z_q**2)

            # Reward when forward_x is negative (facing -X direction, toward goal at x=-0.8)
            # forward_x = -1 means perfectly facing backward (180 degrees)
            robot_orientation_reward = w_robot_orientation * (-forward_x)

            rewards = (
                goal_distance_reward + box_orientation_reward + box_height_reward + robot_orientation_reward
            ).sum(axis=-1)

        elif self.phase == Phase.PLACE:
            # PLACE phase: lower box onto goal platform
            w_place_height = config.place_weights.w_place_height
            w_goal_xy = config.place_weights.w_goal_xy
            w_box_orientation = config.place_weights.w_box_orientation
            w_robot_orientation = config.place_weights.w_robot_orientation
            # Place height reward - lower box to platform height
            box_height_error = (box_pos[..., 2] - config.place_weights.target_box_height) ** 2
            place_height_reward = -w_place_height * box_height_error

            # Robot orientation reward - encourage turning around (180 degrees)
            # Get robot orientation from quaternion (w, x, y, z format in qpos)
            # The robot starts facing forward (+X), goal is at -X, so robot should turn 180 degrees
            body_quat = qpos[..., self.body_pose_idx[3:7]]  # (batch, T, 4) - quaternion [w, x, y, z]

            # Calculate forward direction (X-axis) from quaternion
            # For a quaternion [w, x, y, z], the rotated forward vector (originally [1, 0, 0]) is:
            # forward_x = 1 - 2(y^2 + z^2)
            y_q = body_quat[..., 2]
            z_q = body_quat[..., 3]
            forward_x = 1 - 2 * (y_q**2 + z_q**2)

            # Reward when forward_x is negative (facing -X direction, toward goal at x=-0.8)
            # forward_x = -1 means perfectly facing backward (180 degrees)
            robot_orientation_reward = w_robot_orientation * (-forward_x)

            # Goal XY alignment - align box with goal platform center
            goal_xy = PLATFORM_GOAL_POS[0:2]
            box_to_goal_xy_dist = np.linalg.norm(box_pos[..., 0:2] - goal_xy[None, None], axis=-1)
            goal_xy_reward = -w_goal_xy * box_to_goal_xy_dist

            # Box orientation reward
            box_orientation_reward = w_box_orientation * z_alignment

            rewards = (place_height_reward + goal_xy_reward + box_orientation_reward).sum(axis=-1)

        else:
            raise ValueError(f"Invalid phase: {self.phase}. Must be one of {list(Phase)}.")

        # Global rewards (applied in all phases)
        # Fallen penalty - check all states including final state
        all_qpos = states[..., : self.model.nq]  # (batch, T+1, nq)
        all_body_height = all_qpos[..., self.body_pose_idx[2]]  # (batch, T+1)
        g1_fallen_reward = -config.fall_penalty * (all_body_height <= config.fall_threshold).any(axis=-1)

        # Control cost
        base_vel = controls[..., :3]
        vel_cost = np.linalg.norm(base_vel, axis=-1).sum(-1)

        if self.use_left_arm or self.use_right_arm or self.use_left_wrist or self.use_right_wrist:
            arm_commands = controls[..., 3:]
            arm_defaults = self.default_command[3:]
            arm_deviation = arm_commands - arm_defaults
            arm_cost = np.linalg.norm(arm_deviation, axis=-1).sum(-1)
            controls_reward = -config.global_weights.w_controls * (vel_cost + arm_cost)
        else:
            controls_reward = -config.global_weights.w_controls * vel_cost

        # Combine phase rewards with global rewards
        rewards += g1_fallen_reward + controls_reward

        assert rewards.shape == (batch_size,), f"Expected shape ({batch_size},), got {rewards.shape}"
        return rewards

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and box."""
        # Box starts on first platform
        box_x = PLATFORM_START_POS[0]
        box_y = PLATFORM_START_POS[1]
        box_z = PLATFORM_START_POS[2] + 0.3  # 0.2 = box height above platform surface

        # G1 starts near the first platform
        robot_x = 0.3  # Close to first platform
        robot_y = 0.0

        # G1 reset pose: base position + orientation + joint positions + box pose
        return np.array(
            [
                robot_x,
                robot_y,
                STANDING_HEIGHT,  # Standing height
                1,
                0,
                0,
                0,  # Quaternion (upright orientation)
                *DEFAULT_JOINT_POSITIONS,  # Joint positions
                box_x,
                box_y,
                box_z,  # Box position
                1,
                0,
                0,
                0,  # Box quaternion (upright)
            ]
        )
