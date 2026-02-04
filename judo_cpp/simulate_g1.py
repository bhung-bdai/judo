#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

"""Standalone MuJoCo simulation with ONNX policy.

This script runs the ONNX policy in raw MuJoCo (without the mjlab environment),
useful for understanding deployment requirements and testing the policy independently.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime as ort


class OnnxMujocoSimulator:
    """Simulates a robot using ONNX policy in raw MuJoCo."""

    def __init__(
        self,
        onnx_path: str,
        mjcf_path: str,
        dt: float = 0.02,
        physics_dt: float = 0.005,
    ):
        """Initialize the simulator.

        Args:
            onnx_path: Path to ONNX policy file
            mjcf_path: Path to MuJoCo XML model (should include ground plane)
            dt: Control timestep (default: 0.02s = 50Hz)
            physics_dt: Physics timestep (default: 0.005s = 200Hz)
        """
        # Load ONNX model and metadata
        onnx_model = onnx.load(onnx_path)
        self.metadata = self._parse_metadata(onnx_model)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Parse metadata
        self.joint_names = self.metadata["joint_names"]
        self.action_scale = np.array(self.metadata["action_scale"], dtype=np.float32)
        self.default_joint_pos = np.array(self.metadata["default_joint_pos"], dtype=np.float32)

        print(f"[INFO] Loaded ONNX policy from: {onnx_path}")
        print(f"[INFO] Policy controls {len(self.joint_names)} joints")
        print(f"[INFO] Observation names: {self.metadata['observation_names']}")

        # Load MuJoCo model (supports both .xml and .mjb)
        print(f"[INFO] Loading model from: {mjcf_path}")
        if mjcf_path.endswith(".mjb"):
            self.model = mujoco.MjModel.from_binary_path(mjcf_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        print(f"[INFO] Loaded model: {self.model.nu} actuators, {self.model.njnt} joints")

        # Set timesteps
        self.dt = dt
        self.physics_dt = physics_dt
        self.model.opt.timestep = physics_dt
        self.n_substeps = int(dt / physics_dt)

        print(f"[INFO] Control frequency: {1 / dt:.1f} Hz")
        print(f"[INFO] Physics frequency: {1 / physics_dt:.1f} Hz")
        print(f"[INFO] Substeps per control: {self.n_substeps}")

        # Map joint/actuator names to MuJoCo IDs
        # Try with and without 'robot/' prefix (mjlab adds this)
        self.ctrl_joint_ids = []
        self.ctrl_actuator_ids = []

        for joint_name in self.joint_names:
            # Try to find joint (with and without prefix)
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}")
            if jnt_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model")
            self.ctrl_joint_ids.append(jnt_id)

            # Try to find actuator (with and without prefix)
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if act_id < 0:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{joint_name}")
            if act_id >= 0:
                self.ctrl_actuator_ids.append(act_id)

        print(f"[INFO] Mapped {len(self.ctrl_joint_ids)} joints, {len(self.ctrl_actuator_ids)} actuators")

        # Find root body (pelvis or base)
        self.root_body_id = self._find_root_body()
        print(f"[INFO] Root body ID: {self.root_body_id}")

        # State variables
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self.command_timer = 0.0
        self.resample_interval = 5.0

        # Initialize
        self.reset()

    def _parse_metadata(self, model: onnx.ModelProto) -> dict:
        """Parse metadata from ONNX model."""
        metadata = {}
        for prop in model.metadata_props:
            key = prop.key
            value = prop.value

            # Try to parse CSV values
            if "," in value:
                try:
                    metadata[key] = [float(v) for v in value.split(",")]
                except ValueError:
                    metadata[key] = value.split(",")
            else:
                metadata[key] = value

        return metadata

    def _find_root_body(self) -> int:
        """Find the root body ID (pelvis/torso)."""
        # Try common root body names (with and without robot/ prefix)
        for name in ["pelvis", "torso_link", "base_link", "torso", "base"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}")
            if body_id >= 0:
                return body_id
        # Default to body 1 (body 0 is world)
        return 1

    def reset(self):
        """Reset simulation to default state."""
        mujoco.mj_resetData(self.model, self.data)

        # Set default joint positions
        for i, jnt_id in enumerate(self.ctrl_joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = self.default_joint_pos[i]

        # Forward to compute derived quantities
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.sample_new_command()
        self.command_timer = 0.0

        print("[INFO] Simulation reset to default pose")

    def sample_new_command(self):
        """Sample a new random velocity command."""
        # Sample from velocity command ranges (matching training config)
        self.command[0] = np.random.uniform(-1.0, 1.0)  # lin_vel_x
        self.command[1] = np.random.uniform(-0.5, 0.5)  # lin_vel_y
        self.command[2] = np.random.uniform(-0.0, 0.0)  # ang_vel_z

        print(f"[CMD] New command: vx={self.command[0]:.2f}, vy={self.command[1]:.2f}, wz={self.command[2]:.2f}")

    def build_observation(self) -> np.ndarray:
        """Build observation vector for the policy.

        Observation structure (99-dim for G1):
        - base_lin_vel (3): Linear velocity in body frame
        - base_ang_vel (3): Angular velocity in body frame
        - projected_gravity (3): Gravity vector in body frame
        - joint_pos (29): Joint positions relative to default
        - joint_vel (29): Joint velocities
        - actions (29): Previous actions
        - command (3): Velocity command
        """
        # Get root body quaternion and convert to rotation matrix
        root_quat = self.data.xquat[self.root_body_id]  # [w, x, y, z]
        rot_mat = np.zeros(9, dtype=np.float64)  # MuJoCo requires float64
        mujoco.mju_quat2Mat(rot_mat, root_quat)
        rot_mat = rot_mat.reshape(3, 3).astype(np.float32)

        # Base linear velocity in body frame
        # MuJoCo stores body velocities in world frame
        base_lin_vel_w = self.data.qvel[0:3]
        base_lin_vel_b = rot_mat.T @ base_lin_vel_w

        # Base angular velocity in body frame
        base_ang_vel_w = self.data.qvel[3:6]
        base_ang_vel_b = rot_mat.T @ base_ang_vel_w

        # Projected gravity in body frame
        gravity_w = np.array([0, 0, -1], dtype=np.float32)
        projected_gravity_b = rot_mat.T @ gravity_w

        # Joint positions (relative to default)
        joint_pos = np.zeros(len(self.ctrl_joint_ids), dtype=np.float32)
        for i, jnt_id in enumerate(self.ctrl_joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            joint_pos[i] = self.data.qpos[qpos_adr] - self.default_joint_pos[i]

        # Joint velocities
        joint_vel = np.zeros(len(self.ctrl_joint_ids), dtype=np.float32)
        for i, jnt_id in enumerate(self.ctrl_joint_ids):
            qvel_adr = self.model.jnt_dofadr[jnt_id]
            joint_vel[i] = self.data.qvel[qvel_adr]

        # Concatenate all observations
        obs = np.concatenate(
            [
                base_lin_vel_b,  # 3
                base_ang_vel_b,  # 3
                projected_gravity_b,  # 3
                joint_pos,  # 29
                joint_vel,  # 29
                self.last_action,  # 29
                self.command,  # 3
            ]
        )

        return obs.astype(np.float32)

    def apply_action(self, action: np.ndarray):
        """Apply action to robot.

        The action goes through post-processing:
        target_pos = action * action_scale + default_joint_pos

        HACK: Keep arm joints (indices 15-28) at default positions.
        Only legs, torso, and waist are controlled by the policy.

        Args:
            action: Raw policy output (29-dim for G1)
        """
        # Post-process: scale and add offset (default pose)
        target_joint_pos = action * self.action_scale + self.default_joint_pos

        # HACK: Override arm joints to stay at default positions
        # Arm joints are indices 15-28 (shoulders, elbows, wrists)
        ARM_JOINT_INDICES = list(range(15, 29))
        for i in ARM_JOINT_INDICES:
            target_joint_pos[i] = self.default_joint_pos[i]

        # Set actuator controls (position targets)
        self.data.ctrl[self.ctrl_actuator_ids] = target_joint_pos

        # Store for next observation
        self.last_action = action

    def step(self):
        """Execute one control step (multiple physics steps)."""
        # Update command timer and resample if needed
        self.command_timer += self.dt
        if self.command_timer >= self.resample_interval:
            self.sample_new_command()
            self.command_timer = 0.0

        # Build observation
        obs = self.build_observation()

        # Run policy inference
        obs_input = obs.reshape(1, -1)  # Add batch dimension [1, 99]
        action = self.session.run(None, {"obs": obs_input})[0]
        action = action.flatten()  # Remove batch dimension

        # Apply action
        self.apply_action(action)

        # Step physics multiple times
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

            time.sleep(self.physics_dt)

    def run_interactive(self):
        """Run simulation with interactive MuJoCo viewer."""
        print("[INFO] Starting interactive simulation...")
        print("[INFO] Commands will be resampled every 5 seconds")
        print("[INFO] Press Ctrl+C or close viewer to exit")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 90

            while viewer.is_running():
                self.step()
                viewer.sync()


def main():
    parser = argparse.ArgumentParser(description="Simulate robot with ONNX policy in standalone MuJoCo")
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to ONNX policy file",
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        default=None,
        help="Path to MuJoCo XML file (should include ground plane)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Control timestep in seconds (default: 0.02 = 50Hz)",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.005,
        help="Physics timestep in seconds (default: 0.005 = 200Hz)",
    )

    args = parser.parse_args()

    # Find ONNX file
    if args.onnx is None:
        log_dir = Path("logs/rsl_rl/g1_velocity")
        if log_dir.exists():
            onnx_files = list(log_dir.glob("*/policy.onnx"))
            if onnx_files:
                onnx_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                args.onnx = str(onnx_files[0])
                print(f"[INFO] Auto-detected ONNX file: {args.onnx}")

    if args.onnx is None:
        raise ValueError("No ONNX file specified and could not auto-detect. Please provide --onnx path.")

    # Find MJCF file (use g1_standalone with actuators and ground)
    if args.mjcf is None:
        mjcf_candidates = [
            "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_standalone.xml",  # XML with actuators
            "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_standalone.mjb",  # Binary fallback
            "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_flat.xml",  # Flat without actuators
        ]
        for candidate in mjcf_candidates:
            if Path(candidate).exists():
                args.mjcf = candidate
                print(f"[INFO] Auto-detected model file: {args.mjcf}")
                break

    if args.mjcf is None:
        raise ValueError(
            "No MJCF file specified and could not auto-detect. Please provide --mjcf path to a scene with ground plane."
        )

    # Create simulator
    simulator = OnnxMujocoSimulator(
        onnx_path=args.onnx,
        mjcf_path=args.mjcf,
        dt=args.dt,
        physics_dt=args.physics_dt,
    )

    # Run interactive simulation
    simulator.run_interactive()


if __name__ == "__main__":
    main()
