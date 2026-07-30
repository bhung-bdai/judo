"""Visual comparison of MJSimulation (solid) vs LearnedSimulation (ghost overlay)."""

import copy
import time

import mujoco
import numpy as np
import tyro

from judo.simulation.base import Simulation
from judo.simulation.mj_simulation import MJSimulation
from judo.visualizers.visualizer import Visualizer
from judo.controller.controller import Controller, make_controller

from learning.dynamics.loader import load_dynamics_model_and_cfgs
from learning.dataset.dynamics_learning_dataset import create_dynamics_learning_dataset
from learning.dynamics.utils import loop_dataloader



EXPERIMENT_DIR = "cylinder_push_rnn/20260730_094834"


def replan_controller(controller: Controller, sim: Simulation) -> None:
    # Initialize the controller and plan
    controller.update_states(sim.sim_state)
    controller.update_action()
    # return controller.action(sim.timestep)


def copy_mj_simulation_state(src: MJSimulation, dst: MJSimulation) -> None:
    """Copy full MuJoCo simulation state from src to dst.

    Requires compatible models (same nq/nv/nu). Copies mjData, task config,
    and paused flag, then runs forward kinematics on the destination.
    """
    src_model = src.task.model
    dst_model = dst.task.model
    if (src_model.nq, src_model.nv, src_model.nu) != (dst_model.nq, dst_model.nv, dst_model.nu):
        raise ValueError(
            "Cannot copy simulation state: model dimensions mismatch "
            f"(src nq/nv/nu={src_model.nq}/{src_model.nv}/{src_model.nu}, "
            f"dst nq/nv/nu={dst_model.nq}/{dst_model.nv}/{dst_model.nu})"
        )

    mujoco.mj_copyData(dst.task.data, dst_model, src.task.data)
    mujoco.mj_forward(dst.task.sim_model, dst.task.data)
    dst.task.config = copy.deepcopy(src.task.config)
    dst.paused = src.paused


def setup_infra(task: str, experiment_dir: str) -> tuple[Simulation, Simulation, Controller, Controller, Visualizer]:
    """Sets up the simulation, visualizer, and controllers."""
    learning_task = f"{task}_learning"
    mj_controller_sim = MJSimulation(init_task=task)
    wm_controller_sim = MJSimulation(init_task=learning_task)
    copy_mj_simulation_state(mj_controller_sim, wm_controller_sim)

    wm_controller = make_controller(
        init_task=learning_task, init_optimizer="cem", rollout_backend_kwargs={"experiment_dir": experiment_dir}
    )
    mj_controller = make_controller(init_task=task, init_optimizer="cem")

    viz = Visualizer(init_task=task, sim_pause_button=False, ghost_vis=True)
    _ = np.random.default_rng(0)
    return mj_controller_sim, wm_controller_sim, wm_controller, mj_controller, viz


def visualize(
    reg_sim: Simulation,
    wm_sim: Simulation,
    viz: Visualizer,
) -> None:
    """Visualizes the simulation."""
    viz.data.xpos[:] = reg_sim.task.data.xpos
    viz.data.xquat[:] = reg_sim.task.data.xquat
    assert viz.ghost_data is not None
    viz.ghost_data.xpos[:] = wm_sim.task.data.xpos
    viz.ghost_data.xquat[:] = wm_sim.task.data.xquat
    viz.set_model_pose()


def set_sim_from_state(sim: MJSimulation, state: np.ndarray) -> None:
    """Set MuJoCo sim qpos/qvel from a flat (nq + nv,) state vector."""
    nq = sim.task.model.nq
    nv = sim.task.model.nv
    sim.task.data.qpos[:] = state[:nq]
    sim.task.data.qvel[:] = state[nq : nq + nv]
    mujoco.mj_forward(sim.task.sim_model, sim.task.data)


def expand_decimated_controls(action_horizon: np.ndarray, num_timesteps: int) -> np.ndarray:
    """Expand decimated dataset controls to sim-rate controls."""
    if action_horizon.ndim == 3:
        action_horizon = action_horizon[0]
    if action_horizon.shape[0] == num_timesteps:
        return action_horizon
    if num_timesteps % action_horizon.shape[0] != 0:
        raise ValueError(
            f"num_timesteps ({num_timesteps}) must be a multiple of "
            f"decimated action length ({action_horizon.shape[0]})"
        )
    return np.repeat(action_horizon, num_timesteps // action_horizon.shape[0], axis=0)


def pad_controls(controls: np.ndarray, num_timesteps: int) -> np.ndarray:
    """Pad or truncate controls to a fixed rollout horizon."""
    if controls.shape[0] >= num_timesteps:
        return controls[:num_timesteps]
    pad = np.repeat(controls[-1:], num_timesteps - controls.shape[0], axis=0)
    return np.concatenate([controls, pad], axis=0)


def compare_controller_states(
    mj_states: np.ndarray,
    wm_states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare rollout states from MJ vs world-model controllers.

    Args:
        mj_states: Shape (num_rollouts, T, nq + nv).
        wm_states: Shape (num_rollouts, T, nq + nv).

    Returns:
        Dict with per-rollout/per-timestep backend errors.
    """
    compare_len = min(mj_states.shape[1], wm_states.shape[1])
    mj = mj_states[:, :compare_len]
    wm = wm_states[:, :compare_len]
    backend_err = np.linalg.norm(mj - wm, axis=-1)
    return {
        "backend_err": backend_err,
        "timestep_mean_err": backend_err.mean(axis=0),
        "rollout_mean_err": backend_err.mean(axis=1),
    }


def sync_controllers(mj_controller: Controller, wm_controller: Controller, sim: Simulation) -> None:
    """Align WM controller with MJ so only the rollout backend differs."""
    state_msg = sim.sim_state
    mj_controller.update_states(state_msg)
    wm_controller.update_states(state_msg)

    wm_controller.optimizer_cfg.num_rollouts = mj_controller.optimizer_cfg.num_rollouts
    wm_controller.nominal_knots = mj_controller.nominal_knots.copy()
    wm_controller.times = mj_controller.times.copy()
    wm_controller.update_spline(wm_controller.times, wm_controller.nominal_knots)

    if hasattr(mj_controller.optimizer, "sigma") and hasattr(wm_controller.optimizer, "sigma"):
        wm_controller.optimizer.sigma = mj_controller.optimizer.sigma.copy()


def compare_rollout_states(
    mj_states: np.ndarray,
    wm_states: np.ndarray,
    state_targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compare MJ vs world-model rollout outputs (single rollout batch)."""
    mj = mj_states[0]
    wm = wm_states[0]
    compare_len = min(mj.shape[0], wm.shape[0])
    mj = mj[:compare_len]
    wm = wm[:compare_len]

    metrics: dict[str, np.ndarray] = {
        "backend_err": np.linalg.norm(mj - wm, axis=-1),
    }
    if state_targets is not None:
        targets = state_targets[:compare_len]
        metrics["mj_target_err"] = np.linalg.norm(mj - targets, axis=-1)
        metrics["wm_target_err"] = np.linalg.norm(wm - targets, axis=-1)
    return metrics


def compare_update_action(
    mj_controller: Controller,
    wm_controller: Controller,
    sim: Simulation,
    *,
    viz: Visualizer | None = None,
    animate: bool = True,
    frame_sleep: float = 0.05,
) -> list[dict[str, np.ndarray]]:
    """Mirror ``Controller.update_action`` on both controllers with shared samples.

    Samples controls once (from the MJ controller), runs MJ and world-model
    rollouts on identical inputs, and compares ``states`` after each optimization
    iteration.
    """
    sync_controllers(mj_controller, wm_controller, sim)

    mj_controller._pre_optimization()
    wm_controller._pre_optimization()
    wm_controller._nominal_knots_normalized = mj_controller._nominal_knots_normalized.copy()

    iteration_metrics: list[dict[str, np.ndarray]] = []
    opt_iter = 0
    while opt_iter < mj_controller.max_opt_iters and not mj_controller.optimizer.stop_cond():
        rollout_controls = mj_controller._sample_controls()
        wm_controller._candidate_knots_normalized = mj_controller._candidate_knots_normalized.copy()
        wm_controller.candidate_knots = mj_controller.candidate_knots.copy()
        wm_controller.rollout_controls = rollout_controls.copy()

        mj_controller._pre_rollout()
        wm_controller._pre_rollout()

        sim_controls = mj_controller.task.task_to_sim_ctrl(rollout_controls)
        x0 = mj_controller.current_state

        mj_controller.states, mj_controller.sensors, mj_policy_output = mj_controller.rollout_backend.rollout(
            x0,
            sim_controls,
            mj_controller._last_policy_output,
        )
        wm_controller.states, wm_controller.sensors, wm_policy_output = wm_controller.rollout_backend.rollout(
            x0,
            sim_controls,
            wm_controller._last_policy_output,
        )
        if mj_policy_output is not None:
            mj_controller._last_policy_output = mj_policy_output
        if wm_policy_output is not None:
            wm_controller._last_policy_output = wm_policy_output

        metrics = compare_controller_states(mj_controller.states, wm_controller.states)
        iteration_metrics.append(metrics)
        print(
            f"  opt_iter={opt_iter}  "
            f"timestep_mean_err={metrics['timestep_mean_err']}  "
            f"rollout_mean_err={metrics['rollout_mean_err']}"
        )

        mj_controller._post_rollout()
        wm_controller._post_rollout()
        wm_controller.rewards = mj_controller.rewards.copy()

        mj_controller._update_iteration()
        wm_controller._nominal_knots_normalized = mj_controller._nominal_knots_normalized.copy()
        wm_controller.action_normalizer.update(wm_controller.candidate_knots)

        opt_iter += 1

    mj_controller._post_optimization()
    wm_controller.nominal_knots = mj_controller.nominal_knots.copy()
    wm_controller.times = mj_controller.times.copy()
    wm_controller.update_spline(wm_controller.times, wm_controller.nominal_knots)
    wm_controller.update_traces()

    if viz is not None and animate and iteration_metrics:
        best_rollout = int(np.argmax(mj_controller.rewards))
        x0 = mj_controller.current_state
        compare_len = min(mj_controller.states.shape[1], wm_controller.states.shape[1])
        visualize_rollout_comparison(
            x0,
            mj_controller.states[best_rollout, :compare_len],
            wm_controller.states[best_rollout, :compare_len],
            viz,
            frame_sleep=frame_sleep,
        )

    return iteration_metrics


def rollout_states_to_traces(states: np.ndarray, site_z: float = 0.15) -> np.ndarray:
    """Convert rollout states (T, nq + nv) to viser trace line segments.

    Returns shape (2 * (T - 1), 2, 3) for pusher and cart xy paths.
    """
    if states.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float64)

    pusher_xyz = np.column_stack([states[:, 0], states[:, 1], np.full(states.shape[0], site_z)])
    cart_xyz = np.column_stack([states[:, 2], states[:, 3], np.full(states.shape[0], site_z)])
    pusher_segments = np.stack([pusher_xyz[:-1], pusher_xyz[1:]], axis=1)
    cart_segments = np.stack([cart_xyz[:-1], cart_xyz[1:]], axis=1)
    return np.concatenate([pusher_segments, cart_segments], axis=0)


def _write_state_to_mjdata(state: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    nq = model.nq
    nv = model.nv
    data.qpos[:nq] = state[:nq]
    data.qvel[:nv] = state[nq : nq + nv]
    mujoco.mj_forward(model, data)


def visualize_rollout_comparison(
    x0: np.ndarray,
    mj_states: np.ndarray,
    wm_states: np.ndarray,
    viz: Visualizer,
    *,
    frame_sleep: float = 0.05,
    site_z: float = 0.15,
) -> None:
    """Animate and trace full MJ (solid) vs world-model (ghost) rollout trajectories."""
    compare_len = min(mj_states.shape[0], wm_states.shape[0])
    mj_states = mj_states[:compare_len]
    wm_states = wm_states[:compare_len]

    mj_trajectory = np.concatenate([x0[np.newaxis, :], mj_states], axis=0)
    wm_trajectory = np.concatenate([x0[np.newaxis, :], wm_states], axis=0)

    mj_traces = rollout_states_to_traces(mj_trajectory, site_z)
    wm_traces = rollout_states_to_traces(wm_trajectory, site_z)
    combined_traces = np.concatenate([mj_traces, wm_traces], axis=0)
    viz.viser_model.set_traces(combined_traces, mj_traces.shape[0])

    model = viz.task.model
    assert viz.ghost_data is not None
    for mj_state, wm_state in zip(mj_trajectory, wm_trajectory, strict=False):
        _write_state_to_mjdata(mj_state, model, viz.data)
        _write_state_to_mjdata(wm_state, model, viz.ghost_data)
        viz.set_model_pose()
        time.sleep(frame_sleep)


def visualize_truth_and_prediction(
    sim: MJSimulation,
    predicted_state: np.ndarray,
    viz: Visualizer,
) -> None:
    """Render ground-truth sim (solid) and world-model prediction (ghost)."""
    viz.data.xpos[:] = sim.task.data.xpos
    viz.data.xquat[:] = sim.task.data.xquat
    if viz.ghost_data is not None:
        nq = sim.task.model.nq
        nv = sim.task.model.nv
        viz.ghost_data.qpos[:nq] = predicted_state[:nq]
        viz.ghost_data.qvel[:nv] = predicted_state[nq : nq + nv]
        mujoco.mj_forward(viz.task.model, viz.ghost_data)
    viz.set_model_pose()


def open_loop_check(
    task: str = "cylinder_push",
    experiment_dir: str = EXPERIMENT_DIR,
    num_rollouts: int = 8,
) -> None:
    """Feed identical x0/controls to both backends and print state divergence.

    Small error -> backend code is fine (closed-loop issues are network/OOD).
    Large error at step 1 -> convention bug. Growing error -> compounding model error.
    """
    from judo.tasks import get_registered_tasks
    from judo.utils.mj_rollout_backend import MJRolloutBackend
    from judo.utils.world_model_backend import WorldModelRolloutBackend

    judo_task = get_registered_tasks()[task].task_type()
    mj_backend = MJRolloutBackend(judo_task.model, num_rollouts)
    wm_backend = WorldModelRolloutBackend(judo_task.model, num_rollouts, experiment_dir)

    rng = np.random.default_rng(0)
    judo_task.reset()
    x0 = np.concatenate([judo_task.data.qpos, judo_task.data.qvel])
    num_timesteps = wm_backend.training_cfg.prediction_horizon_steps
    # Smooth random position targets near the current pusher position (in-distribution-ish).
    knots = x0[:2] + rng.uniform(-0.5, 0.5, (num_rollouts, 5, judo_task.nu))
    controls = np.repeat(knots, num_timesteps // 5, axis=1)

    mj_states, _, _ = mj_backend.rollout(x0, controls)
    wm_states, _, _ = wm_backend.rollout(x0, controls)

    pos_err = np.linalg.norm(mj_states[:, :, :4] - wm_states[:, :, :4], axis=-1).mean(axis=0)
    for t in range(0, num_timesteps, 5):
        print(f"t={t:3d}  mean pos err = {pos_err[t]:.4f}")
    print(f"final mean pos err = {pos_err[-1]:.4f}")


def compare_dataset_to_rollout_output(
    task: str = "cylinder_push",
    experiment_dir: str = EXPERIMENT_DIR,
    num_samples: int = 100,
    sim_steps_per_mpc_step: int = 2,
) -> None:
    """Compare MJ vs world-model rollout backends on dataset samples.

    At each replan interval both backends receive the same x0 (from a single
    MuJoCo sim) and the same control sequence. Their predicted state
    trajectories are compared, then the sim is stepped forward using the
    dataset controls as the ground-truth state propagator.
    """
    from judo.tasks import get_registered_tasks
    from judo.utils.mj_rollout_backend import MJRolloutBackend
    from judo.utils.world_model_backend import WorldModelRolloutBackend

    _, _, training_cfg, dataset_cfg = load_dynamics_model_and_cfgs(
        experiment_dir, load_supplementary_configs=True, randomize_seed=True, overwrite_batch=1
    )
    num_timesteps = training_cfg.prediction_horizon_steps

    judo_task = get_registered_tasks()[task].task_type()
    sim = MJSimulation(init_task=task)
    mj_backend = MJRolloutBackend(judo_task.model, num_threads=1)
    wm_backend = WorldModelRolloutBackend(judo_task.model, num_threads=1, experiment_dir=experiment_dir)
    viz = Visualizer(init_task=task, sim_pause_button=False, ghost_vis=True)

    _, dataloader = create_dynamics_learning_dataset(dataset_cfg)
    data_batches = loop_dataloader(dataloader)

    for sample_idx in range(num_samples):
        current_state, _, action_horizon, state_targets, _ = next(data_batches)
        current_state = current_state.cpu().numpy().reshape(-1)
        action_horizon = action_horizon.cpu().numpy()
        state_targets = state_targets.cpu().numpy()
        if state_targets.ndim == 3:
            state_targets = state_targets[0]

        controls = expand_decimated_controls(action_horizon, num_timesteps)
        set_sim_from_state(sim, current_state)

        horizon_len = min(state_targets.shape[0], num_timesteps, controls.shape[0])
        print(f"\n=== sample {sample_idx} ===")

        # for step in range(horizon_len):
        #     if step % sim_steps_per_mpc_step == 0:
        x0 = np.concatenate([sim.task.data.qpos, sim.task.data.qvel])
        remaining_controls = pad_controls(controls, num_timesteps)
        controls_batched = remaining_controls[np.newaxis, :, :]

        mj_states, _, _ = mj_backend.rollout(x0, controls_batched)
        wm_states, _, _ = wm_backend.rollout(x0, controls_batched)

        compare_len = min(mj_states.shape[1], wm_states.shape[1], horizon_len)
        metrics = compare_rollout_states(
            mj_states[:, :compare_len],
            wm_states[:, :compare_len],
            state_targets[:, :compare_len],
        )
        print(
            f"backend_err={metrics['backend_err']}  "
            f"mj_vs_target={metrics['mj_target_err']}  "
            f"wm_vs_target={metrics['wm_target_err']}"
        )
        visualize_rollout_comparison(
            x0,
            mj_states[0, :compare_len],
            wm_states[0, :compare_len],
            viz,
        )
        time.sleep(0.02)


def setup_controller_pair(
    task: str,
    experiment_dir: str,
) -> tuple[MJSimulation, Controller, Controller, Visualizer]:
    """Create MJ + world-model controller pair sharing one ground-truth sim."""
    learning_task = f"{task}_learning"
    sim = MJSimulation(init_task=task)
    mj_controller = make_controller(init_task=task, init_optimizer="cem")
    wm_controller = make_controller(
        init_task=learning_task,
        init_optimizer="cem",
        rollout_backend_kwargs={"experiment_dir": experiment_dir},
    )
    viz = Visualizer(init_task=task, sim_pause_button=False, ghost_vis=True)
    mj_controller.reset()
    sync_controllers(mj_controller, wm_controller, sim)
    return sim, mj_controller, wm_controller, viz


# def reset_sim(sim: MJSimulation) -> None:
#     theta = 2 * np.pi * np.random.rand(2)
#     self.data.qpos = np.array(
#         [
#             np.cos(theta[0]),
#             np.sin(theta[0]),
#             2 * np.cos(theta[1]),
#             2 * np.sin(theta[1]),
#         ]
#     )
#     self.data.qvel = np.zeros(4)
#     mujoco.mj_forward(self.model, self.data)



def compare_controllers(
    task: str = "cylinder_push",
    experiment_dir: str = EXPERIMENT_DIR,
    num_steps: int = 20,
    sim_steps_per_mpc_step: int = 2,
    seed: int = 0,
) -> None:
    """Compare MJ vs world-model controllers during ``update_action``.

    Uses a single MuJoCo sim as the ground-truth state propagator. At each
    replan, runs both controllers through the same optimization loop with
    identical sampled controls and prints state errors after every rollout.
    """
    np.random.seed(seed)
    sim, mj_controller, wm_controller, viz = setup_controller_pair(task, experiment_dir)

    for step in range(num_steps):
        if step % sim_steps_per_mpc_step == 0:
            print(f"\n=== mpc step {step} ===")
            compare_update_action(mj_controller, wm_controller, sim, viz=viz)

        control = mj_controller.compute(sim.task.data)
        sim.step(control)
        time.sleep(0.05)
        sim.task.reset()



def main(
    task: str = "cylinder_push",
    experiment_dir: str = EXPERIMENT_DIR,
    num_steps: int = 500,
    sim_steps_per_mpc_step: int = 2,
) -> None:
    """Step both sims with identical random commands and render them together."""
    mj_controller_sim, wm_controller_sim, wm_controller, mj_controller, viz = setup_infra(task, experiment_dir)

    for step in range(num_steps):
        if step % sim_steps_per_mpc_step == 0:
            replan_controller(mj_controller, mj_controller_sim)
            replan_controller(wm_controller, wm_controller_sim)

        mj_task_control = mj_controller.compute(mj_controller_sim.task.data)
        wm_task_control = wm_controller.compute(wm_controller_sim.task.data)

        mj_controller_sim.step(mj_task_control)
        wm_controller_sim.step(wm_task_control)

        visualize(mj_controller_sim, wm_controller_sim, viz)
        # err = np.linalg.norm(mj_sim.task.data.qpos - learned_sim.task.data.qpos)
        # if step % 50 == 0:
        #     print(f"step {step}: qpos error = {err:.4f}")
        # We will synch the actions across the controllers now
        wm_controller._nominal_knots_normalized = mj_controller._nominal_knots_normalized
        wm_controller.candidate_knots = mj_controller.candidate_knots
        wm_controller.times = mj_controller.times
        wm_controller.spline = mj_controller.spline
        # wm_controller.update_spline(wm_controller.times, wm_controller.candidate_knots)

        time.sleep(0.1)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "closed-loop": main,
            "open-loop": open_loop_check,
            "visualize-model": compare_dataset_to_rollout_output,
            "compare-controllers": compare_controllers,
        }
    )
