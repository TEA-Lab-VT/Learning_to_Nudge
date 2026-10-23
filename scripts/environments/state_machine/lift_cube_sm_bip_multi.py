# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generic DMP/BIP orchestration loop for IsaacLab manipulation tasks.
Subtask sequences and descriptions are defined in task_configs/<task>.yaml.

The "reach" subtask is driven by a Bayesian Interaction Primitive (BIP) fit to
recorded demonstration trajectories: the prior is conditioned on the current
start/goal positions (Gaussian conditioning at phase 0 and phase 1) and the
resulting trajectory is played back open-loop. "carry" and "release" still use
DMP weights generated per subtask via DMPGenerator (LLM).

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm_bip_multi.py --num_envs 1
    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm_bip_multi.py --task lift_cube --num_envs 1
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generic DMP orchestration loop for IsaacLab.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="lift_cube", help="Task name (matches task_configs/<name>.yaml)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import os
import sys
import glob
import yaml
import unittest.mock

import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as R_scipy

import importlib.util

_INTPRIM_SRC = "/home/jhaixin/intprim"
sys.path.insert(0, _INTPRIM_SRC)
import intprim

_LMP_SRC = "/home/jhaixin/lmp-26.github.io/src"
sys.path.insert(0, _LMP_SRC)

_spec = importlib.util.spec_from_file_location(
    "lmp_controller", f"{_LMP_SRC}/controller.py"
)
_lmp_controller_mod = importlib.util.module_from_spec(_spec)
with unittest.mock.patch.dict("sys.modules", {
    "pygame": unittest.mock.MagicMock(),
    "pynput": unittest.mock.MagicMock(),
    "pynput.keyboard": unittest.mock.MagicMock(),
    "utils": unittest.mock.MagicMock(),
}):
    _spec.loader.exec_module(_lmp_controller_mod)

DMP = _lmp_controller_mod.DMP

from dotenv import load_dotenv
load_dotenv("/home/jhaixin/lmp-26.github.io/src/lmp/.env")

_gen_spec = importlib.util.spec_from_file_location(
    "lmp_dmp_generator", "/home/jhaixin/lmp-26.github.io/src/lmp/llm_dmp_generator.py"
)
_gen_mod = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen_mod)
DMPGenerator = _gen_mod.DMPGenerator

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

N_FUNCTIONS = 11
N_DOF = 5
STEP_SIZE = 0.002
DONE_THRESHOLD = 0.0112

_TASK_LIBRARY = "/home/jhaixin/lmp-26.github.io/src/lmp/prompt/task_library"
_TASK_CONFIGS  = "/home/jhaixin/lmp-26.github.io/src/lmp/prompt/task_configs"

# ---------------------------------------------------------------------------
# BIP config — "reach" only. carry/release keep using DMPGenerator (LLM).
# ---------------------------------------------------------------------------
BIP_DEMO_DIR = "/home/jhaixin/IsaacLab/recorded_trajectories"
BIP_NUM_DOF = 3  # x, y, z only — yaw is always 0 in this task, gripper handled separately
BIP_NUM_FUNCTIONS = 8
BIP_BASIS_VARIANCE = 0.1
BIP_NUM_SAMPLES = 200  # density of the rendered reference trajectory, independent of sim step count
BIP_BOUNDARY_NOISE = np.diag([1e-4, 1e-4, 1e-4])  # how confident we are in the start/goal points
BIP_STATIONARY_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _read_ee(scene):
    return (
        scene["ee_frame"].data.target_pos_w[..., 0, :] - scene.env_origins
    )[0].cpu().numpy()


def _read_obj(scene, object_name):
    return (
        scene[object_name].data.root_pos_w - scene.env_origins
    )[0].cpu().numpy()


def _read_target(env):
    return env.unwrapped.command_manager.get_command("object_pose")[0, :3].cpu().numpy()


# ---------------------------------------------------------------------------
# Goal resolvers — add new subtask types here
# Each returns a 5-D numpy array: [x, y, z, yaw, gripper]
# gripper: 1.0 = close, 0.0 = open
# ---------------------------------------------------------------------------

def _reach_goal(scene, obj, env, subtask_cfg):
    pos = _read_obj(scene, obj)[:3].copy()
    reach_height = subtask_cfg.get("reach_height", None)
    if reach_height is not None:
        pos[2] = reach_height
    else:
        pos[2] += subtask_cfg.get("z_offset", 0.0)
    return np.array([*pos, 0.0, 0.0])  # gripper OPEN during approach

def _carry_goal(scene, obj, env, subtask_cfg):
    target = _read_target(env)[:3].copy()
    carry_height = subtask_cfg.get("carry_height", None)
    if carry_height is not None:
        target[2] = carry_height
    return np.array([*target, 0.0, 1.0])

def _release_goal(scene, obj, env, subtask_cfg):
    return np.array([*_read_ee(scene)[:3], 0.0, 0.0])

GOAL_RESOLVERS = {
    "reach":   _reach_goal,
    "carry":   _carry_goal,
    "release": _release_goal,
}


# ---------------------------------------------------------------------------
# BIP helpers ("reach" only)
#
# The prior is built once from recorded demonstrations, then conditioned on
# this trial's actual start (phase 0) and goal (phase 1) via the standard
# ProMP/IP Gaussian conditioning update — no online filter/localization is
# needed since both boundary points are known exactly before the phase starts.
# ---------------------------------------------------------------------------

def _remove_stationary_frames(trajectory, threshold=BIP_STATIONARY_THRESHOLD):
    """Collapse held/paused frames so phase reflects motion progress, not wall-clock samples."""
    keep_indices = [0]
    for idx in range(1, trajectory.shape[1]):
        if np.linalg.norm(trajectory[:, idx] - trajectory[:, keep_indices[-1]]) > threshold:
            keep_indices.append(idx)
    if keep_indices[-1] != trajectory.shape[1] - 1:
        keep_indices.append(trajectory.shape[1] - 1)
    return trajectory[:, keep_indices]


def _condition_on_point(basis_model, weight_mean, weight_cov, phase, dof_indices, y_star, observation_noise):
    """Standard ProMP/IP Gaussian conditioning: force the trajectory through y_star at phase."""
    basis_at_phase = basis_model.get_block_diagonal_basis_matrix(np.array([phase]))
    H = basis_at_phase.T[dof_indices, :]

    innovation_cov = H.dot(weight_cov).dot(H.T) + observation_noise
    kalman_gain = weight_cov.dot(H.T).dot(np.linalg.inv(innovation_cov))

    conditioned_mean = weight_mean + kalman_gain.dot(y_star - H.dot(weight_mean))
    conditioned_cov = weight_cov - kalman_gain.dot(H).dot(weight_cov)

    return conditioned_mean, conditioned_cov


def build_bip_reach_model(data_dir: str = BIP_DEMO_DIR):
    """Fit a BIP prior to all recorded reach demonstrations in data_dir.

    Returns (basis_model, primitive, prior_mean, prior_cov).
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No recorded trajectories found in {data_dir}")

    dof_names = np.array(["x", "y", "z"])
    basis_model = intprim.basis.GaussianModel(BIP_NUM_FUNCTIONS, BIP_BASIS_VARIANCE, dof_names)
    primitive = intprim.BayesianInteractionPrimitive(basis_model)

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        trajectory = _remove_stationary_frames(data["trajectory"][:BIP_NUM_DOF, :])
        primitive.add_demonstration(trajectory)

    prior_mean, prior_cov = primitive.get_basis_weight_parameters()
    print(f"[BIP] Fit reach prior from {len(npz_files)} demonstrations in {data_dir}")

    return basis_model, primitive, prior_mean, prior_cov


def render_reach_trajectory(basis_model, primitive, prior_mean, prior_cov,
                             start_xyz: np.ndarray, goal_xyz: np.ndarray,
                             num_samples: int = BIP_NUM_SAMPLES) -> np.ndarray:
    """Condition the BIP prior on this trial's start/goal and render the full trajectory.

    Returns a (3, num_samples) array covering phase 0 -> 1.
    """
    mean, cov = _condition_on_point(
        basis_model, prior_mean, prior_cov, 0.0, [0, 1, 2], start_xyz, BIP_BOUNDARY_NOISE
    )
    mean, cov = _condition_on_point(
        basis_model, mean, cov, 1.0, [0, 1, 2], goal_xyz, BIP_BOUNDARY_NOISE
    )

    domain = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    return primitive.basis_inverse_transform(domain, mean)


# ---------------------------------------------------------------------------
# DMP helpers
# ---------------------------------------------------------------------------

def build_dmps(weights: np.ndarray) -> list:
    dmps = [DMP(basis_type="gaussian") for _ in range(3)]
    dmps.append(DMP(basis_type="gaussian"))
    dmps.append(DMP(basis_type="step", dmp_type="gripper"))
    for i, dmp in enumerate(dmps):
        dmp.set_weights(weights[i])
        dmp.set_step(STEP_SIZE)
    return dmps


def xi_to_action(xi: np.ndarray, device: str) -> torch.Tensor:
    x, y, z, yaw, grip = xi
    r = R_scipy.from_euler("x", np.pi) * R_scipy.from_euler("z", yaw)
    q = r.as_quat()
    qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])
    gripper = -1.0 if grip > 0.5 else 1.0
    return torch.tensor([[x, y, z, qw, qx, qy, qz, gripper]], dtype=torch.float32, device=device)


def run_dmp_phase(dmps, env, scene, label: str, goal_xyz: np.ndarray,
                  prev_action=None, pos_threshold: float = 0.012):
    """Run one DMP phase, stopping as soon as the EE reaches goal_xyz.

    Termination conditions (whichever comes first):
      1. EE is within pos_threshold of goal_xyz  ← primary: distance-based
      2. DMP canonical system expires (DONE_THRESHOLD)  ← fallback

    Returns (last_action, terminated_early).
    """
    if prev_action is not None:
        actions = prev_action.clone()
    else:
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0  # identity quaternion, gripper open

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            if dmps[0].x > DONE_THRESHOLD:
                xi = np.array([dmp.step() for dmp in dmps])
                actions = xi_to_action(xi, env.unwrapped.device)

                # Distance-based early exit: switch phase as soon as EE arrives
                ee_pos = _read_ee(scene)
                dist = float(np.linalg.norm(ee_pos - goal_xyz))
                if dist < pos_threshold:
                    print(f"[{label}] reached goal at step {step} (dist={dist:.4f}), switching.")
                    return actions, False

                if step % 50 == 0:
                    print(
                        f"[{label}] step {step:4d} | "
                        f"pos=[{xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f}] | "
                        f"dist={dist:.4f}"
                    )
            else:
                print(f"[{label}] DMP expired after {step} steps (fallback).")
                return actions, False

            if dones.any():
                return actions, True

            step += 1

    return actions, True


def run_bip_phase(bip_trajectory: np.ndarray, gripper_dmp, env, scene, label: str, goal_xyz: np.ndarray,
                   prev_action=None, pos_threshold: float = 0.012):
    """Run one BIP phase, playing back a precomputed (3, N) reach trajectory open-loop.

    bip_trajectory covers phase 0 -> 1 densely (BIP_NUM_SAMPLES points); a throwaway
    DMP is used purely as a phase/timing pacer (its own xi output is discarded) so
    that timing and the DONE_THRESHOLD fallback stay consistent with the DMP phases.
    yaw is hardcoded to 0.0 since every goal in this task has yaw = 0.

    Termination conditions mirror run_dmp_phase: distance-based early exit, or
    pacer phase expiry as a fallback.
    """
    pacer = DMP(basis_type="gaussian")
    pacer.set_weights(np.zeros(N_FUNCTIONS))
    pacer.set_step(STEP_SIZE)
    pacer.set_start(0.0)
    pacer.set_goal(0.0)

    domain = np.linspace(0.0, 1.0, bip_trajectory.shape[1])

    if prev_action is not None:
        actions = prev_action.clone()
    else:
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0  # identity quaternion, gripper open

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            if pacer.x > DONE_THRESHOLD:
                pacer.step()
                progress = min(pacer.get_progress(), 1.0)
                xyz = np.array([np.interp(progress, domain, bip_trajectory[i]) for i in range(3)])
                grip = gripper_dmp.step()
                xi = np.array([xyz[0], xyz[1], xyz[2], 0.0, grip])
                actions = xi_to_action(xi, env.unwrapped.device)

                # Distance-based early exit: switch phase as soon as EE arrives
                ee_pos = _read_ee(scene)
                dist = float(np.linalg.norm(ee_pos - goal_xyz))
                if dist < pos_threshold:
                    print(f"[{label}] reached goal at step {step} (dist={dist:.4f}), switching.")
                    return actions, False

                if step % 50 == 0:
                    print(
                        f"[{label}] step {step:4d} | "
                        f"pos=[{xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f}] | "
                        f"dist={dist:.4f}"
                    )
            else:
                print(f"[{label}] BIP phase expired after {step} steps (fallback).")
                return actions, False

            if dones.any():
                return actions, True

            step += 1

    return actions, True


def hold_and_grasp(env, prev_action, steps: int = 150):
    """Hold EE position and close gripper for N steps.

    Called after reach finishes so the gripper can firmly close around the
    object before the carry phase begins (mirrors the real robot's sleep(2.0)).
    """
    actions = prev_action.clone()
    actions[:, 7] = -1.0  # close gripper, keep EE position/orientation
    for _ in range(steps):
        with torch.inference_mode():
            dones = env.step(actions)[-2]
            if dones.any():
                return actions, True
    print(f"[grasp] closed gripper for {steps} steps.")
    return actions, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    task_name = args_cli.task
    task_cfg_path = os.path.join(_TASK_CONFIGS, f"{task_name}.yaml")
    with open(task_cfg_path, "r") as f:
        task_cfg = yaml.safe_load(f)

    object_name = task_cfg["object"]           # human-readable name for LLM
    scene_key   = task_cfg.get("scene_key", object_name)  # IsaacLab scene entity key
    subtasks    = task_cfg["subtasks"]
    print(f"Task     : {task_name}")
    print(f"Object   : {object_name} (scene key: {scene_key})")
    print(f"Subtasks : {[s['name'] for s in subtasks]}")

    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    #set position of the cube to be fixed in the scene, so that the cube always appears at the same default location (makes weight experiments reproducible).
    env_cfg.events.randomize_objects = None

    env_cfg.episode_length_s = 60.0  # enough for all subtasks to finish
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene

    
    obstacle_keys = task_cfg.get("obstacle_scene_keys", [])
    # obstacle_positions = {k: _read_obj(scene, k).tolist() for k in obstacle_keys}
    OBSTACLE_HALF_HEIGHT = task_cfg.get("obstacle_half_height", 0.0)
    obstacle_positions = {
        k: (_read_obj(scene, k) + np.array([0, 0, OBSTACLE_HALF_HEIGHT])).tolist()
        for k in obstacle_keys
    }
    if obstacle_positions:
        print(f"Obstacles: {list(obstacle_positions.keys())}")

    # --- Pre-phase: move EE to home position before starting subtasks ---
    HOME_GOAL = np.array([0.35, 0.0, 0.10, 0.0, 0.0])  # [x, y, z, yaw, gripper=open]
    print(f"\n=== Pre-phase: moving EE to home {HOME_GOAL[:3]} ===")
    ee_pos = _read_ee(scene)
    start_home = np.array([ee_pos[0], ee_pos[1], ee_pos[2], 0.0, 0.0])
    home_weights = np.zeros((N_DOF, N_FUNCTIONS))
    home_dmps = build_dmps(home_weights)
    for i, dmp in enumerate(home_dmps):
        dmp.set_start(start_home[i])
        dmp.set_goal(HOME_GOAL[i])
    last_action, terminated = run_dmp_phase(
        home_dmps, env, scene,
        label="home",
        goal_xyz=HOME_GOAL[:3],
        prev_action=None,
    )
    if terminated:
        print("Environment terminated during home phase, stopping.")
        env.close()
        return
    print("Home position reached.")
    # --- End pre-phase ---

    bip_reach_model = None  # (basis_model, primitive, prior_mean, prior_cov), built lazily on first use

    for subtask in subtasks:
        subtask_name = subtask["name"]
        subtask_desc = subtask["description"].format(object=object_name)
        task_file    = os.path.join(_TASK_LIBRARY, f"{subtask_name}.yaml")

        print(f"\n=== Subtask: [{subtask_name}] \"{subtask_desc}\" ===")

        ee_pos      = _read_ee(scene)
        goal        = GOAL_RESOLVERS[subtask_name](scene, scene_key, env, subtask)

        print(f"  EE start : {ee_pos}")
        print(f"  Goal     : {goal}")

        if subtask_name == "reach":
            if bip_reach_model is None:
                bip_reach_model = build_bip_reach_model()
            basis_model, primitive, prior_mean, prior_cov = bip_reach_model

            bip_trajectory = render_reach_trajectory(
                basis_model, primitive, prior_mean, prior_cov,
                start_xyz=ee_pos[:3], goal_xyz=goal[:3],
            )

            gripper_dmp = DMP(basis_type="step", dmp_type="gripper")
            gripper_dmp.set_weights(np.zeros(N_FUNCTIONS))
            gripper_dmp.set_step(STEP_SIZE)
            gripper_dmp.set_start(0.0)   # reach always starts and ends with the gripper open
            gripper_dmp.set_goal(goal[4])

            last_action, terminated = run_bip_phase(
                bip_trajectory, gripper_dmp, env, scene,
                label=subtask_name,
                goal_xyz=goal[:3],
                prev_action=last_action,
            )
        else:
            obj_pos_llm = _read_obj(scene, scene_key)

            # Fresh generator per subtask — mirrors orchestrator /refresh
            generator = DMPGenerator()
            weights = np.zeros((N_DOF, N_FUNCTIONS))
            try:
                weights, _, _ = generator.get_initial_weights(
                    task=subtask_desc,
                    n_functions=N_FUNCTIONS,
                    n_dof=N_DOF,
                    target_object=object_name,
                    movable_objects=[{object_name: obj_pos_llm.tolist(), **obstacle_positions}, {}],
                    task_file=task_file,
                )
                print(f"  LLM weights:\n{weights}")
            except Exception as e:
                import traceback
                print(f"  [WARNING] LLM failed, using zero weights. ({type(e).__name__}: {e})")
                traceback.print_exc()

            WEIGHT_SCALE_Z = 2.0 #20
            weights = weights.astype(float)
            weights[2] *= WEIGHT_SCALE_Z  # scale z only; DOF order: x=0, y=1, z=2, yaw=3, gripper=4
            gripper_start = 1.0  # carry/release always follow reach, so the gripper starts closed
            start = np.array([ee_pos[0], ee_pos[1], ee_pos[2], 0.0, gripper_start])

            dmps = build_dmps(weights)
            for i, dmp in enumerate(dmps):
                dmp.set_start(start[i])
                dmp.set_goal(goal[i])

            last_action, terminated = run_dmp_phase(
                dmps, env, scene,
                label=subtask_name,
                goal_xyz=goal[:3],
                prev_action=last_action,
            )

        if terminated:
            print("Environment terminated early, stopping.")
            break

        hold_steps = subtask.get("hold_after", 0)
        if hold_steps > 0:
            last_action, terminated = hold_and_grasp(env, last_action, steps=hold_steps)
            if terminated:
                print("Environment terminated during grasp hold, stopping.")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
