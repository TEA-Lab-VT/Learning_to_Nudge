# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test LMP DMP execution in IsaacLab using zero weights (no LLM API needed).
DMP class is imported from lmp-26/src/controller.py.

.. code-block:: bash

    ./isaaclab.sh -p "scripts/environments/state_machine/lift_cube_sm copy.py" --num_envs 1

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LMP DMP test in IsaacLab (zero weights, no API).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import sys
import unittest.mock

import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as R_scipy

# Load DMP directly from lmp-26's controller.py by file path to avoid any
# sys.modules naming conflicts with IsaacSim's own modules. controller.py
# imports utils.py which needs pygame/pynput (real-robot libs) — mock them.
import importlib.util

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

# Load DMPGenerator directly — bypasses the HTTP server entirely
from dotenv import load_dotenv
load_dotenv("/home/jhaixin/lmp-26.github.io/src/lmp/.env")

_gen_spec = importlib.util.spec_from_file_location(
    "lmp_dmp_generator", "/home/jhaixin/lmp-26.github.io/src/lmp/llm_dmp_generator.py"
)
_gen_mod = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen_mod)
DMPGenerator = _gen_mod.DMPGenerator

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# DMP parameters — match lmp-26/src/lmp/orchestrator_config.yaml
N_FUNCTIONS = 11   # lmp-26 DMP forcing_function hardcodes N=11
N_DOF = 5          # x, y, z, yaw, gripper
STEP_SIZE = 0.002  # how fast to move through the trajectory
DONE_THRESHOLD = 0.0112  # stop when canonical system x drops below this


def build_dmps(weights: np.ndarray) -> list:
    """Create 5 DMP objects from a (5, 11) weight matrix.

    Mirrors the DMP setup in lmp-26/src/controller.py lines 338-340.
    DOF order: x, y, z (gaussian) | yaw (gaussian) | gripper (step).
    """
    dmps = [DMP(basis_type="gaussian") for _ in range(3)]   # x, y, z
    dmps.append(DMP(basis_type="gaussian"))                  # yaw
    dmps.append(DMP(basis_type="step", dmp_type="gripper"))  # gripper
    for i, dmp in enumerate(dmps):
        dmp.set_weights(weights[i])
        dmp.set_step(STEP_SIZE)
    return dmps

#hard code or read from current robot
#check the defination/direction with xyz need to coordinate with the promt

def xi_to_action(xi: np.ndarray, device: str) -> torch.Tensor:
    """Convert DMP 5D output to IK-Abs 8D action tensor.

    IK-Abs action format: [x, y, z, qw, qx, qy, qz, gripper]
    gripper: 1.0 = open, -1.0 = close
    """
    x, y, z, yaw, grip = xi
    # Downward-facing base orientation (pi around x) composed with yaw
    r = R_scipy.from_euler("x", np.pi) * R_scipy.from_euler("z", yaw)
    q = r.as_quat()  # scipy returns [qx, qy, qz, qw]
    qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])
    gripper = -1.0 if grip > 0.5 else 1.0
    return torch.tensor([[x, y, z, qw, qx, qy, qz, gripper]], dtype=torch.float32, device=device)


def main():
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Fix object spawn position — disable pose randomization so the cube always
    # appears at the same default location (makes weight experiments reproducible).
    # env_cfg.events.reset_object_position.params["pose_range"] = {}

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    # Read start (EE) and goal (cube) positions from IsaacLab scene
    ee_frame = env.unwrapped.scene["ee_frame"]
    ee_pos = (
        ee_frame.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
    )[0].cpu().numpy()
    
    obj_pos = (
        env.unwrapped.scene["object"].data.root_pos_w - env.unwrapped.scene.env_origins
    )[0].cpu().numpy()

    # Read obstacle positions (object2, object3, object4 from lift_env_cfg.py)
    obstacle_names = [k for k in env.unwrapped.scene.keys() if k.startswith("object") and k != "object"]
    obstacle_names.sort()
    obstacle_positions = {}
    for name in obstacle_names:
        pos = (
            env.unwrapped.scene[name].data.root_pos_w - env.unwrapped.scene.env_origins
        )[0].cpu().numpy()
        label = f"obstacle_{name.replace('object', '')}"
        obstacle_positions[label] = pos
        print(f"{label}: {pos}")

    print(f"EE start : {ee_pos}")
    print(f"Cube goal: {obj_pos}")

    # Get DMP weights from LLM generator server (run llm_dmp_generator.py first).
    # Falls back to zero weights if server is not running.
    weights = np.zeros((N_DOF, N_FUNCTIONS))
    
    #set weight manually to push the cube up in z direction (for testing)
    # weights[2, 2] = 200   # z DOF, phase=0.2 → push UP at step~100 (clearly visible)

    # Call LLM directly via DMPGenerator (no HTTP server needed)
    try:
        _generator = DMPGenerator()
        # object_dict: target + obstacles; bbox_dict: physical size info for each
        object_dict = {"cube": obj_pos.tolist()}
        object_dict.update(obstacle_positions)
        bbox_dict = {
            "cube": {"height": 0.05, "width": 0.05, "length": 0.05, "angle": 0.0},
        }
        for label in obstacle_positions:
            bbox_dict[label] = {"height": 0.20, "width": 0.06, "length": 0.06, "angle": 0.0}

        weights, _, _ = _generator.get_initial_weights(
            task="pick up the cube and lift it above the table, avoiding the tall red obstacles",
            n_functions=N_FUNCTIONS,
            n_dof=N_DOF,
            target_object="cube",
            movable_objects=(object_dict, bbox_dict),
            task_file="/home/jhaixin/lmp-26.github.io/src/lmp/prompt/task_library/lift_cube.yaml",
        )
        print(f"LLM weights loaded:\n{weights}")
    except Exception as _e:
        import traceback
        print("[WARNING] LLM call failed, using zero weights.")
        print(f"[DEBUG] Exception type : {type(_e).__name__}")
        print(f"[DEBUG] Exception msg  : {_e}")
        traceback.print_exc()

    # weights[2, 2] = 200   # manual test: push z UP at step~100

    start = np.array([ee_pos[0], ee_pos[1], ee_pos[2], 0.0, 0.0])  # yaw=0, gripper open
    goal  = np.array([obj_pos[0], obj_pos[1], obj_pos[2], 0.0, 1.0])  # yaw=0, gripper close

    dmps = build_dmps(weights)
    for i, dmp in enumerate(dmps):
        dmp.set_start(start[i])
        dmp.set_goal(goal[i])

    print(f"DMP start : {start}")
    print(f"DMP goal  : {goal}")
    print("Running DMP execution loop...")

    # Hold in place before DMP starts (identity quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            if dmps[0].x > DONE_THRESHOLD:
                xi = np.array([dmp.step() for dmp in dmps])
                actions = xi_to_action(xi, env.unwrapped.device)
                if step % 50 == 0:
                    print(
                        f"step {step:4d} | "
                        f"pos=[{xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f}] | "
                        f"progress={1 - dmps[0].x:.3f}"
                    )
            else:
                print(f"DMP finished after {step} steps.")
                break

            if dones.any():
                break

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
