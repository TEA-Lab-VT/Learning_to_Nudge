# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Load environment and restore state from JSON.")
parser.add_argument("--num_envs", type=int, default=2000, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--json_file", type=str, default="/home/jhaixin/IsaacLab_from_sif/ncbf_trajectories_correct_0725_007/trajectories.json", help="Path to JSON file containing state data.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

parser.add_argument("--batch_size", type=int, default=2000, help="batch size")

parser.add_argument("--model_path", type=str, default="/home/jhaixin/IsaacLab_from_sif/results_correct_0725_007/model_iter_60.pt", help="model path")
parser.add_argument("--h_obj_normalizer_path", type=str, default="/home/jhaixin/IsaacLab_from_sif/results_correct_0725_007/local_normalizer", help="object state normalizer path")
parser.add_argument("--ee_normalizer_path", type=str, default="/home/jhaixin/IsaacLab_from_sif/results_correct_0725_007/arm_normalizer", help="end-effector normalizer path")
parser.add_argument("--cbf_min", type=float, default=0.00, help="minimum CBF value")
parser.add_argument("--cbf_max", type=float, default=0.05, help="maximum CBF value") #0.015,0.05

parser.add_argument("--horizon", type=int, default=1, help="history window size")
parser.add_argument("--step", type=int, default=2, help="next step size")
parser.add_argument("--min_seq_len", type=int, default=15, help="minimum sequence length")

parser.add_argument("--output_dir", type=str, default="boundary_check_results_006", help="output directory")

parser.add_argument("--batch_interval", type=float, default=5.0, help="batch interval time (seconds)")

# add training related parameters
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--training_batch_size", type=int, default=256, help="training batch size")
parser.add_argument("--training_epochs", type=int, default=100, help="training epochs per batch")
parser.add_argument("--margin_threshold", type=float, default=0.01, help="CBF margin threshold")
parser.add_argument("--derivative_threshold", type=float, default=0.02, help="derivative threshold")
parser.add_argument("--best_model_path", type=str, default="best_model.pt", help="best model save path")

# add buffer related parameters
parser.add_argument("--buffer_size", type=int, default=10, help="training data buffer size (batch times)")
parser.add_argument("--use_buffer", action="store_true", default=True, help="whether to use training data buffer")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import gymnasium as gym
import numpy as np
import torch
import omni.log
import json
import time
import os
import copy
import itertools

"""Rest everything follows."""

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# import managers
from isaaclab_tasks.utils.config import SafetyConfig, RobotConfig, EnvironmentConfig
from isaaclab_tasks.utils.environment_manager import EnvironmentManager
from isaaclab_tasks.utils.robot_manager import RobotManager
from isaaclab_tasks.utils.geometry_calculator import GeometryCalculator
from isaaclab_tasks.utils.safety_calculator import SafetyCalculator

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from models import NCBF, Normalizer

from tqdm import tqdm
from collections import defaultdict

env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()
tilt_violations = []
env_snapshot_time_indices = {}  # store time_index for each environment
# 添加全局变量
all_safe_transitions = []  # 存储所有安全转换信息
# add global variables for boundary samples data
boundary_samples_data = []  # store samples near the safety boundary after CBF calculation
current_batch_index = 0      # current batch index
total_batches = 0            # total number of batches
batch_size = 0               # number of samples processed per batch
batch_results = {}           # store results of each batch
env_instance = None          # environment instance

env_object_mappings = {}        # store environment ID to object information mapping
all_violations = []             # store all violation information
data = None                     # store processed data
indices = None                  # store index information

# training related global variables
best_performance = -float('inf')  # best performance score
performance_history = []          # performance history

# training data buffer
training_buffer = {
    "safe_local_xs": [],
    "safe_local_nxs": [],
    "safe_ee_xs": [],
    "safe_ee_nxs": [],
    "unsafe_local_xs": [],
    "unsafe_ee_xs": []
}
buffer_indices = {
    "safe_indices": [],
    "unsafe_indices": [],
    "safe_obj_names": [],
    "unsafe_obj_names": []
}
current_buffer_size = 0  # current buffer size
max_buffer_size = 5      # maximum buffer size (batch times)

_STATE_RESTORED = False

def pre_process_actions(
    delta_pose: np.ndarray,
    num_envs: int,
    device: str,
    env,  # used to get current EE state
) -> torch.Tensor:

    # get current EE state
    ee_pos, _ = robot_manager.get_ee_state(env)
    if ee_pos is None:
        # return 3D: Δx, Δy, gripper
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    # Δx, Δy, gripper
    if len(delta_pose) >= 2:
        delta_xy = delta_pose[:2]
    else:
        delta_xy = [0.0, 0.0]

    delta_xy_tensor = torch.tensor(delta_xy, dtype=torch.float32, device=device).repeat(num_envs, 1)

    # gripper velocity: close or open
    gripper_vel = torch.full((num_envs, 1), -1.0, dtype=torch.float32, device=device)  # -1.0 means close

    # output: Δx, Δy, gripper
    return torch.cat([delta_xy_tensor, gripper_vel], dim=1)

object_history_buffer = {}  # 格式: {env_idx: {obj_name: [历史状态列表]}}

def update_object_history_buffer(env, object_states, object_names):
    """更新物体历史状态缓冲区"""
    global object_history_buffer
    
    # 获取当前物体状态
    for env_idx in range(env.num_envs):
        # 确保环境索引存在于缓冲区中
        if env_idx not in object_history_buffer:
            object_history_buffer[env_idx] = {}
        
        for obj_idx, obj_name in enumerate(object_names):
            # 确保物体名称存在于缓冲区中
            if obj_name not in object_history_buffer[env_idx]:
                object_history_buffer[env_idx][obj_name] = []
            
            # 获取当前物体状态
            obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
            obj_quat = object_states[env_idx, obj_idx, 3:7]
            tilt_angle_rad = geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat.unsqueeze(0)).item()
            
            # 获取基准位置（第一个状态的位置）
            if not object_history_buffer[env_idx][obj_name]:
                base_pos = obj_pos
            else:
                # 从第一个历史状态中提取基准位置
                first_state = object_history_buffer[env_idx][obj_name][0]
                # 基准位置 = 当前位置 - 相对位置
                base_pos = obj_pos.copy()
                base_pos[0] -= first_state[1]  # rel_x
                base_pos[1] -= first_state[2]  # rel_y
                base_pos[2] -= first_state[3]  # rel_z
            
            # 计算相对于基准位置的相对位置
            rel_pos = obj_pos - base_pos
            
            # 构建状态 [tilt, rel_x, rel_y, rel_z]
            state = [tilt_angle_rad, rel_pos[0], rel_pos[1], rel_pos[2]]
            
            # 添加到历史列表
            object_history_buffer[env_idx][obj_name].append(state)
            
            # 保持列表长度为horizon
            if len(object_history_buffer[env_idx][obj_name]) > args_cli.horizon:
                object_history_buffer[env_idx][obj_name] = object_history_buffer[env_idx][obj_name][-args_cli.horizon:]

# def move_ee_towards_object(
#     env,
#     ee_position,
#     obj_position,
#     env_idx,
#     obj_idx,
#     object_states,
#     object_names,
#     model,
#     h_obj_normalizer,
#     ee_normalizer,
#     move_step_size=0.01,
#     grid_size=8,
#     safety_threshold=-0.05
# ):

#     global object_history_buffer
    
#     # calculate the distance between the end effector and the object
#     distance = torch.norm(obj_position[:2] - ee_position[:2])
    
#     # initialize the action
#     ACTION_DIM = 3
#     action = torch.zeros(ACTION_DIM, device=env.device)
    
#     # if the distance is large enough, calculate the safe action
#     if distance > 0.01:
#         obj_name = object_names[obj_idx]
        
#         # get the object history state
#         obj_history = []
#         if env_idx in object_history_buffer and obj_name in object_history_buffer[env_idx]:
#             obj_history = np.array(object_history_buffer[env_idx][obj_name], dtype=np.float32)
        
#         # if there is not enough history data, use the current state to fill
#         if len(obj_history) < args_cli.horizon:
#             # get the current object state
#             obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
#             obj_quat = object_states[env_idx, obj_idx, 3:7]
#             tilt_angle_rad = geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat.unsqueeze(0)).item()
            
#             # fill the history
#             current_state = [tilt_angle_rad, 0.0, 0.0, 0.0]
#             obj_history = np.array([current_state for _ in range(args_cli.horizon)], dtype=np.float32)
        
#         # ensure the history data length is correct
#         if len(obj_history) > args_cli.horizon:
#             obj_history = obj_history[-args_cli.horizon:]
        
#         # get the current object position
#         obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
        
#         # --------- gradient-based optimization starts ---------
        
#         # initialize multiple random actions as starting points
#         num_samples = 10  # number of samples for multiple random restarts
#         actions = torch.zeros((num_samples, 2), device=env.device).uniform_(-move_step_size, move_step_size)
#         actions.requires_grad = True  # enable gradient calculation
        
#         # normalize the history state
#         normalized_obj_history = normalize_state_sequence(obj_history, h_obj_normalizer, update=False)
#         obj_history_tensor = torch.tensor(normalized_obj_history, dtype=torch.float32, device=env.device).unsqueeze(0).repeat(num_samples, 1, 1)
        
#         # prepare the optimizer
#         optimizer = torch.optim.Adam([actions], lr=0.001)
#         best_cbf = float('inf')
#         best_action = None
        
#         max_iters = 20  # maximum number of iterations
#         for iter in range(max_iters):
#             # make sure the action is within the valid range
#             bounded_actions = torch.clamp(actions, -move_step_size, move_step_size)
            
#             # predict the position after executing the action
#             predicted_positions = torch.zeros((num_samples, 2), device=env.device)
#             predicted_positions[:, 0] = ee_position[0] + bounded_actions[:, 0]
#             predicted_positions[:, 1] = ee_position[1] + bounded_actions[:, 1]
            
#             # calculate the relative position
#             obj_pos_tensor = torch.tensor(obj_pos[:2], device=env.device)
#             rel_ee_positions = predicted_positions - obj_pos_tensor
            
#             # normalize the relative position - note that this will break the gradient flow
#             normalized_rel_ee_positions = []
#             for i in range(num_samples):
#                 rel_pos = rel_ee_positions[i].detach().cpu().numpy().reshape(1, -1)
#                 norm_rel_pos = ee_normalizer(rel_pos)
#                 normalized_rel_ee_positions.append(norm_rel_pos)
            
#             # only convert once
#             normalized_rel_ee_positions = torch.tensor(np.vstack(normalized_rel_ee_positions),
#                                                 dtype=torch.float32,
#                                                 device=env.device)
            
#             # use the NCBF model to evaluate the safety
#             was_training = model.training
#             # temporarily switch to training method
#             model.train()
#             cbf_values = model([obj_history_tensor, normalized_rel_ee_positions])
#             # switch abck to eval method
#             if not was_training:
#                 model.eval()
            
#             # calculate the loss and optimize
#             loss = cbf_values.mean()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # evaluate the best action in the current iteration
#             with torch.no_grad():
#                 bounded_actions = torch.clamp(actions, -move_step_size, move_step_size)
#                 current_cbf_values = cbf_values.squeeze()
#                 min_cbf, min_idx = torch.min(current_cbf_values, dim=0)
#                 min_cbf_val = min_cbf.item()
                
#                 if min_cbf_val < best_cbf:
#                     best_cbf = min_cbf_val
#                     best_action = bounded_actions[min_idx].clone().cpu().numpy()

#         # safety check
#         if best_cbf < safety_threshold and best_action is not None:
#             # use the best safe action
#             direction_np = np.array([best_action[0], best_action[1], 0.0])
#             single_action = pre_process_actions(direction_np[:2], 1, env.device, env)
#             action = single_action[0]
            
#             if env_idx % 10 == 0: 
#                 print(f"env {env_idx} - use safe action (gradient): [{best_action[0]:.4f}, {best_action[1]:.4f}], CBF value: {best_cbf:.4f}")
#         else:
#             # if all actions are unsafe, choose stationary
#             print(f"env {env_idx} - all actions are unsafe (best CBF value: {best_cbf:.4f} > threshold: {safety_threshold}), choose stationary")

#         return action, distance
    
def move_ee_towards_object(
    env,
    ee_position,
    obj_position,
    env_idx,
    obj_idx,
    object_states,
    object_names,
    model,
    h_obj_normalizer,
    ee_normalizer,
    move_step_size=0.01,
    grid_size=8,
    safety_threshold=-0.05
):

    global object_history_buffer
    
    # calculate the distance between the end effector and the object
    distance = torch.norm(obj_position[:2] - ee_position[:2])
    
    # initialize the action
    ACTION_DIM = 3
    action = torch.zeros(ACTION_DIM, device=env.device)
    
    # if the distance is large enough, calculate the safe action
    if distance > 0.01:
        # generate the discrete action space (the grid on the XY plane)
        x_moves = np.linspace(-move_step_size, move_step_size, grid_size)
        y_moves = np.linspace(-move_step_size, move_step_size, grid_size)
        action_grid = np.array(list(itertools.product(x_moves, y_moves)))
        
        # evaluate the safety of each possible action
        cbf_values = []
        safe_actions = []
        
        obj_name = object_names[obj_idx]
        
        # get the object history state
        obj_history = []
        if env_idx in object_history_buffer and obj_name in object_history_buffer[env_idx]:
            obj_history = np.array(object_history_buffer[env_idx][obj_name], dtype=np.float32)
        
        # if there is not enough history data, use the current state to fill
        if len(obj_history) < args_cli.horizon:
            # get the current object state
            obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
            obj_quat = object_states[env_idx, obj_idx, 3:7]
            tilt_angle_rad = geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat.unsqueeze(0)).item()
            
            # fill the history
            current_state = [tilt_angle_rad, 0.0, 0.0, 0.0]
            obj_history = np.array([current_state for _ in range(args_cli.horizon)], dtype=np.float32)
        
        # ensure the history data length is correct
        if len(obj_history) > args_cli.horizon:
            obj_history = obj_history[-args_cli.horizon:]
        
        # get the current object position
        obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
        
        for ac in action_grid:
            # predict the position after executing the action
            predicted_pos = ee_position.clone()
            predicted_pos[0] += ac[0]
            predicted_pos[1] += ac[1]
            
            rel_ee_pos = predicted_pos[:2].cpu().numpy() - obj_pos[:2]
            # print(f"rel_ee_pos shape: {rel_ee_pos.shape}, type: {type(rel_ee_pos)}")
            # print(f"normalizer mean shape: {ee_normalizer._mean.shape}")

            # reshape to 2D tensor - (1, features) format
            if rel_ee_pos.ndim == 1:
                rel_ee_pos = rel_ee_pos.reshape(1, -1)
                
            # print(f"reshaped rel_ee_pos shape: {rel_ee_pos.shape}")
            normalized_rel_ee_pos = ee_normalizer(rel_ee_pos)

            # normalize the features
            normalized_obj_history = np.zeros_like(obj_history)
            for t in range(obj_history.shape[0]):
                normalized_obj_history = normalize_state_sequence(obj_history, h_obj_normalizer, update=False)
            
            normalized_rel_ee_pos = ee_normalizer(rel_ee_pos)
            
            # convert to tensor
            obj_history_tensor = torch.tensor(normalized_obj_history, dtype=torch.float32, device=env.device).unsqueeze(0)
            rel_ee_pos_tensor = torch.tensor(normalized_rel_ee_pos, dtype=torch.float32, device=env.device).unsqueeze(0)

            # print the shape for debugging
            # print(f"obj_history_tensor shape: {obj_history_tensor.shape}")
            # print(f"rel_ee_pos_tensor shape: {rel_ee_pos_tensor.shape}")

            # ensure the rel_ee_pos_tensor is 2D (batch, features)
            if rel_ee_pos_tensor.dim() == 3:  # 如果是(batch, 1, features)
                rel_ee_pos_tensor = rel_ee_pos_tensor.squeeze(1)

            # print(f"the shape of the adjusted rel_ee_pos_tensor: {rel_ee_pos_tensor.shape}")
            
            # use the NCBF model to evaluate the safety
            with torch.no_grad():
                cbf_value = model([obj_history_tensor, rel_ee_pos_tensor]).cpu().numpy()[0, 0]
            
            # record the safety value and the corresponding action
            cbf_values.append(cbf_value)
            safe_actions.append(ac)
        
        # find the safest action (the lowest CBF value)
        if cbf_values:
            # find the index of the action with the lowest CBF value
            best_idx = np.argmin(cbf_values)
            best_cbf = cbf_values[best_idx]
            best_action = safe_actions[best_idx]
            
            # if the best CBF value is greater than the safety threshold, use the action
            if best_cbf < safety_threshold:
                # process the best action
                direction_np = np.array([best_action[0], best_action[1], 0.0])
                single_action = pre_process_actions(direction_np[:2], 1, env.device, env)
                action = single_action[0]
                
                if env_idx % 10 == 0: 
                    print(f"env {env_idx} - use safe action: [{best_action[0]:.4f}, {best_action[1]:.4f}], CBF value: {best_cbf:.4f}")
            else:
                # if all actions are unsafe, choose stationary
                print(f"env {env_idx} - all actions are unsafe (best CBF value: {best_cbf:.4f} < threshold: {safety_threshold}), choose stationary")
        else:
            print(f"env {env_idx} - cannot evaluate action, choose stationary")
    
    return action, distance

def organize_by_object(json_data, max_tilt_angle=.0, batch_indices=None):
    """organize data by object and filter out samples with tilt angle > max_tilt_angle"""
    if batch_indices is None:
        batch_indices = range(len(json_data))
    
    organized = defaultdict(lambda: defaultdict(list))
    
    # 统计变量
    total_samples = 0
    large_tilt_count = 0
    
    for batch_idx, t in enumerate(batch_indices):
        entry = json_data[t]
        objects = entry["objects_state"]
        is_safe = entry.get("is_safe", [[True] for _ in objects])
        ee_state = entry["ee_state"]
        
        for obj_idx, (obj, safe_flag) in enumerate(zip(objects, is_safe)):
            total_samples += 1
            
            # 获取 tilt_cost (弧度)
            tilt_angle_rad = obj.get("tilt_cost", 0)
            
            # 检查 tilt angle 是否大于阈值 (弧度)
            max_tilt_rad = np.radians(max_tilt_angle)
            if abs(tilt_angle_rad) > max_tilt_rad:
                large_tilt_count += 1
                continue  # 跳过此样本
            
            safe = safe_flag[0] if isinstance(safe_flag, list) else safe_flag
            obj_name = obj.get("name", f"object_{obj_idx}")
            
            data_point = {
                "t": t,                 # time index in original JSON
                "batch_idx": batch_idx,
                "ee_state": ee_state,
                "obj_state": obj,
                "is_safe": safe,
                "obj_name": obj_name
            }
            organized[obj_idx][t] = data_point
    
    print(f"total samples: {total_samples}")
    print(f"samples with tilt angle > {max_tilt_angle} degrees: {large_tilt_count}")
    print(f"samples with tilt angle > {max_tilt_angle} degrees: {large_tilt_count / total_samples * 100:.2f}%")
    print(f"retained samples: {total_samples - large_tilt_count}")
    
    return organized

def build_sequences(organized, horizon, step, min_seq_len):
    """build sequence data，并为所有样本（包括unsafe）准备next state"""
    safe_local_xs, safe_local_nxs, safe_ee_xs, safe_ee_nxs = [], [], [], []
    unsafe_local_xs, unsafe_local_nxs, unsafe_ee_xs, unsafe_ee_nxs = [], [], [], []  # 为unsafe也添加nxs
    
    safe_indices, unsafe_indices = [], []
    safe_obj_names, unsafe_obj_names = [], []
    
    for obj_idx, time_dict in tqdm(organized.items(), desc="process object data"):
        times = sorted(time_dict.keys())
        if len(times) < min_seq_len:
            continue
        
        for i in range(horizon, len(times) - step):
            # 当前时间窗口
            idxs = times[i-horizon:i]
            # 下一个时间窗口
            next_idxs = times[i-horizon+1:i+1]
            curr_idx = times[i]
            next_idx = times[i+step]
            
            # 基准位置
            base_idx = idxs[0]
            nbase_idx = next_idxs[0]
            
            obj_name = time_dict[base_idx]["obj_name"]
            
            # 获取基准位置坐标
            base_pos = np.array([
                time_dict[base_idx]["obj_state"]["position"]["x"],
                time_dict[base_idx]["obj_state"]["position"]["y"],
                time_dict[base_idx]["obj_state"]["position"]["z"]
            ])
            
            nbase_pos = np.array([
                time_dict[nbase_idx]["obj_state"]["position"]["x"],
                time_dict[nbase_idx]["obj_state"]["position"]["y"],
                time_dict[nbase_idx]["obj_state"]["position"]["z"]
            ])
            
            # 构建当前状态序列
            local_x = []
            for j in idxs:
                obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                obj_pos = np.array([
                    time_dict[j]["obj_state"]["position"]["x"],
                    time_dict[j]["obj_state"]["position"]["y"],
                    time_dict[j]["obj_state"]["position"]["z"]
                ])
                rel_pos = obj_pos - base_pos
                features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                features.append(time_dict[j]["t"])
                features.append(obj_name)
                local_x.append(features)
            
            # 构建下一状态序列
            local_nx = []
            for j in next_idxs:
                obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                obj_pos = np.array([
                    time_dict[j]["obj_state"]["position"]["x"],
                    time_dict[j]["obj_state"]["position"]["y"],
                    time_dict[j]["obj_state"]["position"]["z"]
                ])
                rel_pos = obj_pos - nbase_pos
                features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                features.append(time_dict[j]["t"])
                features.append(obj_name)
                local_nx.append(features)
            
            # 当前物体位置
            curr_obj_pos = np.array([
                time_dict[curr_idx]["obj_state"]["position"]["x"],
                time_dict[curr_idx]["obj_state"]["position"]["y"],
                time_dict[curr_idx]["obj_state"]["position"]["z"]
            ])
            
            # 下一物体位置
            next_obj_pos = np.array([
                time_dict[next_idx]["obj_state"]["position"]["x"],
                time_dict[next_idx]["obj_state"]["position"]["y"],
                time_dict[next_idx]["obj_state"]["position"]["z"]
            ])
            
            # 当前末端执行器相对位置
            curr_ee_pos = np.array([
                time_dict[curr_idx]["ee_state"]["x"],
                time_dict[curr_idx]["ee_state"]["y"]
            ])
            
            # 下一末端执行器相对位置
            next_ee_pos = np.array([
                time_dict[next_idx]["ee_state"]["x"],
                time_dict[next_idx]["ee_state"]["y"]
            ])
            
            rel_ee_x = curr_ee_pos - curr_obj_pos[:2]
            rel_ee_nx = next_ee_pos - next_obj_pos[:2]
            
            is_safe = time_dict[curr_idx]["is_safe"]
            orig_index = time_dict[curr_idx]["t"]
            
            # 根据安全状态分类
            if is_safe:
                safe_local_xs.append(local_x)
                safe_local_nxs.append(local_nx)
                safe_ee_xs.append(rel_ee_x)
                safe_ee_nxs.append(rel_ee_nx)
                safe_indices.append(orig_index)
                safe_obj_names.append(obj_name)
            else:
                unsafe_local_xs.append(local_x)
                unsafe_local_nxs.append(local_nx)  # 为unsafe样本添加next state
                unsafe_ee_xs.append(rel_ee_x)
                unsafe_ee_nxs.append(rel_ee_nx)    # 为unsafe样本添加ee next state
                unsafe_indices.append(orig_index)
                unsafe_obj_names.append(obj_name)
    
    # 构建返回数据
    data = {
        "safe_local_xs": np.array(safe_local_xs, dtype=object),
        "safe_local_nxs": np.array(safe_local_nxs, dtype=object),
        "safe_ee_xs": np.array(safe_ee_xs),
        "safe_ee_nxs": np.array(safe_ee_nxs),
        "unsafe_local_xs": np.array(unsafe_local_xs, dtype=object),
        "unsafe_local_nxs": np.array(unsafe_local_nxs, dtype=object),  # 添加next state
        "unsafe_ee_xs": np.array(unsafe_ee_xs),
        "unsafe_ee_nxs": np.array(unsafe_ee_nxs)  # 添加ee next state
    }
    
    indices = {
        "safe_indices": np.array(safe_indices),
        "unsafe_indices": np.array(unsafe_indices),
        "safe_obj_names": np.array(safe_obj_names),
        "unsafe_obj_names": np.array(unsafe_obj_names)
    }
    
    return data, indices


def load_model(model_path, h_obj_normalizer_path, ee_normalizer_path):
    """load NCBF model and normalizer"""
    h_obj_dim = 4
    ee_dim = 2

    model = NCBF.load(
        path=model_path,
        hiddens=[64, 64],
        seq_hiddens=[64]
    )
    
    h_obj_normalizer = Normalizer(h_obj_dim)
    ee_normalizer = Normalizer(ee_dim)
    h_obj_normalizer.load_model(h_obj_normalizer_path)
    ee_normalizer.load_model(ee_normalizer_path)
    
    print("NCBF model loaded successfully")
    return model, h_obj_normalizer, ee_normalizer


def prepare_features_data(data):
    """prepare feature data for unsafe samples"""
    features_data = {}
    
    # 处理不安全样本特征
    if "unsafe_local_xs" in data and len(data["unsafe_local_xs"]) > 0:
        unsafe_local_xs = data["unsafe_local_xs"]
        unsafe_features = np.zeros((unsafe_local_xs.shape[0], unsafe_local_xs.shape[1], 4), dtype=np.float32)
        
        for i in range(unsafe_local_xs.shape[0]):
            for j in range(unsafe_local_xs.shape[1]):
                unsafe_features[i, j] = unsafe_local_xs[i, j, :4].astype(np.float32)
        
        features_data["unsafe_local_xs"] = unsafe_features
        print(f"processed {len(unsafe_local_xs)} unsafe samples features")
    else:
        print("warning: no unsafe samples features (unsafe_local_xs)")
    
    # 处理不安全样本的末端执行器状态
    if "unsafe_ee_xs" in data and data["unsafe_ee_xs"].size > 0:
        features_data["unsafe_ee_xs"] = data["unsafe_ee_xs"].astype(np.float32)
        print(f"processed {data['unsafe_ee_xs'].shape[0]} unsafe samples ee features")
    else:
        print("warning: no unsafe samples ee features (unsafe_ee_xs)")
    
    # 如果需要，也可以处理不安全样本的next state特征
    if "unsafe_local_nxs" in data and len(data["unsafe_local_nxs"]) > 0:
        unsafe_local_nxs = data["unsafe_local_nxs"]
        unsafe_nxs_features = np.zeros((unsafe_local_nxs.shape[0], unsafe_local_nxs.shape[1], 4), dtype=np.float32)
        
        for i in range(unsafe_local_nxs.shape[0]):
            for j in range(unsafe_local_nxs.shape[1]):
                unsafe_nxs_features[i, j] = unsafe_local_nxs[i, j, :4].astype(np.float32)
        
        features_data["unsafe_local_nxs"] = unsafe_nxs_features
    
    if "unsafe_ee_nxs" in data and data["unsafe_ee_nxs"].size > 0:
        features_data["unsafe_ee_nxs"] = data["unsafe_ee_nxs"].astype(np.float32)
    
    return features_data

def extract_training_features(data):
    """extract training features from data"""
    training_features = {}
    
    # extract safe sample features
    if "safe_local_xs" in data and len(data["safe_local_xs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["safe_local_xs"], list):
            safe_local_xs = np.array(data["safe_local_xs"], dtype=object)
        else:
            safe_local_xs = data["safe_local_xs"]
            
        safe_h_obj_features = np.zeros((len(safe_local_xs), len(safe_local_xs[0]), 4), dtype=np.float32)
        
        for i in range(len(safe_local_xs)):
            for j in range(len(safe_local_xs[0])):
                safe_h_obj_features[i, j] = safe_local_xs[i][j][:4].astype(np.float32)
        
        training_features["safe_h_obj_xs"] = safe_h_obj_features
    
    if "safe_local_nxs" in data and len(data["safe_local_nxs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["safe_local_nxs"], list):
            safe_local_nxs = np.array(data["safe_local_nxs"], dtype=object)
        else:
            safe_local_nxs = data["safe_local_nxs"]
            
        safe_h_obj_nxs_features = np.zeros((len(safe_local_nxs), len(safe_local_nxs[0]), 4), dtype=np.float32)
        
        for i in range(len(safe_local_nxs)):
            for j in range(len(safe_local_nxs[0])):
                safe_h_obj_nxs_features[i, j] = safe_local_nxs[i][j][:4].astype(np.float32)
        
        training_features["safe_h_obj_nxs"] = safe_h_obj_nxs_features
    
    if "safe_ee_xs" in data and len(data["safe_ee_xs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["safe_ee_xs"], list):
            safe_ee_xs = np.array(data["safe_ee_xs"], dtype=np.float32)
        else:
            safe_ee_xs = data["safe_ee_xs"]
            
        training_features["safe_ee_xs"] = safe_ee_xs
    
    if "safe_ee_nxs" in data and len(data["safe_ee_nxs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["safe_ee_nxs"], list):
            safe_ee_nxs = np.array(data["safe_ee_nxs"], dtype=np.float32)
        else:
            safe_ee_nxs = data["safe_ee_nxs"]
            
        training_features["safe_ee_nxs"] = safe_ee_nxs
    
    # extract unsafe sample features
    if "unsafe_local_xs" in data and len(data["unsafe_local_xs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["unsafe_local_xs"], list):
            unsafe_local_xs = np.array(data["unsafe_local_xs"], dtype=object)
        else:
            unsafe_local_xs = data["unsafe_local_xs"]
            
        unsafe_h_obj_features = np.zeros((len(unsafe_local_xs), len(unsafe_local_xs[0]), 4), dtype=np.float32)
        
        for i in range(len(unsafe_local_xs)):
            for j in range(len(unsafe_local_xs[0])):
                unsafe_h_obj_features[i, j] = unsafe_local_xs[i][j][:4].astype(np.float32)
        
        training_features["unsafe_h_obj_xs"] = unsafe_h_obj_features
    
    if "unsafe_ee_xs" in data and len(data["unsafe_ee_xs"]) > 0:
        # ensure data is numpy array
        if isinstance(data["unsafe_ee_xs"], list):
            unsafe_ee_xs = np.array(data["unsafe_ee_xs"], dtype=np.float32)
        else:
            unsafe_ee_xs = data["unsafe_ee_xs"]
            
        training_features["unsafe_ee_xs"] = unsafe_ee_xs
    
    return training_features

# normalize h_obj sequence data
def normalize_state_sequence(local_xs, normalizer, update=False):
    """flatten sequence data and then restore original shape"""
    flattened_local_xs = copy.deepcopy(local_xs.reshape([-1, local_xs.shape[-1]]))
    if update:
        normalizer.experience(flattened_local_xs)
    normalized = normalizer(flattened_local_xs)
    return normalized.reshape(local_xs.shape)
    
def normalize_training_data(features, h_obj_normalizer, ee_normalizer, update=False):
    """normalize training data, use normalizer's experience and reshape method"""
    normalized_data = []
    
    # if no all necessary data, return empty list
    if not all(k in features for k in ["safe_h_obj_xs", "safe_ee_xs", "unsafe_h_obj_xs", "unsafe_ee_xs"]):
        print("warning: missing necessary training features")
        return normalized_data
    
    # extract features
    safe_h_obj_xs = features["safe_h_obj_xs"]
    safe_ee_xs = features["safe_ee_xs"]
    unsafe_h_obj_xs = features["unsafe_h_obj_xs"]
    unsafe_ee_xs = features["unsafe_ee_xs"]
    
    # optional features
    safe_h_obj_nxs = features.get("safe_h_obj_nxs", None)
    safe_ee_nxs = features.get("safe_ee_nxs", None)
    
    print(f"safe feature shape: safe_h_obj_xs={safe_h_obj_xs.shape}, safe_ee_xs={safe_ee_xs.shape}")
    print(f"unsafe feature shape: unsafe_h_obj_xs={unsafe_h_obj_xs.shape}, unsafe_ee_xs={unsafe_ee_xs.shape}")
    

    
    # normalize object history state
    normalized_safe_h_obj_xs = normalize_state_sequence(
        safe_h_obj_xs, h_obj_normalizer, update=update).astype(np.float32)
    
    normalized_unsafe_h_obj_xs = normalize_state_sequence(
        unsafe_h_obj_xs, h_obj_normalizer, update=False).astype(np.float32)
    
    # update ee normalizer's statistics
    if update:
        ee_normalizer.experience(safe_ee_xs)
        ee_normalizer.experience(unsafe_ee_xs)
    
    # normalize EE state
    normalized_safe_ee_xs = ee_normalizer(safe_ee_xs).astype(np.float32)
    normalized_unsafe_ee_xs = ee_normalizer(unsafe_ee_xs).astype(np.float32)
    
    # process optional features
    normalized_safe_h_obj_nxs = None
    normalized_safe_ee_nxs = None
    
    if safe_h_obj_nxs is not None:
        normalized_safe_h_obj_nxs = normalize_state_sequence(
            safe_h_obj_nxs, h_obj_normalizer, update=False).astype(np.float32)
    
    if safe_ee_nxs is not None:
        normalized_safe_ee_nxs = ee_normalizer(safe_ee_nxs).astype(np.float32)
    
    # assemble training data
    normalized_data = [
        normalized_safe_h_obj_xs,
        normalized_safe_h_obj_nxs,
        normalized_safe_ee_xs,
        normalized_safe_ee_nxs,
        normalized_unsafe_h_obj_xs,
        normalized_unsafe_ee_xs
    ]
    
    # filter out None values
    normalized_data = [data for data in normalized_data if data is not None]
    
    return normalized_data


def evaluate_model_performance(model, normalized_data):
    """evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # safe sample prediction
    normalized_safe_h_obj_xs = torch.tensor(normalized_data[0], dtype=torch.float32, device=device)
    normalized_safe_ee_xs = torch.tensor(normalized_data[2], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        safe_preds = model([normalized_safe_h_obj_xs, normalized_safe_ee_xs]).cpu().numpy()
    
    # unsafe sample prediction
    normalized_unsafe_h_obj_xs = torch.tensor(normalized_data[4], dtype=torch.float32, device=device)
    normalized_unsafe_ee_xs = torch.tensor(normalized_data[5], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        unsafe_preds = model([normalized_unsafe_h_obj_xs, normalized_unsafe_ee_xs]).cpu().numpy()
    
    # calculate correct classification rate
    safe_accuracy = np.mean(safe_preds < 0)
    unsafe_accuracy = np.mean(unsafe_preds > 0)
    
    # calculate comprehensive performance score (average correct rate of safe and unsafe classification)
    performance = (safe_accuracy + unsafe_accuracy) / 2.0
    
    print(f"model performance evaluation:")
    print(f"  safe sample classification accuracy: {safe_accuracy:.4f} ({np.sum(safe_preds < 0)}/{len(safe_preds)})")
    print(f"  unsafe sample classification accuracy: {unsafe_accuracy:.4f} ({np.sum(unsafe_preds > 0)}/{len(unsafe_preds)})")
    print(f"  comprehensive performance score: {performance:.4f}")
    
    return performance


def calculate_cbf_values(model, h_obj_normalizer, ee_normalizer, features_data, batch_size=128):
    """calculate CBF values"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_cbf_values = {}
    
    # 处理不安全样本而非安全样本
    if "unsafe_local_xs" in features_data and "unsafe_ee_xs" in features_data:
        unsafe_h_obj_xs = features_data["unsafe_local_xs"]
        unsafe_ee_xs = features_data["unsafe_ee_xs"]
        
        unsafe_cbf_values = []
        
        print(f"start processing {len(unsafe_h_obj_xs)} unsafe samples CBF values...")
        
        num_batches = (len(unsafe_h_obj_xs) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="processing unsafe samples"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unsafe_h_obj_xs))
            
            batch_h_obj_xs = unsafe_h_obj_xs[start_idx:end_idx]
            batch_ee_xs = unsafe_ee_xs[start_idx:end_idx]

            normalized_batch_h_obj_xs = np.zeros_like(batch_h_obj_xs)
            for t in range(batch_h_obj_xs.shape[1]):
                normalized_batch_h_obj_xs[:, t, :] = h_obj_normalizer(batch_h_obj_xs[:, t, :])

            normalized_batch_ee_xs = ee_normalizer(batch_ee_xs)

            h_obj_xs_tensor = torch.tensor(normalized_batch_h_obj_xs, dtype=torch.float32, device=device)
            ee_xs_tensor = torch.tensor(normalized_batch_ee_xs, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                batch_cbf_values = model([h_obj_xs_tensor, ee_xs_tensor]).cpu().numpy().flatten()
            
            unsafe_cbf_values.extend(batch_cbf_values)
        
        all_cbf_values["unsafe"] = np.array(unsafe_cbf_values)
    else:
        print("warning: no unsafe samples features data, cannot calculate unsafe samples CBF values")
    
    return all_cbf_values

def extract_boundary_samples(features_data, indices, all_cbf_values, cbf_min=0.0, cbf_max=0.015):
    """提取CBF值在指定范围内的不安全样本"""
    boundary_info = []
    
    if "unsafe" in all_cbf_values and "unsafe_local_xs" in features_data:
        unsafe_cbf_values = all_cbf_values["unsafe"]
        # 选择CBF值在指定范围内的不安全样本（可能转为安全）
        boundary_indices = np.where((unsafe_cbf_values >= cbf_min) & (unsafe_cbf_values <= cbf_max))[0]
        
        if len(boundary_indices) > 0:
            unsafe_indices = indices["unsafe_indices"]
            unsafe_obj_names = indices["unsafe_obj_names"]
            
            for i, idx in enumerate(boundary_indices):
                orig_idx = unsafe_indices[idx]
                obj_name = unsafe_obj_names[idx]
                
                sample_info = {
                    "sample_index": int(idx),  # unsafe_local_xs中的索引
                    "orig_index": int(orig_idx),  # 原始JSON中的索引
                    "object_name": str(obj_name),
                    "cbf_value": float(unsafe_cbf_values[idx])
                }
                boundary_info.append(sample_info)
    
    return boundary_info


def clear_scene():
    """clear existing elements in scene"""
    import omni.usd
    from pxr import Usd
    
    stage = omni.usd.get_context().get_stage()
    
    env_path = "/World/envs"
    if stage.GetPrimAtPath(env_path).IsValid():
        print(f"clear environment: {env_path}")
        stage.RemovePrim(env_path)
    
    print("scene cleared")

def restore_state_from_json(self, env_cfg, env_idx=None, params=None):
    """restore state from calculated boundary samples and establish detailed mapping relationship"""
    global boundary_samples_data, current_batch_index, batch_size, total_batches
    global env_snapshot_time_indices, env_object_mappings
    
    # calculate current batch sample range
    start_idx = current_batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(boundary_samples_data))
    current_batch_samples = boundary_samples_data[start_idx:end_idx]
    
    print(f"\n===== restore batch {current_batch_index + 1}/{total_batches} =====")
    print(f"process samples {start_idx} to {end_idx-1} (total {len(current_batch_samples)} samples)")
    
    env_ids = list(range(self.num_envs))
    
    # initialize current batch environment to object mapping
    env_object_mappings = {}
    
    # prepare environment data
    env_data = []
    for i, env_id in enumerate(env_ids):
        # if environment number exceeds current batch sample number, cycle through samples
        sample_idx = i % len(current_batch_samples)
        env_state = current_batch_samples[sample_idx]
        
        # record original JSON index
        orig_index = start_idx + sample_idx
        if "orig_index" in env_state:
            orig_index = env_state["orig_index"]
        
        # save time index mapping relationship
        env_snapshot_time_indices[env_id] = orig_index
        
        # initialize object mapping for this environment
        env_object_mappings[env_id] = {}
        
        # record object information
        for obj_info in env_state["objects_state"]:
            obj_name = obj_info["name"]
            env_object_mappings[env_id][obj_name] = {
                "orig_index": orig_index,
                "sample_idx": sample_idx + start_idx,
                "object_data": obj_info
            }
        
        env_data.append({
            "env_id": env_id,
            "sample_idx": sample_idx + start_idx,
            "robot_joints": env_state["robot_joints"],
            "objects_state": env_state["objects_state"],
            "ee_state": env_state.get("ee_state", None)
        })
    
    # apply robot joint state
    robot = self.scene.articulations.get("robot", None)
    
    # get environment origin for coordinate conversion
    env_origins = self.scene.env_origins
    
    # first apply robot joint state
    print("\n===== restore robot state =====")
    for data in env_data:
        env_id = data["env_id"]
        env_tensor_id = torch.tensor([env_id], device=self.device)
        
        # apply robot joint state
        joint_pos = torch.tensor(data["robot_joints"], device=self.device).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_pos)
        
        robot.set_joint_position_target(joint_pos, env_ids=env_tensor_id)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_tensor_id)
        print(f"environment {env_id}: robot joint state restored (sample {data['sample_idx']})")
    
    # render several frames to update robot state
    print("render frames to update robot state...")
    for _ in range(5):
        self.sim.render()
        time.sleep(0.02)
    
    # get current actual EE position
    actual_ee_positions = robot_manager.get_ee_positions_batch(self)
    
    # now restore object state, use EE position difference as offset
    for data in env_data:
        env_id = data["env_id"]
        env_tensor_id = torch.tensor([env_id], device=self.device)
        
        # get the origin of this environment
        env_origin = env_origins[env_id]
        
        # calculate offset (actual EE position - EE position in JSON)
        position_offset = torch.zeros(3, device=self.device)  # default no offset
        
        if data["ee_state"] is not None and actual_ee_positions is not None:
            # get EE position from JSON
            json_ee_pos = torch.tensor([
                data["ee_state"]["x"],
                data["ee_state"]["y"],
                data["ee_state"]["z"]
            ], device=self.device)
            
            # get actual EE position
            actual_ee_pos = actual_ee_positions[env_id]
            
            # calculate offset
            position_offset = actual_ee_pos - json_ee_pos
            print(f"environment {env_id} - EE position offset: {position_offset.cpu().numpy()}")
        else:
            print(f"environment {env_id} - cannot calculate EE offset, use zero offset")
        
        # restore object state
        for obj_info in data["objects_state"]:
            original_name = obj_info["name"]

            # try to get object from scene attributes
            obj = None
            if hasattr(self.scene, "rigid_objects"):
                obj = self.scene.rigid_objects[original_name]
            
            if obj is None:
                print(f"warning: cannot find object '{original_name}'")
                continue

            # extract position and orientation
            world_position = torch.tensor([
                obj_info["position"]["x"],
                obj_info["position"]["y"],
                obj_info["position"]["z"]
            ], device=self.device).unsqueeze(0)
            
            # apply EE offset to correct local position
            local_position = world_position + position_offset.unsqueeze(0)
            
            orientation = torch.tensor([
                obj_info["orientation"]["qx"],
                obj_info["orientation"]["qy"],
                obj_info["orientation"]["qz"],
                obj_info["orientation"]["qw"]
            ], device=self.device).unsqueeze(0)
            
            # set zero velocity
            lin_vel = torch.zeros_like(local_position)
            ang_vel = torch.zeros_like(local_position)
            
            # build root state tensor
            root_state = torch.cat([local_position, orientation, lin_vel, ang_vel], dim=1)

            # use write_root_state_to_sim method to set state
            obj.write_root_state_to_sim(root_state, env_tensor_id)

            # 新增: 恢复物体的质量（如果存在）
            if "mass" in obj_info and hasattr(obj, "root_physx_view"):
                try:
                    # 获取当前质量张量，以确保维度正确
                    current_masses = obj.root_physx_view.get_masses()
                    
                    # 创建新的质量张量，保持相同的形状，确保在CPU上
                    masses = torch.clone(current_masses)
                    masses[env_id] = float(obj_info["mass"])  # 设置单个环境的质量
                    
                    # 将张量和索引移至CPU
                    masses = masses.cpu() if hasattr(masses, 'cpu') else masses
                    cpu_env_tensor_id = env_tensor_id.cpu() if hasattr(env_tensor_id, 'cpu') else env_tensor_id
                    
                    # 应用质量设置
                    obj.root_physx_view.set_masses(masses, cpu_env_tensor_id)
                    print(f"environment {env_id} - object '{original_name}' mass set to {obj_info['mass']}")
                except Exception as e:
                    print(f"warning: failed to set mass for object '{original_name}': {e}")
            
            # 新增: 恢复物体的材料属性（如果存在）
            if "material" in obj_info and hasattr(obj, "root_physx_view"):
                try:
                    # 获取材料数据
                    material_data = obj_info["material"]
                    
                    # 获取当前材料属性张量
                    current_materials = obj.root_physx_view.get_material_properties()
                    
                    # 创建新的材料属性张量
                    materials = torch.clone(current_materials)
                    
                    # 提取材料属性值
                    static_friction = float(material_data["static_friction"])
                    dynamic_friction = float(material_data["dynamic_friction"])
                    restitution = float(material_data["restitution"])
                    
                    # 为物体的所有形状设置相同的材料属性
                    num_shapes = materials.size(1)  # 形状数量在第二个维度
                    for shape_idx in range(num_shapes):
                        materials[env_id, shape_idx, 0] = static_friction
                        materials[env_id, shape_idx, 1] = dynamic_friction
                        materials[env_id, shape_idx, 2] = restitution
                    
                    # 将张量和索引移至CPU
                    materials = materials.cpu() if hasattr(materials, 'cpu') else materials
                    cpu_env_tensor_id = env_tensor_id.cpu() if hasattr(env_tensor_id, 'cpu') else env_tensor_id
                    
                    # 应用材料属性设置
                    obj.root_physx_view.set_material_properties(materials, cpu_env_tensor_id)
                    print(f"environment {env_id} - object '{original_name}' material properties set: "
                        f"static_friction={static_friction}, dynamic_friction={dynamic_friction}, "
                        f"restitution={restitution}")
                except Exception as e:
                    print(f"warning: failed to set material properties for object '{original_name}': {e}")

            # asset.root_physx_view.set_masses(masses, env_ids)

            print(f"environment {env_id} - object '{original_name}' state restored (sample {data['sample_idx']})")
    
    # render several frames to apply all states
    print("render frames to apply object states...")
    for _ in range(5):
        self.sim.render()
        time.sleep(0.02)
    
    # mark state restored
    _STATE_RESTORED = True
    print(f"batch {current_batch_index + 1}/{total_batches} restored!")
    return env_cfg

def create_environment(args):
    """create environment"""
    # parse environment configuration
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.env_name = args.task
    
    # set environment number
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
        env_cfg.scene.num_envs = args.num_envs
    
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
        env_cfg.sim.num_envs = args.num_envs
    
    # modify configuration
    env_cfg.terminations.time_out = None

    if "Lift" in args.task:
        # set resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    
    # ensure events configuration exists
    if not hasattr(env_cfg, 'events'):
        env_cfg.events = type('EventsCfg', (), {})()
    
    # register state restore event handler
    env_cfg.events.restore_state = EventTerm(
        func=restore_state_from_json,
        mode="reset",  # trigger when environment reset
        params={}  # no extra parameters
    )
    
    # create environment
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    
    # reset environment - this will trigger event processing system
    env.reset()
    
    # render several frames
    for _ in range(5):
        env.sim.render()
        time.sleep(0.02)
    
    return env

# enhanced version of safety violation detection function
def check_safety_states(env):
    """检查安全状态并记录详细的安全信息（倾斜角度小于15度的对象）"""
    global all_safe_transitions, env_object_mappings
    
    safe_states = []
    
    # 获取对象状态
    object_states, object_names = env_manager.get_object_states_batch(env)
    
    # 检查每个环境中的每个对象
    for env_idx in range(env.num_envs):
        env_safe_states = []
        
        for obj_idx, name in enumerate(object_names):
            # 计算倾斜角度
            obj_quat = object_states[env_idx, obj_idx, 3:7]
            tilt_angle_rad = geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat.unsqueeze(0)).item()
            tilt_angle_deg = np.degrees(tilt_angle_rad)
            
            # check whether tilt angle is less than 15
            if tilt_angle_deg < 15.0:
            # if tilt_angle_deg < 15.0:
               
                orig_data = None
                if env_idx in env_object_mappings and name in env_object_mappings[env_idx]:
                    orig_data = env_object_mappings[env_idx][name]
                
                safe_state = {
                    "object_name": name,
                    "tilt_angle_deg": float(f"{tilt_angle_deg:.2f}"),
                    "env_idx": env_idx,
                    "orig_index": env_object_mappings[env_idx][name]["orig_index"] if env_idx in env_object_mappings and name in env_object_mappings[env_idx] else None,
                    "sample_idx": env_object_mappings[env_idx][name]["sample_idx"] if env_idx in env_object_mappings and name in env_object_mappings[env_idx] else None
                }
                env_safe_states.append(safe_state)
                print(f"object '{name}' tilt angle {tilt_angle_deg:.2f}° < 15° threshold, now is safe!")
        
        if env_safe_states:
            safe_states.append({
                "env_idx": env_idx,
                "safe_states": env_safe_states
            })
    
    # 添加当前批次安全状态到全局列表
    all_safe_transitions.extend(safe_states)
    
    return safe_states

# new function: find samples in safe_local_xs
def find_samples_in_unsafe_local_xs(safe_state_info):
    """
    在unsafe_local_xs中查找变成安全状态的样本
    
    参数:
    - safe_state_info: 安全状态信息列表
    
    返回:
    - positions_to_move: 需要从unsafe移动到safe的样本
    """
    global indices
    
    positions_to_move = []
    mapping_details = []
    
    for safe_state in safe_state_info:
        env_idx = safe_state["env_idx"]
        
        for obj_safe_state in safe_state["safe_states"]:
            obj_name = obj_safe_state["object_name"]
            
            # 获取原始索引
            if env_idx in env_object_mappings and obj_name in env_object_mappings[env_idx]:
                orig_index = env_object_mappings[env_idx][obj_name]["orig_index"]
                
                # 在unsafe_indices和unsafe_obj_names中查找匹配项
                for i, (idx, name) in enumerate(zip(indices["unsafe_indices"], indices["unsafe_obj_names"])):
                    if idx == orig_index and name == obj_name:
                        positions_to_move.append(i)
                        mapping_details.append({
                            "position": i,
                            "orig_index": int(orig_index),
                            "object_name": obj_name,
                            "tilt_angle": obj_safe_state["tilt_angle_deg"]
                        })
                        print(f"found safe sample: position={i}, index={orig_index}, object={obj_name}, tilt angle={obj_safe_state['tilt_angle_deg']}°")
    
    return list(set(positions_to_move)), mapping_details  # remove duplicates

def update_safety_labels(positions_to_move):
    """
    move samples from unsafe to safe
    
    parameters:
    - positions_to_move: samples to move from unsafe to safe
    
    return:
    - updated data and indices
    """
    global data, indices
    
    if not positions_to_move:
        print("no samples to update")
        return data, indices
    
    print(f"\n===== update safety labels =====")
    print(f"need to move {len(positions_to_move)} samples from unsafe to safe")
    
    # 创建新的数据结构
    new_safe_local_xs = list(data["safe_local_xs"])
    new_safe_local_nxs = list(data["safe_local_nxs"]) if "safe_local_nxs" in data else []
    new_safe_ee_xs = list(data["safe_ee_xs"]) if "safe_ee_xs" in data else []
    new_safe_ee_nxs = list(data["safe_ee_nxs"]) if "safe_ee_nxs" in data else []
    new_safe_indices = list(indices["safe_indices"])
    new_safe_obj_names = list(indices["safe_obj_names"])
    
    new_unsafe_local_xs = []
    new_unsafe_local_nxs = []
    new_unsafe_ee_xs = []
    new_unsafe_ee_nxs = []
    new_unsafe_indices = []
    new_unsafe_obj_names = []
    
    # 重组数据
    for i in range(len(data["unsafe_local_xs"])):
        if i in positions_to_move:
            # 移动到safe
            new_safe_local_xs.append(data["unsafe_local_xs"][i])
            # 现在可以直接使用对应的next state数据
            if "unsafe_local_nxs" in data and i < len(data["unsafe_local_nxs"]):
                new_safe_local_nxs.append(data["unsafe_local_nxs"][i])
            
            if "unsafe_ee_xs" in data and i < len(data["unsafe_ee_xs"]):
                new_safe_ee_xs.append(data["unsafe_ee_xs"][i])
            
            # 现在可以直接使用对应的ee next state数据
            if "unsafe_ee_nxs" in data and i < len(data["unsafe_ee_nxs"]):
                new_safe_ee_nxs.append(data["unsafe_ee_nxs"][i])
            
            new_safe_indices.append(indices["unsafe_indices"][i])
            new_safe_obj_names.append(indices["unsafe_obj_names"][i])
            print(f"sample {i} (object: {indices['unsafe_obj_names'][i]}, index: {indices['unsafe_indices'][i]}) is relabeled as safe")
        else:
            # 保留在unsafe
            new_unsafe_local_xs.append(data["unsafe_local_xs"][i])
            if "unsafe_local_nxs" in data and i < len(data["unsafe_local_nxs"]):
                new_unsafe_local_nxs.append(data["unsafe_local_nxs"][i])
            if "unsafe_ee_xs" in data and i < len(data["unsafe_ee_xs"]):
                new_unsafe_ee_xs.append(data["unsafe_ee_xs"][i])
            if "unsafe_ee_nxs" in data and i < len(data["unsafe_ee_nxs"]):
                new_unsafe_ee_nxs.append(data["unsafe_ee_nxs"][i])
            new_unsafe_indices.append(indices["unsafe_indices"][i])
            new_unsafe_obj_names.append(indices["unsafe_obj_names"][i])
    
    # 更新数据结构
    data["safe_local_xs"] = np.array(new_safe_local_xs, dtype=object)
    data["safe_local_nxs"] = np.array(new_safe_local_nxs, dtype=object)
    data["safe_ee_xs"] = np.array(new_safe_ee_xs)
    data["safe_ee_nxs"] = np.array(new_safe_ee_nxs)
    
    data["unsafe_local_xs"] = np.array(new_unsafe_local_xs, dtype=object)
    data["unsafe_local_nxs"] = np.array(new_unsafe_local_nxs, dtype=object)
    data["unsafe_ee_xs"] = np.array(new_unsafe_ee_xs)
    data["unsafe_ee_nxs"] = np.array(new_unsafe_ee_nxs)
    
    indices["safe_indices"] = np.array(new_safe_indices)
    indices["unsafe_indices"] = np.array(new_unsafe_indices)
    indices["safe_obj_names"] = np.array(new_safe_obj_names)
    indices["unsafe_obj_names"] = np.array(new_unsafe_obj_names)
    
    print(f"update completed: {len(positions_to_move)} samples moved from unsafe to safe")
    print(f"updated data statistics:")
    print(f"  safe samples: {len(new_safe_local_xs)}")
    print(f"  safe_local_nxs: {len(new_safe_local_nxs)}")
    print(f"  safe_ee_xs: {len(new_safe_ee_xs)}")
    print(f"  safe_ee_nxs: {len(new_safe_ee_nxs)}")
    print(f"  unsafe samples: {len(new_unsafe_local_xs)}")
    print(f"  unsafe_local_nxs: {len(new_unsafe_local_nxs)}")
    print(f"  unsafe_ee_xs: {len(new_unsafe_ee_xs)}")
    print(f"  unsafe_ee_nxs: {len(new_unsafe_ee_nxs)}")
    
    return data, indices


def save_updated_dataset(data, indices, mapping_details, output_dir):
    """
    save updated dataset and mapping details
    
    parameters:
    - data: updated data
    - indices: updated indices
    - mapping_details: mapping details
    - output_dir: output directory
    """
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # save dataset
    dataset_path = os.path.join(output_dir, "updated_dataset.npz")
    np.savez(
        dataset_path,
        safe_local_xs=data["safe_local_xs"],
        safe_local_nxs=data["safe_local_nxs"] if "safe_local_nxs" in data else [],
        safe_ee_xs=data["safe_ee_xs"] if "safe_ee_xs" in data else [],
        safe_ee_nxs=data["safe_ee_nxs"] if "safe_ee_nxs" in data else [],
        unsafe_local_xs=data["unsafe_local_xs"],
        unsafe_ee_xs=data["unsafe_ee_xs"] if "unsafe_ee_xs" in data else [],
        safe_indices=indices["safe_indices"],
        unsafe_indices=indices["unsafe_indices"],
        safe_obj_names=indices["safe_obj_names"],
        unsafe_obj_names=indices["unsafe_obj_names"]
    )
    
    # save mapping details
    mapping_path = os.path.join(output_dir, "relabeled_samples.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping_details, f, indent=2)
    
    # save statistics
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    stats = {
        "safe_samples": len(data["safe_local_xs"]),
        "unsafe_samples": len(data["unsafe_local_xs"]),
        "total_samples": len(data["safe_local_xs"]) + len(data["unsafe_local_xs"]),
        "relabeled_samples": len(mapping_details),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\ndataset saved to {dataset_path}")
    print(f"relabeled samples details saved to {mapping_path}")
    print(f"dataset statistics saved to {stats_path}")

def add_to_training_buffer(batch_data, batch_indices, buffer_size=5):
    """add current batch data to training buffer
    
    parameters:
    - batch_data: current batch data
    - batch_indices: current batch indices
    - buffer_size: maximum capacity of buffer (number of batches)
    
    return:
    - updated buffer
    """
    global training_buffer, buffer_indices, current_buffer_size, max_buffer_size
    
    # update maximum buffer size
    max_buffer_size = buffer_size
    
    # check if buffer is full
    if current_buffer_size >= max_buffer_size:
        # remove oldest batch data
        for key in training_buffer:
            if len(training_buffer[key]) > 0:
                # calculate average size of each batch
                batch_len = len(training_buffer[key]) // current_buffer_size
                # remove oldest batch data
                training_buffer[key] = training_buffer[key][batch_len:]
        
        for key in buffer_indices:
            if len(buffer_indices[key]) > 0:
                # calculate average size of each batch
                batch_len = len(buffer_indices[key]) // current_buffer_size
                # remove oldest batch data
                buffer_indices[key] = buffer_indices[key][batch_len:]
        
        # reduce current buffer size count
        current_buffer_size -= 1
    
    # add current batch data to buffer
    for key in batch_data:
        if key in training_buffer:
            if len(batch_data[key]) > 0:
                if len(training_buffer[key]) == 0:
                    # if buffer is empty, assign directly
                    training_buffer[key] = batch_data[key]
                else:
                    # check type and merge appropriately
                    if isinstance(training_buffer[key], np.ndarray):
                        # if buffer is NumPy array, use concatenate
                        training_buffer[key] = np.concatenate([training_buffer[key], batch_data[key]])
                    elif isinstance(training_buffer[key], list):
                        # if buffer is list, check batch data type
                        if isinstance(batch_data[key], np.ndarray):
                            # if batch data is NumPy array, convert buffer to array first
                            training_buffer[key] = np.concatenate([np.array(training_buffer[key]), batch_data[key]])
                        else:
                            # if both are lists, use extend
                            training_buffer[key].extend(batch_data[key])
    
    # perform same operation on index data
    for key in batch_indices:
        if key in buffer_indices:
            if len(batch_indices[key]) > 0:
                if len(buffer_indices[key]) == 0:
                    buffer_indices[key] = batch_indices[key]
                else:
                    if isinstance(buffer_indices[key], np.ndarray):
                        buffer_indices[key] = np.concatenate([buffer_indices[key], batch_indices[key]])
                    elif isinstance(buffer_indices[key], list):
                        if isinstance(batch_indices[key], np.ndarray):
                            buffer_indices[key] = np.concatenate([np.array(buffer_indices[key]), batch_indices[key]])
                        else:
                            buffer_indices[key].extend(batch_indices[key])
    
    # increase current buffer size count
    current_buffer_size += 1
    
    print(f"current buffer state: {current_buffer_size}/{max_buffer_size} batches")
    for key in training_buffer:
        print(f"  {key}: {len(training_buffer[key])} samples")
    
    return training_buffer, buffer_indices

def extract_current_batch_data(global_data, global_indices, positions_to_move=None, extract_all=False, batch_indices=None):
    """extract training data from current batch
    
    parameters:
    - global_data: global data
    - global_indices: global indices
    - positions_to_move: positions to move from safe to unsafe
    - extract_all: 是否提取所有当前批次数据（而不仅是违规样本）
    - batch_indices: 当前批次的原始索引列表（用于extract_all=True时）
    
    返回:
    - 当前批次数据
    - 当前批次索引
    """
    # 复制数据，避免修改原始数据
    batch_data = {}
    batch_indices_out = {}
    
    # 如果要提取所有数据
    if extract_all and batch_indices is not None:
        # 初始化数据结构
        batch_data["safe_local_xs"] = []
        batch_data["safe_local_nxs"] = []
        batch_data["safe_ee_xs"] = []
        batch_data["safe_ee_nxs"] = []
        batch_data["unsafe_local_xs"] = []
        batch_data["unsafe_ee_xs"] = []
        batch_indices_out["safe_indices"] = []
        batch_indices_out["unsafe_indices"] = []
        batch_indices_out["safe_obj_names"] = []
        batch_indices_out["unsafe_obj_names"] = []
        
        # 从全局数据中提取属于当前批次的所有样本
        for i, idx in enumerate(global_indices["safe_indices"]):
            if idx in batch_indices:
                # 添加安全样本
                batch_data["safe_local_xs"].append(global_data["safe_local_xs"][i])
                if "safe_local_nxs" in global_data and i < len(global_data["safe_local_nxs"]):
                    batch_data["safe_local_nxs"].append(global_data["safe_local_nxs"][i])
                if "safe_ee_xs" in global_data and i < len(global_data["safe_ee_xs"]):
                    batch_data["safe_ee_xs"].append(global_data["safe_ee_xs"][i])
                if "safe_ee_nxs" in global_data and i < len(global_data["safe_ee_nxs"]):
                    batch_data["safe_ee_nxs"].append(global_data["safe_ee_nxs"][i])
                batch_indices_out["safe_indices"].append(idx)
                batch_indices_out["safe_obj_names"].append(global_indices["safe_obj_names"][i])
        
        for i, idx in enumerate(global_indices["unsafe_indices"]):
            if idx in batch_indices:
                # 添加不安全样本
                batch_data["unsafe_local_xs"].append(global_data["unsafe_local_xs"][i])
                if "unsafe_ee_xs" in global_data and i < len(global_data["unsafe_ee_xs"]):
                    batch_data["unsafe_ee_xs"].append(global_data["unsafe_ee_xs"][i])
                batch_indices_out["unsafe_indices"].append(idx)
                batch_indices_out["unsafe_obj_names"].append(global_indices["unsafe_obj_names"][i])
        
        # 转换为NumPy数组
        for key in batch_data:
            if len(batch_data[key]) > 0:
                if "xs" in key:  # 对于所有特征数据
                    batch_data[key] = np.array(batch_data[key], dtype=object if "local" in key else np.float32)
        
        for key in batch_indices_out:
            if len(batch_indices_out[key]) > 0:
                batch_indices_out[key] = np.array(batch_indices_out[key])
        
        return batch_data, batch_indices_out
    
    # 如果没有指定位置，则返回空数据
    if (positions_to_move is None or len(positions_to_move) == 0) and not extract_all:
        # 创建空的数据结构
        for key in global_data:
            if isinstance(global_data[key], np.ndarray):
                batch_data[key] = np.array([], dtype=global_data[key].dtype)
            else:
                batch_data[key] = []
        
        for key in global_indices:
            if isinstance(global_indices[key], np.ndarray):
                batch_indices_out[key] = np.array([], dtype=global_indices[key].dtype)
            else:
                batch_indices_out[key] = []
        
        return batch_data, batch_indices_out
    
    # 提取更新后的unsafe样本
    batch_data["unsafe_local_xs"] = []
    batch_data["unsafe_ee_xs"] = []
    batch_indices_out["unsafe_indices"] = []
    batch_indices_out["unsafe_obj_names"] = []
    
    # 提取被移动的样本(从safe变为unsafe的样本)
    for pos in positions_to_move:
        batch_data["unsafe_local_xs"].append(global_data["safe_local_xs"][pos])
        if "safe_ee_xs" in global_data and pos < len(global_data["safe_ee_xs"]):
            batch_data["unsafe_ee_xs"].append(global_data["safe_ee_xs"][pos])
        batch_indices_out["unsafe_indices"].append(global_indices["safe_indices"][pos])
        batch_indices_out["unsafe_obj_names"].append(global_indices["safe_obj_names"][pos])
    
    # 转换为numpy数组
    batch_data["unsafe_local_xs"] = np.array(batch_data["unsafe_local_xs"], dtype=object)
    batch_data["unsafe_ee_xs"] = np.array(batch_data["unsafe_ee_xs"])
    batch_indices_out["unsafe_indices"] = np.array(batch_indices_out["unsafe_indices"])
    batch_indices_out["unsafe_obj_names"] = np.array(batch_indices_out["unsafe_obj_names"])
    
    return batch_data, batch_indices_out

def process_batch(env):
    """处理当前批次并立即更新安全标签，然后进行模型训练"""
    global current_batch_index, batch_results, all_violations, data, indices
    global model, h_obj_normalizer, ee_normalizer, best_performance, performance_history
    global training_buffer, buffer_indices, current_buffer_size, max_buffer_size
    global boundary_samples_data
    
    print(f"\n====== batch {current_batch_index + 1}/{total_batches} safety status ======")
    
    # 获取当前批次的原始索引
    start_idx = current_batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(boundary_samples_data))
    current_batch_orig_indices = []
    for i in range(start_idx, end_idx):
        if i < len(boundary_samples_data) and "orig_index" in boundary_samples_data[i]:
            current_batch_orig_indices.append(boundary_samples_data[i]["orig_index"])
    
    # print ee position
    ee_positions = robot_manager.get_ee_positions_batch(env)
    if ee_positions is not None:
        print("\ninitial end-effector positions:")
        for env_idx in range(min(5, env.num_envs)):  
            pos = ee_positions[env_idx].cpu().numpy()
            print(f"env {env_idx} - position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        if env.num_envs > 5:
            print(f"...and {env.num_envs - 5} more environments")
    
    print("\n====== start end-effector NCBF safety movement test ======")
    num_move_steps = 4  # number of movement steps
    move_step_size = 0.01  # maximum step size
    grid_size = 8  # action space grid size
    safety_threshold = 0.05  # safety threshold

    # record active environments
    active_envs = list(range(env.num_envs))

    # reset history buffer
    object_history_buffer = {}

    # get object states
    object_states, object_names = env_manager.get_object_states_batch(env)

    # initialize object history buffer
    print("initialize object history buffer...")
    for _ in range(args_cli.horizon):
        update_object_history_buffer(env, object_states, object_names)

    for step in range(num_move_steps):
        print(f"\nstep {step+1}/{num_move_steps}")
        
        # get all object states
        object_states, object_names = env_manager.get_object_states_batch(env)
        
        # calculate the distance to the objects, get the nearest object information
        distances_tensor, object_names_list, closest_info = robot_manager.calculate_distances_to_objects_batch(env)
        
        actions = torch.zeros((env.num_envs, 3), device=env.device)  # 3 = ACTION_DIM
        
        for env_idx in active_envs:
            # get the nearest object
            if env_idx in closest_info:
                closest = closest_info[env_idx]
                closest_distance = closest['distance']
                closest_name = closest['name']
                
                # find the corresponding index in the object list
                if closest_name in object_names_list:
                    obj_idx = object_names_list.index(closest_name)
                    
                    # get the object position and end-effector position
                    obj_pos = object_states[env_idx, obj_idx, :3]
                    current_ee_pos = ee_positions[env_idx]
                    
                    # use NCBF method to calculate the safe action
                    action, distance = move_ee_towards_object(
                        env,
                        current_ee_pos,
                        obj_pos,
                        env_idx,
                        obj_idx,
                        object_states,
                        object_names,
                        model,  # NCBF模型
                        h_obj_normalizer,  # 物体历史特征归一化器
                        ee_normalizer,  # End Effector位置归一化器
                        move_step_size=move_step_size,
                        grid_size=grid_size,
                        safety_threshold=safety_threshold
                    )
                    
                    actions[env_idx] = action

                    if step == 0 or step == num_move_steps - 1 or env_idx % 20 == 0:
                        print(f"env {env_idx} - target: {closest_name} - distance: {distance:.4f}")
                else:
                    print(f"env {env_idx} - cannot find object {closest_name}")
            else:
                print(f"env {env_idx} - cannot find nearest object information")
        
        # execute the action
        env.step(actions)
        
        # 更新末端执行器位置
        ee_positions = robot_manager.get_ee_positions_batch(env)
        
        # 获取更新后的物体状态
        object_states, object_names = env_manager.get_object_states_batch(env)
        
        # 更新历史状态缓冲区
        update_object_history_buffer(env, object_states, object_names)
        
        # 渲染
        env.sim.render()
        time.sleep(0.05)
    
    print("end-effector movement test completed")
    
    # print the end-effector position after movement
    if ee_positions is not None:
        print("\nafter movement, end-effector position:")
        for env_idx in range(min(5, env.num_envs)):  # only show the first 5 environments
            pos = ee_positions[env_idx].cpu().numpy()
            print(f"env {env_idx} - position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    # check the safety status
    safe_states = check_safety_states(env)
    
    # record the batch results
    batch_results[current_batch_index] = {
        "batch_index": current_batch_index,
        "safe_states": safe_states,
        "has_safe_transition": len(safe_states) > 0
    }

    
    # 保存当前批次违规记录
    positions_to_move = []
    mapping_details = []
    
    if safe_states:
        safe_states_file = os.path.join(args_cli.output_dir, f"batch_{current_batch_index+1}_safe_states.json")
        with open(safe_states_file, 'w') as f:
            json.dump(safe_states, f, indent=2)
        print(f"\nbatch {current_batch_index+1}: found {len(safe_states)} environments with objects becoming safe, saved to {safe_states_file}")
        
        # immediately update the safety labels for the current batch
        print(f"\n===== update the safety labels for batch {current_batch_index+1} =====")
        
        # 查找需要更新的样本
        positions_to_move, mapping_details = find_samples_in_unsafe_local_xs(safe_states)
        
        # 更新安全标签
        if positions_to_move:
            data, indices = update_safety_labels(positions_to_move)
            
            # 保存当前批次更新后的数据集
            batch_mapping_path = os.path.join(args_cli.output_dir, f"relabeled_samples_batch_{current_batch_index+1}.json")
            
            # 只保存更新信息，不保存完整数据集(以减少存储开销)
            with open(batch_mapping_path, 'w') as f:
                json.dump(mapping_details, f, indent=2)
            
            print(f"batch {current_batch_index+1}: moved {len(positions_to_move)} samples from unsafe to safe")
            print(f"relabeled samples details saved to {batch_mapping_path}")
        else:
            print(f"batch {current_batch_index+1}: no samples need to be updated")
    else:
        print(f"\nbatch {current_batch_index+1}: no environments with objects becoming safe")
    
    print(f"\n===== use batch {current_batch_index+1} data to train model =====")
    
    if args_cli.use_buffer:

        # 提取当前批次所有数据 (新增)
        batch_data, batch_indices_out = extract_current_batch_data(
            data, indices, positions_to_move=None, 
            extract_all=True, batch_indices=current_batch_orig_indices
        )
        
        # 添加到缓冲区
        training_buffer, buffer_indices = add_to_training_buffer(batch_data, batch_indices_out, args_cli.buffer_size)
        
        # 使用缓冲区中的数据训练
        if current_buffer_size > 0:
            # 提取特征
            training_features = extract_training_features(training_buffer)
            print(f"use buffer data to train: {current_buffer_size}/{max_buffer_size} batches")
            print(f"buffer safe samples: {len(training_buffer.get('safe_local_xs', []))} , unsafe samples: {len(training_buffer.get('unsafe_local_xs', []))} ")
        else:
            print("buffer is empty, skip training")
            current_batch_index += 1
            return
    else:
        # 使用全局数据
        training_features = extract_training_features(data)
        print(f"use global data to train: {len(data['safe_local_xs'])} safe samples, {len(data['unsafe_local_xs'])} unsafe samples")
    
    # 检查是否有足够的数据进行训练
    if all(k in training_features for k in ["safe_h_obj_xs", "safe_ee_xs", "unsafe_h_obj_xs", "unsafe_ee_xs"]):
        if len(training_features["safe_h_obj_xs"]) >= 10 and len(training_features["unsafe_h_obj_xs"]) >= 10:
            # 归一化数据
            normalized_data = normalize_training_data(training_features, h_obj_normalizer, ee_normalizer)
            
            if normalized_data:
                # 获取设备
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # 显示训练数据统计
                print(f"safe samples: {len(training_features['safe_h_obj_xs'])}")
                print(f"unsafe samples: {len(training_features['unsafe_h_obj_xs'])}")
                
                # 保存训练前的模型状态，以便在性能下降时恢复
                model_state_before = model.state_dict()
                
                # 训练模型
                model.fit(
                    data=normalized_data,
                    epoch=args_cli.training_epochs,
                    verbose_num=1,
                    lr=args_cli.learning_rate,
                    margin_threshold=args_cli.margin_threshold,
                    derivative_threshold=args_cli.derivative_threshold,
                    batch_size=args_cli.training_batch_size,
                    device=device
                )
                
                # 评估模型性能
                performance = evaluate_model_performance(model, normalized_data)
                
                # 记录性能历史
                batch_performance = {
                    "batch_index": current_batch_index,
                    "performance": float(performance),  # 转换为Python基本类型以便JSON序列化
                    "safe_samples": len(training_features["safe_h_obj_xs"]),
                    "unsafe_samples": len(training_features["unsafe_h_obj_xs"]),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                performance_history.append(batch_performance)
                
                # 保存性能历史
                performance_history_path = os.path.join(args_cli.output_dir, "performance_history.json")
                with open(performance_history_path, 'w') as f:
                    json.dump(performance_history, f, indent=2)
                
                # 保存最佳模型
                if performance > best_performance:
                    best_performance = performance
                    best_model_path = os.path.join(args_cli.output_dir, args_cli.best_model_path)
                    model.save(best_model_path)
                    print(f"batch {current_batch_index+1}: new best model saved, performance: {performance:.4f}")
                else:
                    print(f"batch {current_batch_index+1}: performance not improved (current: {performance:.4f}, best: {best_performance:.4f})")
                    # if performance decreases too much, you can choose to restore the previous model state
                    if performance < best_performance - 0.1:  # allow some fluctuation
                        print("performance significantly decreased, restore the previous model state")
                        model.load_state_dict(model_state_before)
                
                # 保存当前批次的模型
                batch_model_path = os.path.join(args_cli.output_dir, f"model_batch_{current_batch_index+1}.pt")
                model.save(batch_model_path)
            else:
                print("warning: normalized data is empty, skip the current batch training")
        else:
            print(f"warning: training samples insufficient (safe: {len(training_features.get('safe_h_obj_xs', []))}, unsafe: {len(training_features.get('unsafe_h_obj_xs', []))}), skip the current batch training")
    else:
        print("警告: 缺少必要的训练特征，跳过当前批次的训练")
    
    # 增加批次索引
    current_batch_index += 1


def proceed_to_next_batch(env):
    """处理下一批次"""
    global current_batch_index, total_batches
    
    if current_batch_index >= total_batches:
        print("\nall the batches are processed!")
        return False
    
    print(f"\n====== prepare to process next batch {current_batch_index + 1}/{total_batches} ======")
    
    env.reset()
    
    # render
    for _ in range(5):
        env.sim.render()
        time.sleep(0.02)
    
    return True

def main():
    """主函数"""
    global boundary_samples_data, tilt_violations, env_snapshot_time_indices
    global current_batch_index, total_batches, batch_size, batch_results, env_instance
    global data, indices, all_violations
    global model, h_obj_normalizer, ee_normalizer 
    global best_performance, performance_history  
    
    # 初始化批次变量
    current_batch_index = 0
    batch_size = args_cli.num_envs  # 每批处理的样本数量等于环境数量
    batch_results = {}
    all_violations = []
    
    # 初始化训练相关变量
    best_performance = -float('inf')
    performance_history = []
    
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    # load json
    json_file_path = args_cli.json_file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    print(f"load {len(json_data)} json data")
    
    print("\n====== calculate CBF value and choose boundary samples ======")
    
    # 加载模型和归一化器 (设为全局变量，以便在训练时使用)
    model, h_obj_normalizer, ee_normalizer = load_model(
        args_cli.model_path, args_cli.h_obj_normalizer_path, args_cli.ee_normalizer_path
    )
    
    
    organized = organize_by_object(json_data, 30)
    print(f"samplea are organized by objects,  {len(organized)} objects in total")
    
    # 构建序列
    data, indices = build_sequences(organized, args_cli.horizon, args_cli.step, args_cli.min_seq_len)
    
    print(f"generated {len(data['safe_local_xs'])} safe samples")
    
    # 准备特征数据
    features_data = prepare_features_data(data)
    
    # 计算CBF值
    all_cbf_values = calculate_cbf_values(model, h_obj_normalizer, ee_normalizer, features_data)
    
    # 提取边界样本
    boundary_info = extract_boundary_samples(features_data, indices, all_cbf_values, args_cli.cbf_min, args_cli.cbf_max)
    print(f"find {len(boundary_info)} samples near safety bounsary")
    
    # 将边界样本的原始数据填充到全局变量
    boundary_samples_data = []
    for sample in boundary_info:
        orig_index = sample["orig_index"]
        if orig_index < len(json_data):
            # 在全局变量中存储完整的样本数据
            boundary_samples_data.append(json_data[orig_index])
            
            # 添加CBF值到样本数据
            boundary_samples_data[-1]["cbf_value"] = sample["cbf_value"]
            boundary_samples_data[-1]["orig_index"] = orig_index  # 确保原始索引被保存
            print(f"ass samples {orig_index}，CBF value: {sample['cbf_value']:.6f}")
    
    # 计算总批次数
    total_batches = (len(boundary_samples_data) + batch_size - 1) // batch_size
    print(f"\nextracted {len(boundary_samples_data)} boundary samples in tortal，in {total_batches} batches")
    
    # 保存边界样本索引
    boundary_indices_path = os.path.join(args_cli.output_dir, "boundary_indices.json")
    with open(boundary_indices_path, 'w') as f:
        boundary_indices = [sample["orig_index"] for sample in boundary_info]
        json.dump(boundary_indices, f, indent=2)
    
    print(f"boundary indices are saves to {boundary_indices_path}")
    
    print("\n====== start to reconstruct scenes ======")
    
    # 清除场景
    clear_scene()
    
    # 创建环境 - 这将通过事件系统自动恢复第一批状态
    env_instance = create_environment(args_cli)
    
    # 处理第一批
    process_batch(env_instance)
    
    # 固定动作维度 (x, y, gripper)
    ACTION_DIM = 3
    
    # 批次处理循环
    try:
        while current_batch_index < total_batches:

            print(f"\nwait {args_cli.batch_interval} second to process next batch...")
            
            # 维持当前场景渲染
            for _ in range(int(args_cli.batch_interval * 10)):  # 10FPS
                env_instance.step(torch.zeros((env_instance.num_envs, ACTION_DIM), device=env_instance.device))
                env_instance.sim.render()
                time.sleep(0.1)
            
            # 处理下一批次
            if proceed_to_next_batch(env_instance):
                process_batch(env_instance)  # 处理批次并立即更新标签
            else:
                break
            
        # 保存总结果
        final_output_path = os.path.join(args_cli.output_dir, "final_results.json")
        with open(final_output_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nall batches are processed, results saved to {final_output_path}")
        
        print("\n===== saved updated dataset =====")
        final_dataset_path = os.path.join(args_cli.output_dir, "final_updated_dataset.npz")
        np.savez(
            final_dataset_path,
            safe_local_xs=data["safe_local_xs"],
            safe_local_nxs=data["safe_local_nxs"] if "safe_local_nxs" in data else [],
            safe_ee_xs=data["safe_ee_xs"] if "safe_ee_xs" in data else [],
            safe_ee_nxs=data["safe_ee_nxs"] if "safe_ee_nxs" in data else [],
            unsafe_local_xs=data["unsafe_local_xs"],
            unsafe_ee_xs=data["unsafe_ee_xs"] if "unsafe_ee_xs" in data else [],
            safe_indices=indices["safe_indices"],
            unsafe_indices=indices["unsafe_indices"],
            safe_obj_names=indices["safe_obj_names"],
            unsafe_obj_names=indices["unsafe_obj_names"]
        )
        
        # 保存统计信息
        stats_path = os.path.join(args_cli.output_dir, "final_dataset_stats.json")
        total_relabeled = sum(len(result["safe_states"]) for result in batch_results.values() if result["has_safe_transition"])
        stats = {
            "safe_samples": len(data["safe_local_xs"]),
            "unsafe_samples": len(data["unsafe_local_xs"]),
            "total_samples": len(data["safe_local_xs"]) + len(data["unsafe_local_xs"]),
            "relabeled_samples_total": total_relabeled,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"final dataset saved to {final_dataset_path}")
        print(f"dataset statistical information saved to {stats_path}")
        
        print("\n====== refinement training done, process ctrl C to quit ======")
        
        while True:
            env_instance.step(torch.zeros((env_instance.num_envs, ACTION_DIM), device=env_instance.device))
            env_instance.sim.render()
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nuser interupt, quit")
    
    finally:
       
        if env_instance:
            env_instance.close()
        print("environments are closed")


if __name__ == "__main__":
    
    main()

    simulation_app.close()