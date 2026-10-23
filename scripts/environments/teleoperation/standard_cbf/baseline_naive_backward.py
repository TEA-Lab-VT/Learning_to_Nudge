# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="NCBF-guided exploration for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-Eval-v0", help="Name of the task.")
parser.add_argument("--max_tipping_angle", type=float, default=15.0, help="Maximum tipping angle for safety (degrees).")
parser.add_argument("--max_steps", type=int, default=1500, help="Maximum steps per exploration run.")
parser.add_argument("--total_runs", type=int, default=1, help="Total number of exploration runs.")
parser.add_argument("--noise_magnitude", type=float, default=0.005, help="Action noise magnitude.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=2222, 
    help="Random seed for reproducibility."
)
parser.add_argument(
    "--scene_type", 
    type=str, 
    default="small", 
    choices=["small", "medium", "large", "extra_large"],
    help="Scene type determining number of objects (small=4, medium=10, large=20, extra_large=40)"
)
parser.add_argument(
    "--safe_mode",
    action="store_true",
    default=True,
    help="Enable safe mode for exploration to prevent safety violations."
)
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
import random
import math
from copy import deepcopy
from time import time
import os
from collections import deque
import datetime
import json
"""Rest everything follows."""

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.managers import TerminationTermCfg as DoneTerm

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect_data import start_new_trajectory, collect_all_environments_trajectory, end_current_trajectory

# from models import NCBF, Normalizer

# Initialize global manager instance
env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()

DEFAULT_PUSH_HEIGHT = 0.19  # push height
OBJECT_RADIUS = 0.05  # object radius
SAMPLING_RADIUS_FACTOR = 1.5 # sampling radius factor   1.5
MOVE_STEP_SIZE = 0.01  # move step size 
SAMPLING_THRESHOLD = 0.05  # reach the sampling point threshold
SAFETY_THRESHOLD_FACTOR = 1.0  # safety threshold factor
BACKUP_SPEED_FACTOR = 1.0  # backup speed factor
MIN_VELOCITY = 0.01  # meter / step
MAX_VELOCITY = 0.01  # meter / step
VELOCITY_CHANGE_INTERVAL = 5  # update velocity every 5 steps

# Note that every 'velocity' is actually: meter / step

# statistic data
exploration_data = {
    'trajectories': [],
    'safety_violations': 0,
    'total_steps': 0,
    'ncbf_preventions': 0 ,
    'trajectory_stats': [] 
}

# global variables
env_object_sequences = {}  # store the random object access sequence for each environment
env_current_target_idx = {}  # store the index of the current target object in the sequence for each environment
env_velocities = {}  # store the current velocity for each environment
env_velocity_steps = {}  # record the number of steps since the last velocity update for each environment

def pre_process_actions(
    delta_pose: np.ndarray,
    num_envs: int,
    device: str,
    env,  # get current EE state
) -> torch.Tensor:
    """
    supports position_zero mode (using only Δx, Δy), but can not keep the end-effector facing downward
    """
    # get current ee state
    ee_pos, _ = robot_manager.get_ee_state(env)
    if ee_pos is None:
        # return 3-dim：Δx, Δy, gripper
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    #  Δx, Δy
    if len(delta_pose) >= 2:
        delta_xy = delta_pose[:2]
    else:
        delta_xy = [0.0, 0.0]

    delta_xy_tensor = torch.tensor(delta_xy, dtype=torch.float32, device=device).repeat(num_envs, 1)

    # gripper velocity: closed or open
    gripper_vel = torch.full((num_envs, 1), -1.0, dtype=torch.float32, device=device)  # -1.0 代表夹爪关闭

    # output: Δx, Δy, gripper
    return torch.cat([delta_xy_tensor, gripper_vel], dim=1)

# def generate_fixed_target_point(env_idx, env, push_height):

#     FIXED_TARGET_X = 0.77
#     FIXED_TARGET_Y = 0.0
    
#     env_origin = env.scene.env_origins[env_idx].cpu().numpy()
    
#     target_world_pos = np.array([
#         env_origin[0] + FIXED_TARGET_X,
#         env_origin[1] + FIXED_TARGET_Y,
#         push_height  
#     ])
    
#     relative_coords = target_world_pos - np.array([env_origin[0], env_origin[1], 0.0])
    
#     print(f"Env {env_idx} - Origin: {env_origin}, Target: {target_world_pos}")
#     print(f"Env {env_idx} - Relative target: ({relative_coords[0]:.2f}, {relative_coords[1]:.2f}, {relative_coords[2]:.2f})")
    
#     return target_world_pos

TARGET_X_MIN = 0.78
TARGET_X_MAX = 0.80
TARGET_Y_MIN = -0.1
TARGET_Y_MAX = 0.1
def generate_fixed_target_point(env_idx, env, push_height):

    random_x = np.random.uniform(TARGET_X_MIN, TARGET_X_MAX)
    random_y = np.random.uniform(TARGET_Y_MIN, TARGET_Y_MAX)
    
    env_origin = env.scene.env_origins[env_idx].cpu().numpy()

    target_world_pos = np.array([
        env_origin[0] + random_x,
        env_origin[1] + random_y,
        push_height  
    ])
    
    relative_coords = target_world_pos - np.array([env_origin[0], env_origin[1], 0.0])
    
    print(f"Env {env_idx} - Origin: {env_origin}, Target: {target_world_pos}")
    print(f"Env {env_idx} - Relative target: ({relative_coords[0]:.2f}, {relative_coords[1]:.2f}, {relative_coords[2]:.2f})")
    
    return target_world_pos

def get_noisy_action(action, random_magnitude):
    """add noise for actions"""
    if random_magnitude == 0:
        return action
    
    randomness = np.random.normal(loc=0, scale=1, size=action.shape) * random_magnitude
    randomness[2] = 0  # do not add noise in z axis
    noisy_action = action + randomness
    return noisy_action

def auto_explore(env, max_steps=500, safe_mode=True, noise_magnitude=0.003):

    global exploration_data, env_velocities, env_velocity_steps
    BACKING_OFF_ANGLE = 13.5
    try:
        env_config = env.task.safe_tabletop_config
    except:
        env_config = None
    
    ee_pos, _ = robot_manager.get_ee_state(env)
    init_position = ee_pos[0].cpu().numpy() if ee_pos is not None else None
    
    start_new_trajectory(env_config, init_position)

    steps = 0
    all_trajectories = [[] for _ in range(env.num_envs)]
    
    # only use half of the environments
    active_envs = list(range(env.num_envs )) 
    original_active_envs = active_envs.copy() 
    print(f"Using only {len(active_envs)} active environments out of {env.num_envs} total")
    
    push_height = DEFAULT_PUSH_HEIGHT

    # initialize safety violation counter for each environment (only for active environments)
    env_safety_violations = {env_idx: 0 for env_idx in active_envs}
    
    # track target achievement status (only for active environments)
    reached_targets = {env_idx: False for env_idx in active_envs}
    
    # initialize sampling points and velocities (only for active environments)
    sampling_points = {}
    for env_idx in active_envs:
        # set initial velocity
        env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
        env_velocity_steps[env_idx] = 0
        
        # set initial sampling points (fixed target points)
        sampling_points[env_idx] = generate_fixed_target_point(env_idx, env, push_height)
        print(f"Env {env_idx}: velocity={env_velocities[env_idx]:.4f}, target={sampling_points[env_idx]}")
    
    # get initial position data
    ee_positions = robot_manager.get_ee_positions_batch(env)
    object_states_tensor, _ = env_manager.get_object_states_batch(env)
    
    # initialize data recording for active environments
    tilt_history = {env_idx: {obj_idx: [] for obj_idx in range(object_states_tensor.shape[1])} 
                    for env_idx in active_envs}
    path_lengths = {env_idx: 0.0 for env_idx in active_envs}
    last_positions = {env_idx: ee_positions[env_idx].cpu().numpy() for env_idx in active_envs}
    
    # initialize previous_actions dictionary for safe backoff
    previous_actions = {env_idx: None for env_idx in active_envs}
    
    # define target achievement threshold
    GOAL_THRESHOLD = 0.01
    
    # main loop
    while steps < max_steps and active_envs:
        # update environment state
        ee_positions = robot_manager.get_ee_positions_batch(env)
        object_states_tensor, _ = env_manager.get_object_states_batch(env)
        
        # check if any environment has reached the target, if so, remove it from active_envs
        envs_to_remove = []
        for env_idx in active_envs:
            current_pos = ee_positions[env_idx].cpu().numpy()
            target_pos = sampling_points[env_idx]
            
            # calculate distance to target
            distance = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            # if distance is less than threshold, consider it as reached target
            if distance <= GOAL_THRESHOLD:  
                print(f"Env {env_idx} reached target! Distance: {distance:.4f}")
                reached_targets[env_idx] = True
                envs_to_remove.append(env_idx)
        
        # remove environments that have reached the target from active_envs
        for env_idx in envs_to_remove:
            active_envs.remove(env_idx)
            # record the final position and path data when the environment reached the target
            if env_idx in last_positions:
                current_pos = ee_positions[env_idx].cpu().numpy()
                path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
                last_positions[env_idx] = current_pos.copy()
                
                # add a final trajectory point
                target_pos = sampling_points[env_idx]
                all_trajectories[env_idx].append({
                    'step': steps,
                    'ee_pos': current_pos.copy(),
                    'target_pos': target_pos.copy(),
                    'action': np.zeros(3), 
                    'env_id': env_idx,
                    'velocity': 0.0 
                })
        
        if not active_envs:
            print("All active environments reached their targets! Ending current run.")
            break

        # generate actions for all environments
        action_list = [torch.zeros(3, device=env.device) for _ in range(env.num_envs)] 
        
        # calculate actions for active environments
        for env_idx in active_envs:
            # record tilt angle history for active environments
            quats = object_states_tensor[env_idx, :, 3:7]  # [num_objects, 4]
            tilt_angles = geometry_calc.calculate_tilt_angle_from_quaternion_batch(quats)
            
            for obj_idx in range(len(tilt_angles)):
                tilt_value = tilt_angles[obj_idx].cpu().item()
                tilt_history[env_idx][obj_idx].append(tilt_value)
            
            # update path length
            current_pos = ee_positions[env_idx].cpu().numpy()
            if env_idx in last_positions:
                path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
            last_positions[env_idx] = current_pos.copy()

            env_velocity_steps[env_idx] += 1
            if env_velocity_steps[env_idx] >= VELOCITY_CHANGE_INTERVAL:
                env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
                env_velocity_steps[env_idx] = 0
            
            current_pos = ee_positions[env_idx].cpu().numpy()
            target_pos = sampling_points[env_idx]
            
            # calculate distance to target
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction[:2])
            
            # generate action
            if distance > 0.01:
                current_velocity = env_velocities[env_idx]
                
                # normalize direction and apply velocity
                if distance > 0:
                    direction[:2] = direction[:2] / distance * current_velocity
                direction[2] = 0  # keep Z axis unchanged

                if noise_magnitude > 0:
                    direction = get_noisy_action(direction, noise_magnitude)
                
                # process action and record trajectory
                single_action = pre_process_actions(direction[:2], 1, env.device, env)
                action_list[env_idx] = single_action[0]  # assign action to corresponding environment index
                
                # record current action for safe backoff
                previous_actions[env_idx] = single_action[0].clone()
                
                all_trajectories[env_idx].append({
                    'step': steps,
                    'ee_pos': current_pos.copy(),
                    'target_pos': target_pos.copy(),
                    'action': direction.copy(),
                    'env_id': env_idx,
                    'velocity': current_velocity
                })
        
        # add safety check before executing actions
        if safe_mode:
            # get tipping cost
            tipping_costs, _ = safety_calc.calculate_task_safety_cost_batch(env)
            # get closeness cost
            closeness_costs, _ = safety_calc.calculate_current_closeness_cost_batch(env)

            for env_idx in active_envs:
                # if tipping_costs[env_idx] > np.radians(args_cli.max_tipping_angle * SAFETY_THRESHOLD_FACTOR):
                if tipping_costs[env_idx] > np.radians(BACKING_OFF_ANGLE): 
                    # get specific cost value for output
                    tipping_value = np.degrees(tipping_costs[env_idx].cpu().item()) if hasattr(tipping_costs, 'cpu') else np.degrees(tipping_costs[env_idx])
                    closeness_value = closeness_costs[env_idx].cpu().item() if hasattr(closeness_costs, 'cpu') else closeness_costs[env_idx]
                    
                    print(f"Environment {env_idx} approaching safety threshold (tilt={tipping_value:.2f}°, closeness={closeness_value:.2f}), backing off")
                     
                    # if there is a recorded previous action, execute it backward
                    if previous_actions[env_idx] is not None:
                        action_list[env_idx] = -previous_actions[env_idx] * BACKUP_SPEED_FACTOR
                    else:
                        # if there is no recorded action, execute the current action backward
                        action_list[env_idx] = -action_list[env_idx] * BACKUP_SPEED_FACTOR
                    
                    # generate new sampling point
                    # sampling_points[env_idx] = generate_fixed_target_point(env_idx, env, push_height)
                    # print(f"Generated new target for env {env_idx}: {sampling_points[env_idx]}")
                    
                    # record NCBF prevention count
                    exploration_data['ncbf_preventions'] += 1
        
        # execute actions (all environments, but only active environments have actual actions)
        actions = torch.stack(action_list, dim=0)

        env.step(actions)
        collect_all_environments_trajectory(env)
        
        # check safety violation (only consider active environments)
        is_violated, tipping_costs, closeness_costs, _ = safety_calc.is_total_safety_violated_batch(env)
        for env_idx in active_envs:  # only check active environments
            if is_violated[env_idx]:
                exploration_data['safety_violations'] += 1
                env_safety_violations[env_idx] += 1
                tipping_value = tipping_costs[env_idx].cpu().item() if hasattr(tipping_costs, 'cpu') else tipping_costs[env_idx]
                print(f"Env {env_idx} safety violation! Tilt: {np.degrees(tipping_value):.2f}°")
        
        steps += 1
        print(f"step {steps}")
        exploration_data['total_steps'] += 1

    trajectory_stats = []
    for env_idx in original_active_envs:  
        if len(all_trajectories[env_idx]) > 0:
            # calculate final distance to target
            final_distance = np.linalg.norm(all_trajectories[env_idx][-1]['ee_pos'][:2] - 
                                            all_trajectories[env_idx][-1]['target_pos'][:2])
            
            # calculate trajectory statistics
            stats = {
                'env_id': env_idx,
                'steps': len(all_trajectories[env_idx]),
                'path_length': path_lengths[env_idx],
                'initial_distance_to_goal': np.linalg.norm(all_trajectories[env_idx][0]['ee_pos'][:2] - 
                                                          all_trajectories[env_idx][0]['target_pos'][:2]),
                'final_distance_to_goal': final_distance,
                'reached_target': reached_targets.get(env_idx, False),
                'violation_rate': env_safety_violations.get(env_idx, 0) / len(all_trajectories[env_idx]) * 100 if len(all_trajectories[env_idx]) > 0 else 0,
                'unsafe_samples': env_safety_violations.get(env_idx, 0)
            }
            
            # calculate max and avg tilt angles
            max_tilt_angles = {}
            avg_tilt_angles = {}
            
            if env_idx in tilt_history:
                for obj_idx in tilt_history[env_idx]:
                    if tilt_history[env_idx][obj_idx]: 
                        max_tilt_angles[obj_idx] = max(tilt_history[env_idx][obj_idx])
                        avg_tilt_angles[obj_idx] = sum(tilt_history[env_idx][obj_idx]) / len(tilt_history[env_idx][obj_idx])
            
            stats['max_tilt_by_object'] = max_tilt_angles
            stats['avg_tilt_by_object'] = avg_tilt_angles
            stats['max_tilt_angle'] = max(max_tilt_angles.values()) if max_tilt_angles else 0.0
            stats['avg_tilt_angle'] = sum(avg_tilt_angles.values()) / len(avg_tilt_angles) if avg_tilt_angles else 0.0
            
            trajectory_stats.append(stats)
    
    exploration_data['trajectory_stats'].extend(trajectory_stats)
    
    return all_trajectories


def clear_scene():
    """Clean up any existing prims in the scene"""
    import omni.usd
    from pxr import Usd
    
    stage = omni.usd.get_context().get_stage()
    
    # clean up the environment directory (recursively delete all child Prim)
    env_path = "/World/envs"
    if stage.GetPrimAtPath(env_path).IsValid():
        print(f"cleaning environment: {env_path}")
        stage.RemovePrim(env_path)
    
    print("Scene cleared")

def print_exploration_stats():
    global exploration_data
    
    print(f"total trajectories: {len(exploration_data['trajectories'])}")
    print(f"total steps: {exploration_data['total_steps']}")
    print(f"safety violations: {exploration_data['safety_violations']}")
    print(f"NCBF preventions: {exploration_data['ncbf_preventions']}")
    
    if exploration_data['total_steps'] > 0:
        violation_rate = exploration_data['safety_violations'] / exploration_data['total_steps'] * 100
        print(f"violation rate: {violation_rate:.1f}%")
    
    # print trajectory detailed statistics
    if 'trajectory_stats' in exploration_data and exploration_data['trajectory_stats']:
        print("\ntrajectory detailed statistics:")
        print("-"*60)
        
        # accumulators for calculating averages
        total_steps = 0
        total_path_length = 0.0
        total_initial_distance = 0.0
        total_final_distance = 0.0
        total_max_tilt_angle = 0.0
        total_avg_tilt_angle = 0.0
        total_unsafe_samples = 0
        total_violation_rate = 0.0
        targets_reached = 0  # target reached count
        total_trajectories = 0  # total number of valid trajectories
        
        # accumulators for safe trajectories
        safe_traj_count = 0
        safe_max_tilt_angle = 0.0
        
        # accumulators for unsafe trajectories
        unsafe_traj_count = 0
        unsafe_max_tilt_angle = 0.0
        
        # accumulators for success and failure trajectory path lengths
        success_path_length = 0.0
        success_count = 0
        failure_path_length = 0.0
        failure_count = 0
        
        safe_and_reached = 0
        
        failure_final_distance = 0.0
        
        for i, stats in enumerate(exploration_data['trajectory_stats']):
           
            # print(f"trajectory {i+1} (env {stats['env_id']}):")
            # print(f"  steps: {stats['steps']}")
            # print(f"  path length: {stats['path_length']:.2f} m")
            # print(f"  initial distance to goal: {stats['initial_distance_to_goal']:.2f} m")
            # print(f"  final distance to goal: {stats['final_distance_to_goal']:.2f} m")
            # print(f"  max tilt angle: {np.degrees(stats['max_tilt_angle']):.2f}°")
            # print(f"  average tilt angle: {np.degrees(stats['avg_tilt_angle']):.2f}°")
            # print(f"  unsafe samples: {stats['unsafe_samples']}")
            # print(f"  violation rate: {stats['violation_rate']:.1f}%")
            # print(f"  reached target: {'Yes' if stats.get('reached_target', False) else 'No'}")
            # print()
            
            # accumulate statistics
            total_trajectories += 1
            total_steps += stats['steps']
            total_path_length += stats['path_length']
            total_initial_distance += stats['initial_distance_to_goal']
            total_final_distance += stats['final_distance_to_goal']
            total_max_tilt_angle += stats['max_tilt_angle']
            total_avg_tilt_angle += stats['avg_tilt_angle']
            total_unsafe_samples += stats['unsafe_samples']
            total_violation_rate += stats['violation_rate']
            
            if stats.get('reached_target', False) and stats['unsafe_samples'] == 0:
                safe_and_reached += 1
            
            # accumulate to success or failure path lengths based on whether target is reached
            if stats.get('reached_target', False):
                targets_reached += 1  # count target reached
                success_path_length += stats['path_length']
                success_count += 1
            else:
                failure_path_length += stats['path_length']
                failure_count += 1
         
                failure_final_distance += stats['final_distance_to_goal']
                
            # separate safe and unsafe trajectories
            if stats['unsafe_samples'] > 0:
                # unsafe trajectory
                unsafe_traj_count += 1
                unsafe_max_tilt_angle += stats['max_tilt_angle']
            else:
                # safe trajectory
                safe_traj_count += 1
                safe_max_tilt_angle += stats['max_tilt_angle']
        
        # calculate averages
        if total_trajectories > 0:
            avg_steps = total_steps / total_trajectories
            avg_path_length = total_path_length / total_trajectories
            avg_initial_distance = total_initial_distance / total_trajectories
            avg_final_distance = total_final_distance / total_trajectories
            avg_max_tilt_angle = total_max_tilt_angle / total_trajectories
            avg_avg_tilt_angle = total_avg_tilt_angle / total_trajectories
            avg_unsafe_samples = total_unsafe_samples / total_trajectories
            avg_violation_rate = total_violation_rate / total_trajectories
            target_success_rate = (targets_reached / total_trajectories) * 100
 
            safe_and_reached_rate = (safe_and_reached / total_trajectories) * 100
            
            avg_failure_final_distance = failure_final_distance / failure_count if failure_count > 0 else 0.0
            
            # calculate avg max tilt angles for safe and unsafe trajectories
            avg_safe_max_tilt_angle = safe_max_tilt_angle / safe_traj_count if safe_traj_count > 0 else 0.0
            avg_unsafe_max_tilt_angle = unsafe_max_tilt_angle / unsafe_traj_count if unsafe_traj_count > 0 else 0.0
            
            # calculate avg path lengths for success and failure trajectories
            avg_success_path_length = success_path_length / success_count if success_count > 0 else 0.0
            avg_failure_path_length = failure_path_length / failure_count if failure_count > 0 else 0.0
            
            # print avg statistics
            print("="*60)
            print(f"  safe and reached success rate: {safe_and_reached_rate:.1f}%")
            print(f"  average final distance to goal before failure: {avg_failure_final_distance:.2f} m")
            print(f"  target success rate: {target_success_rate:.1f}%")
            print(f"  safe trajectories: {safe_traj_count} ({safe_traj_count/total_trajectories*100:.1f}%)")
            print(f"  average final distance to goal: {avg_final_distance:.2f} m")    
            print("="*60)

            print(f"  environments that were both safe and reached target: {safe_and_reached} out of {total_trajectories}")
            print("average statistics of all trajectories:")
            print(f"  average steps: {avg_steps:.2f}")
            print(f"  average path length: {avg_path_length:.2f} m")
            print(f"  average length before success (reached goal): {avg_success_path_length:.2f} m")
            print(f"  average length before failure (not reached goal): {avg_failure_path_length:.2f} m")
            print(f"  average initial distance to goal: {avg_initial_distance:.2f} m")

            print(f"  average max tilt angle (all): {np.degrees(avg_max_tilt_angle):.2f}°")
            print(f"  average max tilt angle (safe trajectories): {np.degrees(avg_safe_max_tilt_angle):.2f}°")
            print(f"  average max tilt angle (unsafe trajectories): {np.degrees(avg_unsafe_max_tilt_angle):.2f}°")
            print(f"  average average tilt angle: {np.degrees(avg_avg_tilt_angle):.2f}°")
            print(f"  average unsafe samples: {avg_unsafe_samples:.2f}")
            print(f"  average violation rate: {avg_violation_rate:.1f}%")
            print(f"  total trajectories counted: {total_trajectories}")
            print(f"  unsafe trajectories: {unsafe_traj_count} ({unsafe_traj_count/total_trajectories*100:.1f}%)")
            print(f"  environments that reached target: {targets_reached} out of {total_trajectories}")
            print(f"  environments that failed to reach target: {failure_count} out of {total_trajectories}")

            # add avg statistics to exploration_data
            exploration_data['avg_stats'] = {
                'avg_steps': avg_steps,
                'avg_path_length': avg_path_length,
                'avg_success_path_length': avg_success_path_length,
                'avg_failure_path_length': avg_failure_path_length,
                'avg_initial_distance': avg_initial_distance,
                'avg_final_distance': avg_final_distance,
      
                'avg_failure_final_distance': avg_failure_final_distance,
                'avg_max_tilt_angle': avg_max_tilt_angle,
                'avg_max_tilt_angle_safe': avg_safe_max_tilt_angle,
                'avg_max_tilt_angle_unsafe': avg_unsafe_max_tilt_angle,
                'avg_avg_tilt_angle': avg_avg_tilt_angle,
                'avg_unsafe_samples': avg_unsafe_samples,
                'avg_violation_rate': avg_violation_rate,
                'target_success_rate': target_success_rate,
            
                'safe_and_reached_rate': safe_and_reached_rate,
                'targets_reached': targets_reached,
                'total_trajectories': total_trajectories,
                'safe_trajectories': safe_traj_count,
                'unsafe_trajectories': unsafe_traj_count,
                'success_count': success_count,
                'failure_count': failure_count,

                'safe_and_reached_count': safe_and_reached
            }


def save_run_stats(all_stats, run_idx, base_dir="run_stats_back_40obj"):

    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_dir}/run_{run_idx}_stats_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"saved {run_idx} statistical data to：{filename}")
    return filename

def save_summary_stats(base_dir="run_stats_summary_back_40obj", run_params=None):
    """Save key statistical metrics to a separate file"""
    global exploration_data

    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_dir}/summary_stats_{timestamp}.txt"

    # Ensure avg_stats exists
    if 'avg_stats' not in exploration_data:
        print("Cannot save summary statistics, avg_stats does not exist")
        return None

    stats = exploration_data['avg_stats']

    with open(filename, 'w') as f:
        if run_params:
            f.write("="*60 + "\n")
            f.write("Run parameters:\n")
            for key, value in run_params.items():
                f.write(f"  {key}: {value}\n")
        f.write("="*60 + "\n")
        f.write(f"Safe and reached success rate: {stats['safe_and_reached_rate']:.1f}%\n")
        f.write(f"Average final distance to goal before failure: {stats['avg_failure_final_distance']:.2f} m\n")
        f.write(f"Target success rate: {stats['target_success_rate']:.1f}%\n")
        f.write(f"Safe trajectories: {stats['safe_trajectories']} ({stats['safe_trajectories']/stats['total_trajectories']*100:.1f}%)\n")
        f.write(f"Average final distance to goal: {stats['avg_final_distance']:.2f} m\n")
        f.write("="*60 + "\n")
        
        f.write(f"Environments both safe and reached target: {stats['safe_and_reached_count']} out of {stats['total_trajectories']}\n")
        f.write("Average statistics of all trajectories:\n")
        f.write(f"  Average steps: {stats['avg_steps']:.2f}\n")
        f.write(f"  Average path length: {stats['avg_path_length']:.2f} m\n")
        f.write(f"  Average length before success (reached goal): {stats['avg_success_path_length']:.2f} m\n")
        f.write(f"  Average length before failure (not reached goal): {stats['avg_failure_path_length']:.2f} m\n")
        f.write(f"  Average initial distance to goal: {stats['avg_initial_distance']:.2f} m\n")
        
        f.write(f"  Average max tilt angle (all): {np.degrees(stats['avg_max_tilt_angle']):.2f}\n")
        f.write(f"  Average max tilt angle (safe trajectories): {np.degrees(stats['avg_max_tilt_angle_safe']):.2f}\n")
        f.write(f"  Average max tilt angle (unsafe trajectories): {np.degrees(stats['avg_max_tilt_angle_unsafe']):.2f}\n")
        f.write(f"  Average average tilt angle: {np.degrees(stats['avg_avg_tilt_angle']):.2f}\n")
        f.write(f"  Average unsafe samples: {stats['avg_unsafe_samples']:.2f}\n")
        f.write(f"  Average violation rate: {stats['avg_violation_rate']:.1f}%\n")
        f.write(f"  Total trajectories counted: {stats['total_trajectories']}\n")
        f.write(f"  Unsafe trajectories: {stats['unsafe_trajectories']} ({stats['unsafe_trajectories']/stats['total_trajectories']*100:.1f}%)\n")
        f.write(f"  Environments that reached target: {stats['targets_reached']} out of {stats['total_trajectories']}\n")
        f.write(f"  Environments that failed to reach target: {stats['failure_count']} out of {stats['total_trajectories']}\n")
    
    print(f"Summary statistics saved to: {filename}")
    return filename


def main():
    """Main function to run NCBF-guided exploration in Isaac Lab environments."""
    seed = args_cli.seed
    print(f"Setting random seed to: {seed}")
    
    # Python内置random模块
    import random
    random.seed(seed)
    
    # NumPy随机数生成器
    np.random.seed(seed)
    
    # PyTorch随机数生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clear_scene()
    
    # parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "randomize_objects"):
        randomize_params = {"scene_type": args_cli.scene_type}
        env_cfg.events.randomize_objects.params = {"params": randomize_params}
    
    print(f"Using scene type: {args_cli.scene_type}")

    print(f"Number of environments: {args_cli.num_envs}")
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # modify other configurations that may affect environment number
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
        env_cfg.sim.num_envs = args_cli.num_envs
    
    # 修改配置
    env_cfg.terminations.time_out = None
    
    if "Lift" in args_cli.task:
        # 将重采样时间范围设置为大数以避免重采样
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # 添加到达目标的终止条件，否则环境不会重置
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped    
    
    safety_calc.config.max_tipping_angle_degrees = args_cli.max_tipping_angle
    safety_calc.config.use_closeness_cost = False  
    safety_calc.config.max_closeness_cost = 5.0   
    
    print(f"Safety threshold set to: {args_cli.max_tipping_angle} degrees")
    print(f"Closeness cost enabled with max value: {safety_calc.config.max_closeness_cost}")
    
    print(f"Robot velocity settings: min={MIN_VELOCITY}, max={MAX_VELOCITY}, change interval={VELOCITY_CHANGE_INTERVAL} steps")
    
    env.reset()
    
    noise_magnitude = args_cli.noise_magnitude
    total_runs = args_cli.total_runs
    max_steps = args_cli.max_steps
    
    print(f"Starting NCBF-guided exploration in all environments, running {total_runs} iterations with noise magnitude: {noise_magnitude}")
    
    for run_idx in range(total_runs):
        print(f"===== Starting exploration run {run_idx+1}/{total_runs} =====")
        
        env.reset()
    
        env_velocities.clear()
        env_velocity_steps.clear()
        
        for _ in range(3):
            env.sim.render()
        
        # calibrate end-effector height to 0.19
        print(f"Calibrating end-effector height to {DEFAULT_PUSH_HEIGHT}...")
        
        zero_delta_pose = np.zeros(6)
        height_adjust_action = pre_process_actions(zero_delta_pose, env.num_envs, env.device, env)
        steps = 50
        
        for step in range(steps):
            step_action = height_adjust_action * (step + 1) / steps
            env.step(step_action)
            env.sim.render()
        
    
        all_trajectories = auto_explore(env, max_steps=max_steps, safe_mode=args_cli.safe_mode, noise_magnitude=noise_magnitude)

        run_stats = {
            'run_id': run_idx + 1,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'environments': {}
        }
        
        for i, stats in enumerate(exploration_data['trajectory_stats'][-env.num_envs:]):
            env_idx = stats['env_id']
            run_stats['environments'][f'env_{env_idx}'] = stats
        
       
        save_run_stats(run_stats, run_idx + 1, base_dir=f"run_stats_back_40obj/seed_{seed}")
        
        # after end-effector height is calibrated to 0.19m, save data (object height is 0.2m)
        for traj in all_trajectories:
            if len(traj) > 0:
                exploration_data['trajectories'].append(traj)
        
        base_dir = f"trajectories_data_back_40obj/seed_{seed}"
        end_current_trajectory(save=True, base_dir=base_dir)
        print(f"Trajectory data has been saved to the {base_dir} directory")
    
    print(f"All {total_runs} exploration runs completed")
    print_exploration_stats()
    run_params = {
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "scene_type": args_cli.scene_type,
        "task": args_cli.task,
        "max_tipping_angle": args_cli.max_tipping_angle,
        "max_steps": args_cli.max_steps,
        "noise_magnitude": args_cli.noise_magnitude,
        "safe_mode": args_cli.safe_mode,
    }
    save_summary_stats(base_dir=f"run_stats_summary_back_4obj/seed_{seed}", run_params=run_params)

    env.close()


if __name__ == "__main__":
   
    main()

    simulation_app.close()