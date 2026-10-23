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
    default=3, 
    help="Random seed for reproducibility."
)
parser.add_argument(
    "--scene_type", 
    type=str, 
    default="extreme_large", 
    choices=["small", "medium", "large", "extra_large", "extreme_large"],
    help="Scene type determining number of objects (small=4, medium=10, large=20, extra_large=40)"
)
# CBF Safety Filter arguments
parser.add_argument(
    "--enable_cbf",
    action="store_true",
    default=True,
    help="Enable CBF safety filter."
)
parser.add_argument(
    "--cbf_alpha",
    type=float,
    default=1.0,
    help="CBF alpha parameter (class-K function coefficient)."
)
parser.add_argument(
    "--cbf_safety_margin",
    type=float,
    default=0.0,
    help="CBF safety margin in meters."
)
parser.add_argument(
    "--cbf_adaptive",
    action="store_true",
    default=False,
    help="Use adaptive CBF (alpha adjusts based on distance to obstacles)."
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

# Import RRT planner module
from rrt_planer import RRT, get_object_obstacles_for_rrt

# Import CBF Safety Filter
from cbf_safety_filter import CBFSafetyFilter, AdaptiveCBFSafetyFilter

# Initialize global manager instance
env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()

# Global CBF filter instance (will be initialized in main)
cbf_filter = None

DEFAULT_PUSH_HEIGHT = 0.19  # push height
OBJECT_RADIUS = 0.05  # object radius
MIN_VELOCITY = 0.01  # meter / step
MAX_VELOCITY = 0.01  # meter / step
VELOCITY_CHANGE_INTERVAL = 5  # update velocity every 5 steps
ROBOT_RADIUS = 0.005  
# Note that every 'velocity' is actually: meter / step
TARGET_X_MIN = 0.78
TARGET_X_MAX = 0.80
TARGET_Y_MIN = -0.1
TARGET_Y_MAX = 0.1

# statistic data
exploration_data = {
    'trajectories': [],
    'safety_violations': 0,
    'total_steps': 0,
    'ncbf_preventions': 0,
    'cbf_preventions': 0,  # New: track CBF interventions
    'trajectory_stats': [] 
}

# global variables
env_velocities = {}  # store the current velocity for each environment
env_velocity_steps = {}  # record the number of steps since the last velocity update for each environment


def pre_process_actions(
    delta_pose: np.ndarray,
    num_envs: int,
    device: str,
    env,
) -> torch.Tensor:
    """
    supports position_zero mode (using only Δx, Δy), but can not keep the end-effector facing downward
    """
    # get current ee state
    ee_pos, _ = robot_manager.get_ee_state(env)
    if ee_pos is None:
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    if len(delta_pose) >= 2:
        delta_xy = delta_pose[:2]
    else:
        delta_xy = [0.0, 0.0]

    delta_xy_tensor = torch.tensor(delta_xy, dtype=torch.float32, device=device).repeat(num_envs, 1)
    gripper_vel = torch.full((num_envs, 1), -1.0, dtype=torch.float32, device=device)

    return torch.cat([delta_xy_tensor, gripper_vel], dim=1)


def generate_target_point(env_idx, env, push_height):
  
    # generate random relative coordinates
    random_x = np.random.uniform(TARGET_X_MIN, TARGET_X_MAX)
    random_y = np.random.uniform(TARGET_Y_MIN, TARGET_Y_MAX)
    
    # get environment origin
    env_origin = env.scene.env_origins[env_idx].cpu().numpy()
    
    # calculate absolute world coordinates
    target_world_pos = np.array([
        env_origin[0] + random_x,
        env_origin[1] + random_y,
        push_height  
    ])
    
    # calculate and print relative coordinates
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

def auto_explore_with_rrt(env, max_steps=500, noise_magnitude=0.003, use_cbf=False):
    """RRT-based exploration with optional CBF safety filter"""
    global exploration_data, env_velocities, env_velocity_steps, cbf_filter

    try:
        env_config = env.task.safe_tabletop_config
    except:
        env_config = None
    
    ee_pos, _ = robot_manager.get_ee_state(env)
    init_position = ee_pos[0].cpu().numpy() if ee_pos is not None else None

    steps = 0
    all_trajectories = [[] for _ in range(env.num_envs)]
    
    active_envs = list(range(env.num_envs // 1))
    original_active_envs = active_envs.copy()
    print(f"Using {len(active_envs)} active environments out of {env.num_envs} total")
    
    if use_cbf:
        print(f"CBF Safety Filter: ENABLED")
        cbf_filter.reset_statistics()
    else:
        print(f"CBF Safety Filter: DISABLED")
    
    push_height = DEFAULT_PUSH_HEIGHT

    # Initialize tracking structures
    env_safety_violations = {env_idx: 0 for env_idx in active_envs}
    env_cbf_interventions = {env_idx: 0 for env_idx in active_envs}  # Track CBF interventions per env
    reached_targets = {env_idx: False for env_idx in active_envs}
    
    # Initialize target points and paths
    target_points = {}
    planned_paths = {}
    path_indices = {}
    
    # Get initial positions and obstacles
    ee_positions = robot_manager.get_ee_positions_batch(env)
    object_states_tensor, _ = env_manager.get_object_states_batch(env)
    
    # ============================================================
    # PHASE 1: GENERATE ALL PATHS FIRST (Before any movement)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Generating RRT paths for all environments...")
    print("="*60)
    
    for env_idx in active_envs:
        env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
        env_velocity_steps[env_idx] = 0
        
        # Generate target point
        target_points[env_idx] = generate_target_point(env_idx, env, push_height)
        
        # Get current position
        current_pos = ee_positions[env_idx].cpu().numpy()
        target_pos = target_points[env_idx]
        
        # Get obstacles for RRT
        obstacles = get_object_obstacles_for_rrt(env_idx, object_states_tensor)
        
        # Create RRT planner using the improved class
        print(f"\nEnv {env_idx}: Planning path...")
        print(f"  Start: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
        print(f"  Goal:  ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
        print(f"  Obstacles: {len(obstacles)}")
        
        # Dynamic workspace bounds based on start and goal
        x_min = min(current_pos[0], target_pos[0]) - 0.1
        x_max = max(current_pos[0], target_pos[0]) + 0.1
        y_min = min(current_pos[1], target_pos[1]) - 0.1
        y_max = max(current_pos[1], target_pos[1]) + 0.1
        
        rrt = RRT(
            q_start=current_pos[:2],
            q_goal=target_pos[:2],
            obstacles=obstacles,
            robot_radius=ROBOT_RADIUS * 1.2,
            q_limits=[(x_min, x_max), (y_min, y_max)],
            max_iter=1000,
            step_size=0.01
        )
        
        # Plan complete path
        path = rrt.plan()
        
        if path is not None:
            planned_paths[env_idx] = path
            path_indices[env_idx] = 1
            path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                            for i in range(len(path)-1))
            print(f"  ✓ Path found: {len(path)} waypoints, length: {path_length:.3f}m, cost: {rrt.q_goal.cost:.3f}")
        else:
            print(f"  ✗ RRT failed! Using direct path to goal")
            planned_paths[env_idx] = [current_pos[:2].tolist(), target_pos[:2].tolist()]
            path_indices[env_idx] = 1
    
    print("\n" + "="*60)
    print("PHASE 2: Executing pre-planned paths...")
    if use_cbf:
        print("  -> CBF Safety Filter is ACTIVE")
    print("="*60 + "\n")
    
    # Initialize tracking
    tilt_history = {env_idx: {obj_idx: [] for obj_idx in range(object_states_tensor.shape[1])} 
                    for env_idx in active_envs}
    path_lengths = {env_idx: 0.0 for env_idx in active_envs}
    last_positions = {env_idx: ee_positions[env_idx].cpu().numpy() for env_idx in active_envs}
    
    GOAL_THRESHOLD = 0.01
    WAYPOINT_THRESHOLD = 0.02
    
    # ============================================================
    # PHASE 2: EXECUTE PRE-PLANNED PATHS with CBF filtering
    # ============================================================
    while steps < max_steps and active_envs:
        # Update environment state
        ee_positions = robot_manager.get_ee_positions_batch(env)
        object_states_tensor, _ = env_manager.get_object_states_batch(env)
        
        # Check for goal achievement
        envs_to_remove = []
        for env_idx in active_envs:
            current_pos = ee_positions[env_idx].cpu().numpy()
            target_pos = target_points[env_idx]
            distance = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if distance <= GOAL_THRESHOLD:
                print(f"✓ Env {env_idx} reached target! Distance: {distance:.4f}m")
                reached_targets[env_idx] = True
                envs_to_remove.append(env_idx)
        
        # Remove completed environments
        for env_idx in envs_to_remove:
            active_envs.remove(env_idx)
            if env_idx in last_positions:
                current_pos = ee_positions[env_idx].cpu().numpy()
                path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
                last_positions[env_idx] = current_pos.copy()
                
                target_pos = target_points[env_idx]
                all_trajectories[env_idx].append({
                    'step': steps,
                    'ee_pos': current_pos.copy(),
                    'target_pos': target_pos.copy(),
                    'action': np.zeros(3),
                    'env_id': env_idx,
                    'velocity': 0.0,
                    'path_complete': True
                })
        
        if not active_envs:
            print("\n✓ All active environments reached their targets!")
            break

        # Generate actions for active environments
        action_list = [torch.zeros(3, device=env.device) for _ in range(env.num_envs)]
        
        for env_idx in active_envs:
            # Record tilt history
            quats = object_states_tensor[env_idx, :, 3:7]
            tilt_angles = geometry_calc.calculate_tilt_angle_from_quaternion_batch(quats)
            
            for obj_idx in range(len(tilt_angles)):
                tilt_value = tilt_angles[obj_idx].cpu().item()
                tilt_history[env_idx][obj_idx].append(tilt_value)
            
            # Update path length
            current_pos = ee_positions[env_idx].cpu().numpy()
            if env_idx in last_positions:
                path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
            last_positions[env_idx] = current_pos.copy()

            # Update velocity periodically
            env_velocity_steps[env_idx] += 1
            if env_velocity_steps[env_idx] >= VELOCITY_CHANGE_INTERVAL:
                env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
                env_velocity_steps[env_idx] = 0
            
            target_pos = target_points[env_idx]
            
            # Follow pre-planned path
            path = planned_paths[env_idx]
            path_idx = path_indices[env_idx]
            
            # Determine next waypoint
            if path_idx < len(path):
                next_waypoint = np.array(path[path_idx])
                direction = next_waypoint - current_pos[:2]
                distance_to_waypoint = np.linalg.norm(direction)
                
                # Move to next waypoint if close enough
                if distance_to_waypoint < WAYPOINT_THRESHOLD:
                    path_indices[env_idx] += 1
                    print(f"Env {env_idx}: Reached waypoint {path_idx}/{len(path)-1}")
                    
                    if path_indices[env_idx] < len(path):
                        next_waypoint = np.array(path[path_indices[env_idx]])
                        direction = next_waypoint - current_pos[:2]
                        distance_to_waypoint = np.linalg.norm(direction)
                    else:
                        direction = target_pos[:2] - current_pos[:2]
                        distance_to_waypoint = np.linalg.norm(direction)
            else:
                direction = target_pos[:2] - current_pos[:2]
                distance_to_waypoint = np.linalg.norm(direction)
            
            # Normalize and scale by velocity
            if distance_to_waypoint > 0.01:
                direction = direction / distance_to_waypoint * env_velocities[env_idx]
            else:
                direction = np.zeros(2)
            
            # Add noise
            direction_3d = np.array([direction[0], direction[1], 0.0])
            if noise_magnitude > 0:
                direction_3d = get_noisy_action(direction_3d, noise_magnitude)
            
            # Process action
            single_action = pre_process_actions(direction_3d[:2], 1, env.device, env)
            action_list[env_idx] = single_action[0]
            
            # Record trajectory (before CBF filtering)
            all_trajectories[env_idx].append({
                'step': steps,
                'ee_pos': current_pos.copy(),
                'target_pos': target_pos.copy(),
                'action_desired': direction_3d.copy(),  # Store desired action
                'env_id': env_idx,
                'velocity': env_velocities[env_idx],
                'waypoint': next_waypoint.tolist() if path_idx < len(path) else target_pos[:2].tolist(),
                'waypoint_idx': path_idx,
                'total_waypoints': len(path)
            })
        
        # Stack all actions
        actions = torch.stack(action_list, dim=0)
        
        # ============================================================
        # CBF SAFETY FILTER APPLICATION
        # ============================================================
        if use_cbf and cbf_filter is not None:
            # Apply CBF safety filter
            filtered_actions, num_filtered = cbf_filter.filter_actions_batch(
                actions, 
                ee_positions, 
                object_states_tensor,
                safety_threshold=0.1
            )
            
            # Track CBF interventions
            if num_filtered > 0:
                exploration_data['cbf_preventions'] += num_filtered
                
                # Track per-environment
                for env_idx in active_envs:
                    action_diff = torch.norm(actions[env_idx] - filtered_actions[env_idx])
                    if action_diff > 1e-4:
                        env_cbf_interventions[env_idx] += 1
                
                if steps % 10 == 0:  # Print less frequently
                    print(f"Step {steps}: CBF filtered {num_filtered}/{len(active_envs)} actions")
            
            # Use filtered actions
            actions = filtered_actions
            
            # Update trajectory records with actual executed action
            for env_idx in active_envs:
                if len(all_trajectories[env_idx]) > 0:
                    all_trajectories[env_idx][-1]['action'] = actions[env_idx].cpu().numpy()
                    all_trajectories[env_idx][-1]['cbf_filtered'] = (
                        torch.norm(actions[env_idx] - action_list[env_idx]) > 1e-4
                    )
        else:
            # No CBF filtering - use desired actions directly
            for env_idx in active_envs:
                if len(all_trajectories[env_idx]) > 0:
                    all_trajectories[env_idx][-1]['action'] = actions[env_idx].cpu().numpy()
                    all_trajectories[env_idx][-1]['cbf_filtered'] = False
        
        # Execute actions
        env.step(actions)
        
        # Check safety violations
        is_violated, tipping_costs, closeness_costs, _ = safety_calc.is_total_safety_violated_batch(env)
        for env_idx in active_envs:
            if is_violated[env_idx]:
                exploration_data['safety_violations'] += 1
                env_safety_violations[env_idx] += 1
                tipping_value = tipping_costs[env_idx].cpu().item() if hasattr(tipping_costs, 'cpu') else tipping_costs[env_idx]
                print(f"⚠ Env {env_idx} safety violation! Tilt: {np.degrees(tipping_value):.2f}°")
        
        steps += 1
        exploration_data['total_steps'] += 1
        
        # Print progress
        if steps % 50 == 0:
            print(f"\nStep {steps}/{max_steps} - Active environments: {len(active_envs)}")
            if use_cbf:
                cbf_stats = cbf_filter.get_statistics()
                print(f"  CBF filter rate: {cbf_stats['filter_rate']:.1f}%")
    
    # Print CBF statistics at end
    if use_cbf and cbf_filter is not None:
        print("\n" + "="*60)
        cbf_filter.print_statistics()
        print("="*60)
    
    # Calculate statistics
    trajectory_stats = []
    for env_idx in original_active_envs:
        if len(all_trajectories[env_idx]) > 0:
            final_distance = np.linalg.norm(all_trajectories[env_idx][-1]['ee_pos'][:2] - 
                                            all_trajectories[env_idx][-1]['target_pos'][:2])
            
            stats = {
                'env_id': env_idx,
                'steps': len(all_trajectories[env_idx]),
                'path_length': path_lengths[env_idx],
                'planned_path_waypoints': len(planned_paths.get(env_idx, [])),
                'initial_distance_to_goal': np.linalg.norm(all_trajectories[env_idx][0]['ee_pos'][:2] - 
                                                          all_trajectories[env_idx][0]['target_pos'][:2]),
                'final_distance_to_goal': final_distance,
                'reached_target': reached_targets.get(env_idx, False),
                'violation_rate': env_safety_violations.get(env_idx, 0) / len(all_trajectories[env_idx]) * 100 if len(all_trajectories[env_idx]) > 0 else 0,
                'unsafe_samples': env_safety_violations.get(env_idx, 0),
                'cbf_interventions': env_cbf_interventions.get(env_idx, 0),  # Add CBF intervention count
                'cbf_intervention_rate': env_cbf_interventions.get(env_idx, 0) / len(all_trajectories[env_idx]) * 100 if len(all_trajectories[env_idx]) > 0 else 0
            }
        
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
    
    print(f"\n{'='*60}")
    print(f"EXPLORATION STATISTICS SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(exploration_data['trajectories'])}")
    print(f"Total steps: {exploration_data['total_steps']}")
    print(f"Safety violations: {exploration_data['safety_violations']}")
    print(f"CBF preventions: {exploration_data['cbf_preventions']}")  # Add CBF stats
    
    if exploration_data['total_steps'] > 0:
        violation_rate = exploration_data['safety_violations'] / exploration_data['total_steps'] * 100
        cbf_rate = exploration_data['cbf_preventions'] / exploration_data['total_steps'] * 100
        print(f"Violation rate: {violation_rate:.1f}%")
        print(f"CBF intervention rate: {cbf_rate:.1f}%")
    
    # print trajectory detailed statistics
    if 'trajectory_stats' in exploration_data and exploration_data['trajectory_stats']:
        print("\n" + "="*60)
        print("TRAJECTORY DETAILED STATISTICS")
        print("="*60)
        
        # accumulators for calculating averages
        total_steps = 0
        total_path_length = 0.0
        total_initial_distance = 0.0
        total_final_distance = 0.0
        total_max_tilt_angle = 0.0
        total_avg_tilt_angle = 0.0
        total_unsafe_samples = 0
        total_cbf_interventions = 0  # Add CBF accumulator
        total_violation_rate = 0.0
        targets_reached = 0
        total_trajectories = 0
        
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
            # accumulate statistics
            total_trajectories += 1
            total_steps += stats['steps']
            total_path_length += stats['path_length']
            total_initial_distance += stats['initial_distance_to_goal']
            total_final_distance += stats['final_distance_to_goal']
            total_max_tilt_angle += stats['max_tilt_angle']
            total_avg_tilt_angle += stats['avg_tilt_angle']
            total_unsafe_samples += stats['unsafe_samples']
            total_cbf_interventions += stats.get('cbf_interventions', 0)  # Add CBF interventions
            total_violation_rate += stats['violation_rate']
            
            if stats.get('reached_target', False) and stats['unsafe_samples'] == 0:
                safe_and_reached += 1
            
            if stats.get('reached_target', False):
                targets_reached += 1
                success_path_length += stats['path_length']
                success_count += 1
            else:
                failure_path_length += stats['path_length']
                failure_count += 1
                failure_final_distance += stats['final_distance_to_goal']
                
            if stats['unsafe_samples'] > 0:
                unsafe_traj_count += 1
                unsafe_max_tilt_angle += stats['max_tilt_angle']
            else:
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
            avg_cbf_interventions = total_cbf_interventions / total_trajectories  # Add CBF average
            avg_violation_rate = total_violation_rate / total_trajectories
            target_success_rate = (targets_reached / total_trajectories) * 100
            safe_and_reached_rate = (safe_and_reached / total_trajectories) * 100
            avg_failure_final_distance = failure_final_distance / failure_count if failure_count > 0 else 0.0
            
            avg_safe_max_tilt_angle = safe_max_tilt_angle / safe_traj_count if safe_traj_count > 0 else 0.0
            avg_unsafe_max_tilt_angle = unsafe_max_tilt_angle / unsafe_traj_count if unsafe_traj_count > 0 else 0.0
            
            avg_success_path_length = success_path_length / success_count if success_count > 0 else 0.0
            avg_failure_path_length = failure_path_length / failure_count if failure_count > 0 else 0.0
            
            # print avg statistics
            print(f"\n{'='*60}")
            print(f"KEY METRICS")
            print(f"{'='*60}")
            print(f"Safe and reached success rate: {safe_and_reached_rate:.1f}%")
            print(f"Average final distance before failure: {avg_failure_final_distance:.2f} m")
            print(f"Target success rate: {target_success_rate:.1f}%")
            print(f"Safe trajectories: {safe_traj_count} ({safe_traj_count/total_trajectories*100:.1f}%)")
            print(f"Average CBF interventions per trajectory: {avg_cbf_interventions:.2f}")  # Add CBF metric
            print(f"Average final distance to goal: {avg_final_distance:.2f} m")    
            print(f"{'='*60}")

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
                'avg_cbf_interventions': avg_cbf_interventions,  # Add CBF to stats
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


def save_run_stats(all_stats, run_idx, base_dir="run_stats_with_cbf"):
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_dir}/run_{run_idx}_stats_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"Saved run {run_idx} statistics to: {filename}")
    return filename

def save_summary_stats(base_dir="run_stats_summary", run_params=None):
    """Save key statistical metrics to a separate file"""
    global exploration_data

    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_dir}/summary_stats_{timestamp}.txt"

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
        f.write(f"Average CBF interventions per trajectory: {stats['avg_cbf_interventions']:.2f}\n")
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
        f.write(f"  Average CBF interventions: {stats['avg_cbf_interventions']:.2f}\n")
        f.write(f"  Average violation rate: {stats['avg_violation_rate']:.1f}%\n")
        f.write(f"  Total trajectories counted: {stats['total_trajectories']}\n")
        f.write(f"  Unsafe trajectories: {stats['unsafe_trajectories']} ({stats['unsafe_trajectories']/stats['total_trajectories']*100:.1f}%)\n")
        f.write(f"  Environments that reached target: {stats['targets_reached']} out of {stats['total_trajectories']}\n")
        f.write(f"  Environments that failed to reach target: {stats['failure_count']} out of {stats['total_trajectories']}\n")

    print(f"Summary statistics saved to: {filename}")
    return filename

def main():
    """Main function to run RRT exploration with optional CBF safety filter."""
    global cbf_filter
    
    seed = args_cli.seed
    print(f"Setting random seed to: {seed}")
    
    import random
    random.seed(seed)
    np.random.seed(seed)
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
    
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
        env_cfg.sim.num_envs = args_cli.num_envs
    
    env_cfg.terminations.time_out = None
    
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # Initialize CBF Safety Filter if enabled
    if args_cli.enable_cbf:
        print(f"\n{'='*60}")
        print("INITIALIZING CBF SAFETY FILTER")
        print(f"{'='*60}")
        
        if args_cli.cbf_adaptive:
            print("Using Adaptive CBF Safety Filter")
            cbf_filter = AdaptiveCBFSafetyFilter(
                alpha_min=args_cli.cbf_alpha * 0.5,
                alpha_max=args_cli.cbf_alpha * 2.0,
                robot_radius=ROBOT_RADIUS,
                object_radius=OBJECT_RADIUS,
                safety_margin=args_cli.cbf_safety_margin,
                device=env.device
            )
        else:
            print("Using Standard CBF Safety Filter")
            cbf_filter = CBFSafetyFilter(
                alpha=args_cli.cbf_alpha,
                robot_radius=ROBOT_RADIUS,
                object_radius=OBJECT_RADIUS,
                safety_margin=args_cli.cbf_safety_margin,
                device=env.device
            )
        print(f"{'='*60}\n")
    else:
        print("\nCBF Safety Filter: DISABLED\n")
        cbf_filter = None
    
    safety_calc.config.max_tipping_angle_degrees = args_cli.max_tipping_angle
    safety_calc.config.use_closeness_cost = False 
    safety_calc.config.max_closeness_cost = 5.0  
    
    env.reset()
    
    noise_magnitude = args_cli.noise_magnitude
    total_runs = args_cli.total_runs
    max_steps = args_cli.max_steps
  
    for run_idx in range(total_runs):
        print(f"\n{'='*60}")
        print(f"STARTING RUN {run_idx+1}/{total_runs}")
        print(f"{'='*60}")
        
        env.reset()
        env_velocities.clear()
        env_velocity_steps.clear()
        
        # Reset CBF statistics for this run
        if cbf_filter is not None:
            cbf_filter.reset_statistics()
        
        for _ in range(3):
            env.sim.render()
        
        print(f"Calibrating end-effector height to {DEFAULT_PUSH_HEIGHT}...")
        zero_delta_pose = np.zeros(6)
        height_adjust_action = pre_process_actions(zero_delta_pose, env.num_envs, env.device, env)
        steps = 50
        
        for step in range(steps):
            step_action = height_adjust_action * (step + 1) / steps
            env.step(step_action)
            env.sim.render()
     
        # Run exploration with CBF
        all_trajectories = auto_explore_with_rrt(
            env, 
            max_steps=max_steps, 
            noise_magnitude=noise_magnitude,
            use_cbf=args_cli.enable_cbf
        )
     
        run_stats = {
            'run_id': run_idx + 1,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'cbf_enabled': args_cli.enable_cbf,
            'cbf_alpha': args_cli.cbf_alpha if args_cli.enable_cbf else None,
            'cbf_adaptive': args_cli.cbf_adaptive if args_cli.enable_cbf else None,
            'environments': {}
        }
        
        # Add CBF statistics to run stats
        if cbf_filter is not None:
            run_stats['cbf_statistics'] = cbf_filter.get_statistics()
        
        # extract environment statistics for this run
        for i, stats in enumerate(exploration_data['trajectory_stats'][-env.num_envs:]):
            env_idx = stats['env_id']
            run_stats['environments'][f'env_{env_idx}'] = stats
        
        # save statistics for this run
        save_run_stats(run_stats, run_idx + 1, base_dir=f"run_stats_with_cbf/seed_{seed}")
    
    print(f"\n{'='*60}")
    print(f"ALL {total_runs} EXPLORATION RUNS COMPLETED")
    print(f"{'='*60}\n")
    
    print_exploration_stats()
    run_params = {
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "scene_type": args_cli.scene_type,
        "task": args_cli.task,
        "max_tipping_angle": args_cli.max_tipping_angle,
        "max_steps": args_cli.max_steps,
        "noise_magnitude": args_cli.noise_magnitude,
        "enable_cbf": args_cli.enable_cbf,
        "cbf_alpha": args_cli.cbf_alpha,
        "cbf_safety_margin": args_cli.cbf_safety_margin,
        "cbf_adaptive": args_cli.cbf_adaptive,
    }
    save_summary_stats(base_dir=f"run_stats_summary_rrt_50obj/seed_{seed}", run_params=run_params)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()