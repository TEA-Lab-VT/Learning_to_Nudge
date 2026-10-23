# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standard CBF (Control Barrier Function) Safety Filter
Uses Quadratic Programming to find the safest action that is closest to the desired action
while satisfying safety constraints.
"""

import torch
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional

class CBFSafetyFilter:
    """
    Standard CBF Safety Filter using QP optimization.
    
    The CBF ensures safety by enforcing:
        h_dot(x, u) + alpha * h(x) >= 0
    where h(x) is the barrier function and alpha is a class-K function parameter.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        robot_radius: float = 0.05,
        object_radius: float = 0.05,
        safety_margin: float = 0.02,
        cbf_gamma: float = 1.0,
        enable_velocity_damping: bool = True,
        max_action_magnitude: float = 0.05,
        device: str = "cuda:0"
    ):
        """
        Initialize CBF Safety Filter.
        
        Args:
            alpha: Class-K function parameter for CBF (higher = more conservative)
            robot_radius: Radius of the robot end-effector
            object_radius: Radius of objects to avoid
            safety_margin: Additional safety margin beyond radii
            cbf_gamma: Weight for CBF constraints in QP
            enable_velocity_damping: Whether to use velocity-based damping
            max_action_magnitude: Maximum allowed action magnitude
            device: Device for torch tensors
        """
        self.alpha = alpha
        self.robot_radius = robot_radius
        self.object_radius = object_radius
        self.safety_margin = safety_margin
        self.cbf_gamma = cbf_gamma
        self.enable_velocity_damping = enable_velocity_damping
        self.max_action_magnitude = max_action_magnitude
        self.device = device
        
        # Statistics
        self.total_filters = 0
        self.active_filters = 0
        self.qp_failures = 0
        
        print(f"CBF Safety Filter initialized:")
        print(f"  alpha={alpha}, robot_radius={robot_radius}, safety_margin={safety_margin}")
        print(f"  cbf_gamma={cbf_gamma}, velocity_damping={enable_velocity_damping}")
    
    def compute_barrier_function(
        self, 
        ee_pos: torch.Tensor, 
        object_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute barrier function h(x) for all objects.
        h(x) = distance_to_object - (robot_radius + object_radius + margin)
        
        Args:
            ee_pos: End-effector position [x, y, z]
            object_states: Object states [N, 7] (pos, quat)
            
        Returns:
            h_values: Barrier function values for each object [N]
            distances: Actual distances to each object [N]
        """
        # Extract object positions (only x, y for 2D navigation)
        object_positions = object_states[:, :2]  # [N, 2]
        ee_pos_2d = ee_pos[:2]  # [2]
        
        # Compute distances
        distances = torch.norm(object_positions - ee_pos_2d.unsqueeze(0), dim=1)
        
        # Compute barrier function
        safe_distance = self.robot_radius + self.object_radius + self.safety_margin
        h_values = distances - safe_distance
        
        return h_values, distances
    
    def compute_barrier_gradient(
        self,
        ee_pos: torch.Tensor,
        object_states: torch.Tensor,
        h_values: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of barrier function with respect to position.
        ∇h = (ee_pos - obj_pos) / ||ee_pos - obj_pos||
        
        Args:
            ee_pos: End-effector position [x, y, z]
            object_states: Object states [N, 7]
            h_values: Barrier function values [N]
            distances: Distances to objects [N]
            
        Returns:
            gradients: Barrier function gradients [N, 2]
        """
        object_positions = object_states[:, :2]  # [N, 2]
        ee_pos_2d = ee_pos[:2]  # [2]
        
        # Direction from object to ee
        directions = ee_pos_2d.unsqueeze(0) - object_positions  # [N, 2]
        
        # Normalize (gradient of distance)
        gradients = directions / (distances.unsqueeze(1) + 1e-6)
        
        return gradients
    
    def solve_cbf_qp_simple(
        self,
        desired_action: torch.Tensor,
        ee_pos: torch.Tensor,
        object_states: torch.Tensor,
        h_values: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified QP solver using only linear constraints (more robust).
        
        This version avoids potential issues with CVXPY by using a simpler formulation.
        """
        u_des = desired_action[:2].cpu().numpy().astype(np.float64)
        gradients = self.compute_barrier_gradient(ee_pos, object_states, h_values, distances)
        
        # Build constraint matrices manually
        A_list = []
        b_list = []
        
        cbf_threshold = 0.15
        for i, h_val in enumerate(h_values):
            h_np = float(h_val.cpu().item())
            
            if h_np < cbf_threshold:
                grad = gradients[i].cpu().numpy().astype(np.float64)
                
                if self.enable_velocity_damping:
                    damping_factor = max(0.5, min(1.5, 1.0 / (abs(h_np) + 0.01)))
                    alpha_effective = self.alpha * damping_factor
                else:
                    alpha_effective = self.alpha
                
                # Constraint: -grad^T @ u <= alpha * h
                A_list.append(-grad)
                b_list.append(alpha_effective * h_np)
        
        # Add box constraints: -max <= u_i <= max
        A_list.extend([
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0])
        ])
        b_list.extend([
            self.max_action_magnitude,
            self.max_action_magnitude,
            self.max_action_magnitude,
            self.max_action_magnitude
        ])
        
        if len(A_list) == 4:  # Only box constraints
            return desired_action
        
        try:
            A = np.vstack(A_list)
            b = np.array(b_list)
            
            # Solve using CVXPY with explicit matrix form
            u = cp.Variable(2)
            objective = cp.Minimize(cp.sum_squares(u - u_des))
            constraints = [A @ u <= b]
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=1000)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                filtered_u = u.value
                if filtered_u is not None:
                    result = torch.tensor(
                        [float(filtered_u[0]), float(filtered_u[1]), desired_action[2].item()],
                        dtype=torch.float32,
                        device=self.device
                    )
                    return result
            
            self.qp_failures += 1
            return torch.zeros_like(desired_action)
            
        except Exception as e:
            self.qp_failures += 1
            if self.qp_failures % 100 == 1:
                print(f"CBF QP simple solver error (count: {self.qp_failures}): {e}")
            return torch.zeros_like(desired_action)
    
    def solve_cbf_qp(
        self,
        desired_action: torch.Tensor,
        ee_pos: torch.Tensor,
        object_states: torch.Tensor,
        h_values: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve QP to find safe action closest to desired action.
        
        Optimization problem:
            min_u  ||u - u_desired||^2
            s.t.   ∇h · u + alpha * h >= 0  (for all objects)
        
        Args:
            desired_action: Desired action [3] (dx, dy, gripper)
            ee_pos: Current end-effector position [3]
            object_states: Object states [N, 7]
            h_values: Barrier function values [N]
            distances: Distances to objects [N]
            
        Returns:
            filtered_action: Safe filtered action [3]
        """
        # Use the simplified solver which is more robust
        return self.solve_cbf_qp_simple(
            desired_action, ee_pos, object_states, h_values, distances
        )
    
    def filter_action(
        self,
        desired_action: torch.Tensor,
        ee_pos: torch.Tensor,
        object_states: torch.Tensor,
        safety_threshold: float = 0.1
    ) -> Tuple[torch.Tensor, bool]:
        """
        Filter a single action using CBF constraints.
        
        Args:
            desired_action: Desired action [3]
            ee_pos: End-effector position [3]
            object_states: Object states [N, 7]
            safety_threshold: Threshold to trigger filtering
            
        Returns:
            filtered_action: Safe action [3]
            was_filtered: Whether action was modified
        """
        self.total_filters += 1
        
        # Compute barrier functions
        h_values, distances = self.compute_barrier_function(ee_pos, object_states)
        
        # Check if filtering is needed
        min_h = torch.min(h_values)
        
        if min_h > safety_threshold:
            # Safe - no filtering needed
            return desired_action, False
        
        # Unsafe - apply CBF filter
        self.active_filters += 1
        filtered_action = self.solve_cbf_qp(
            desired_action, ee_pos, object_states, h_values, distances
        )
        
        was_filtered = torch.norm(filtered_action - desired_action) > 1e-4
        
        return filtered_action, was_filtered
    
    def filter_actions_batch(
        self,
        desired_actions: torch.Tensor,
        ee_positions: torch.Tensor,
        object_states_batch: torch.Tensor,
        safety_threshold: float = 0.1
    ) -> Tuple[torch.Tensor, int]:
        """
        Filter actions for a batch of environments.
        
        Args:
            desired_actions: Desired actions [B, 3]
            ee_positions: End-effector positions [B, 3]
            object_states_batch: Object states [B, N, 7]
            safety_threshold: Threshold to trigger filtering
            
        Returns:
            filtered_actions: Safe actions [B, 3]
            num_filtered: Number of actions that were filtered
        """
        batch_size = desired_actions.shape[0]
        filtered_actions = []
        num_filtered = 0
        
        for i in range(batch_size):
            filtered_action, was_filtered = self.filter_action(
                desired_actions[i],
                ee_positions[i],
                object_states_batch[i],
                safety_threshold
            )
            filtered_actions.append(filtered_action)
            
            if was_filtered:
                num_filtered += 1
        
        return torch.stack(filtered_actions), num_filtered
    
    def get_statistics(self) -> dict:
        """Get filter statistics."""
        return {
            'total_filters': self.total_filters,
            'active_filters': self.active_filters,
            'qp_failures': self.qp_failures,
            'filter_rate': self.active_filters / max(1, self.total_filters) * 100
        }
    
    def reset_statistics(self):
        """Reset filter statistics."""
        self.total_filters = 0
        self.active_filters = 0
        self.qp_failures = 0
    
    def print_statistics(self):
        """Print filter statistics."""
        stats = self.get_statistics()
        print(f"\nCBF Safety Filter Statistics:")
        print(f"  Total filter calls: {stats['total_filters']}")
        print(f"  Active filters: {stats['active_filters']}")
        print(f"  QP failures: {stats['qp_failures']}")
        print(f"  Filter rate: {stats['filter_rate']:.1f}%")


class AdaptiveCBFSafetyFilter(CBFSafetyFilter):
    """
    Adaptive CBF Safety Filter that adjusts alpha based on environment state.
    More conservative when near obstacles, more permissive when far away.
    """
    
    def __init__(
        self,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
        **kwargs
    ):
        """
        Initialize Adaptive CBF Safety Filter.
        
        Args:
            alpha_min: Minimum alpha value (far from obstacles)
            alpha_max: Maximum alpha value (near obstacles)
            **kwargs: Additional arguments passed to CBFSafetyFilter
        """
        super().__init__(**kwargs)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.base_alpha = self.alpha
        
    def compute_adaptive_alpha(self, h_values: torch.Tensor) -> float:
        """
        Compute adaptive alpha based on minimum barrier value.
        
        Args:
            h_values: Barrier function values
            
        Returns:
            adaptive_alpha: Adapted alpha value
        """
        min_h = float(torch.min(h_values).cpu().item())
        
        # Linear interpolation based on minimum barrier value
        if min_h > 0.15:
            # Far from obstacles - use minimum alpha
            return self.alpha_min
        elif min_h < 0.0:
            # Very close or penetrating - use maximum alpha
            return self.alpha_max
        else:
            # Interpolate
            ratio = 1.0 - (min_h / 0.15)
            return self.alpha_min + ratio * (self.alpha_max - self.alpha_min)
    
    def filter_action(
        self,
        desired_action: torch.Tensor,
        ee_pos: torch.Tensor,
        object_states: torch.Tensor,
        safety_threshold: float = 0.1
    ) -> Tuple[torch.Tensor, bool]:
        """Filter action with adaptive alpha."""
        # Compute barrier functions
        h_values, distances = self.compute_barrier_function(ee_pos, object_states)
        
        # Adapt alpha based on current state
        self.alpha = self.compute_adaptive_alpha(h_values)
        
        # Use parent class filtering with adapted alpha
        return super().filter_action(
            desired_action, ee_pos, object_states, safety_threshold
        )