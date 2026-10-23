import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any
from .config import SafetyConfig
from .environment_manager import EnvironmentManager
from .robot_manager import RobotManager
from .geometry_calculator import GeometryCalculator

class SafetyCalculator:
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """
        Parameters:
            config: safety configuration; creates a new configuration by default
        """
        self.config = config or SafetyConfig()
        self.env_manager = EnvironmentManager()
        self.robot_manager = RobotManager()
        self.geometry_calc = GeometryCalculator()
    
    def calculate_task_safety_cost_batch(self, env) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用四元数方法计算所有环境的倾斜安全成本
        
        Parameters:
            env: Isaac Lab environment instance
            
        Returns:
            tuple: (max_safety_costs, safety_costs_tensor)
                - max_safety_costs: maximum safety cost per environment [num_envs]
                - safety_costs_tensor: all object safety costs [num_envs, num_objects]
        """
        # 获取物体状态
        object_states_tensor, object_names = self.env_manager.get_object_states_batch(env)

        # 提取四元数 [num_envs, num_objects, 4]
        quats = object_states_tensor[..., 3:7]
        
        # 使用四元数直接计算tilt angle [num_envs, num_objects]
        tilt_angles = self.geometry_calc.calculate_tilt_angle_from_quaternion_batch(quats)
        
        # 安全成本就是tilt angle的绝对值
        safety_costs_tensor = torch.abs(tilt_angles)
        
        # 计算每个环境的最大安全成本
        max_safety_costs = torch.max(safety_costs_tensor, dim=1)[0]  # [num_envs]
        
        # print(f"Tilt angles (degrees): {torch.rad2deg(tilt_angles)}")
        # print(f"Max tilt per env (degrees): {torch.rad2deg(max_safety_costs)}")
        
        return max_safety_costs, safety_costs_tensor
    
    def calculate_current_closeness_cost_batch(self, env) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute current closeness safety cost in batch
        
        Parameters:
            env: Isaac Lab environment instance
            
        Returns:
            tuple: (closeness_safety_costs, all_object_closeness_costs)
                - closeness_safety_costs: closeness cost to the nearest object [num_envs]
                - all_object_closeness_costs: all object closeness costs [num_envs, num_objects]
        """
        # check if closeness cost is enabled
        if not self.config.use_closeness_cost:
            num_envs = env.num_envs
            device = env.device
            object_states_tensor, _ = self.env_manager.get_object_states_batch(env)
            num_objects = object_states_tensor.shape[1]
            return torch.zeros(num_envs, device=device), torch.zeros((num_envs, num_objects), device=device)
        
        # get end effector positions
        ee_positions = self.robot_manager.get_ee_positions_batch(env)
 
        # get object positions
        object_states_tensor, object_names = self.env_manager.get_object_states_batch(env)
        object_positions = object_states_tensor[..., :3]  # [num_envs, num_objects, 3]
        
        max_cost = self.config.max_closeness_cost
        
        # expand end effector positions to match object positions shape
        ee_positions_expanded = ee_positions.unsqueeze(1).expand(-1, object_positions.shape[1], -1)
        
        # calculate distances [num_envs, num_objects]
        distances = self.geometry_calc.calculate_distance_2d_batch(ee_positions_expanded, object_positions)
        
        # calculate all object closeness costs [num_envs, num_objects]
        all_object_closeness_costs = 1.0 / ((1.0 / max_cost) + distances)
        
        # calculate closeness cost to the nearest object [num_envs]
        closest_distances = torch.min(distances, dim=1)[0]  # [num_envs]
        closeness_safety_costs = 1.0 / ((1.0 / max_cost) + closest_distances)
        
        return closeness_safety_costs, all_object_closeness_costs
    
    def calculate_total_safety_cost_batch(self, env, radians: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total safety cost in batch
        
        Parameters:
            env: Isaac Lab environment instance
            radians: whether to return in radians; otherwise returns in degrees
            
        Returns:
            tuple: (total_safety_costs, object_safety_costs)
                - total_safety_costs: [num_envs]
                - object_safety_costs: [num_envs, num_objects]
        """
        # Get tipping safety costs (现在使用四元数方法)
        tipping_costs, object_tipping_costs = self.calculate_task_safety_cost_batch(env)
        
        # Convert units
        if not radians:
            tipping_costs = tipping_costs * (180.0 / torch.pi)
            object_tipping_costs = object_tipping_costs * (180.0 / torch.pi)
        
        # Add closeness cost if enabled
        if self.config.use_closeness_cost:
            closeness_costs, object_closeness_costs = self.calculate_current_closeness_cost_batch(env)
            
            total_costs = tipping_costs + closeness_costs
            total_object_costs = object_tipping_costs + object_closeness_costs
        else:
            total_costs = tipping_costs
            total_object_costs = object_tipping_costs
        
        return total_costs, total_object_costs
    
    def is_total_safety_violated_batch(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check total safety violation (including tipping and closeness) in batch
        
        Parameters:
            env: Isaac Lab environment instance
            
        Returns:
            tuple: (is_violated, max_tipping_costs, closeness_costs, total_costs)
                - is_violated: boolean tensor [num_envs]
                - max_tipping_costs: maximum tipping cost per environment [num_envs]
                - closeness_costs: closeness cost per environment [num_envs]
                - total_costs: total safety cost per environment [num_envs]
        """
        # get tipping safety costs (现在使用四元数方法)
        max_tipping_costs, _ = self.calculate_task_safety_cost_batch(env)
        tipping_threshold = torch.tensor(self.config.max_tipping_angle_radians, device=max_tipping_costs.device)
        is_tipping_violated = max_tipping_costs > tipping_threshold
        
        # get closeness safety costs
        if self.config.use_closeness_cost:
            closeness_costs, _ = self.calculate_current_closeness_cost_batch(env)
            # set closeness cost threshold (can be adjusted)
            closeness_threshold = torch.tensor(self.config.max_closeness_cost * 0.9, device=closeness_costs.device)
            is_closeness_violated = closeness_costs > closeness_threshold
        else:
            # if closeness cost is not enabled, create zero tensor
            closeness_costs = torch.zeros_like(max_tipping_costs)
            is_closeness_violated = torch.zeros_like(max_tipping_costs, dtype=torch.bool)
        
        # calculate total safety costs
        if self.config.use_closeness_cost:
            total_costs = max_tipping_costs + closeness_costs  # simple addition, can be adjusted
        else:
            total_costs = max_tipping_costs
        
        # merge violation results: any violation is considered a safety violation
        is_violated = torch.logical_or(is_tipping_violated, is_closeness_violated)
        
        return is_violated, max_tipping_costs, closeness_costs, total_costs