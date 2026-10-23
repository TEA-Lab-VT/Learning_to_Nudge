import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any
from .config import RobotConfig
from .environment_manager import EnvironmentManager
from .geometry_calculator import GeometryCalculator

class RobotManager:

    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Initialize the Robot Manager
        
        Parameters:
            config: Robot configuration, creates a new configuration if None
        """
        self.config = config or RobotConfig()
        self.env_manager = EnvironmentManager()
        self.geometry_calc = GeometryCalculator()
    
    def get_ee_state(self, env) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            tuple: (ee_pos_w, ee_quat_w)
                - ee_pos_w: end effector positions, shape: (num_envs, 3)
                - ee_quat_w: end effector quaternions, shape: (num_envs, 4)
        """

        ee_frame_data = env.scene[self.config.ee_frame_name].data
        ee_pos_w = ee_frame_data.target_pos_w[..., 0, :]
        ee_quat_w = ee_frame_data.target_quat_w[..., 0, :]
        return ee_pos_w, ee_quat_w
    
    def get_ee_positions_batch(self, env) -> Optional[torch.Tensor]:

        ee_pos, _ = self.get_ee_state(env)
        return ee_pos

    def calculate_distances_to_objects_batch(self, env) -> Tuple[torch.Tensor, List[str], Optional[Dict[str, Any]]]:
        """
        Batch calculate distances from the end effector to all objects in all environments
    
        Returns:
            tuple: (distances_tensor, object_names, closest_objects_info)
                - distances_tensor: [num_envs, num_objects] distance tensor
                - object_names: list of object names
                - closest_objects_info: dict of closest object info for each environment
        """
        ee_positions = self.get_ee_positions_batch(env)  # [num_envs, 3]
 
        object_states_tensor, object_names = self.env_manager.get_object_states_batch(env)  # [num_envs, num_objects, 7]
        
        # Extract object positions [num_envs, num_objects, 3]
        object_positions = object_states_tensor[..., :3]
        
        # Expand ee_positions to match object_positions shape
        ee_positions_expanded = ee_positions.unsqueeze(1).expand(-1, object_positions.shape[1], -1)  # [num_envs, num_objects, 3]
        
        # Batch compute distances [num_envs, num_objects]
        distances_tensor = self.geometry_calc.calculate_distance_2d_batch(ee_positions_expanded, object_positions)
        
        # Find the closest object in each environment
        closest_objects_info = {}
        if distances_tensor.shape[1] > 0:
            closest_distances, closest_indices = torch.min(distances_tensor, dim=1)  # [num_envs]
            
            for env_idx in range(distances_tensor.shape[0]):
                closest_obj_idx = closest_indices[env_idx].item()
                closest_objects_info[env_idx] = {
                    "name": object_names[closest_obj_idx],
                    "distance": closest_distances[env_idx].item(),
                    "index": closest_obj_idx
                }
        
        return distances_tensor, object_names, closest_objects_info
    
    
 