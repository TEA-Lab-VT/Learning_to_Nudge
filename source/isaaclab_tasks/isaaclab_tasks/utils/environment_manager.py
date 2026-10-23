import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from .config import EnvironmentConfig
from .geometry_calculator import GeometryCalculator

class EnvironmentManager:
    """Environment state manager class - supports multi-environment batch processing"""
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize environment manager

        Args:
            config: environment configuration, default creates a new config
        """
        self.config = config or EnvironmentConfig()
        self.geometry_calc = GeometryCalculator()
    
    def get_object_states_batch(self, env) -> Tuple[torch.Tensor, List[str]]:
        """
        get states of all objects in the environment

        Returns:
            tuple: (object_states_tensor, object_names)
                - object_states_tensor: [num_envs, num_objects, 7] - [x,y,z,qx,qy,qz,qw]
                - object_names: list of object names
        """
        object_states_list = []
        object_names = []
        
        for obj_name in env.scene.rigid_objects.keys():
            # Skip target area
            if self.config.skip_target_area and obj_name == "target_area":
                continue
                
            obj = env.scene.rigid_objects[obj_name]
            
            # Get states for all environments [num_envs, 13] -> [num_envs, 7]
            pos_batch = obj.data.root_state_w[:, 0:3]  # [num_envs, 3]
            quat_batch = obj.data.root_state_w[:, 3:7]  # [num_envs, 4]
            
            # # Coordinate transformation (if needed)
            # if self.config.coordinate_transform:
            #     # Transform batched data
            #     pos_batch = self._transform_coordinates_batch(pos_batch)
            
            object_state_batch = torch.cat([pos_batch, quat_batch], dim=-1)  # [num_envs, 7]
            object_states_list.append(object_state_batch)
            object_names.append(obj_name)
        
        # Stack into [num_envs, num_objects, 7]
        object_states_tensor = torch.stack(object_states_list, dim=1)
        
        return object_states_tensor, object_names
    
    def get_object_states(self, env) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get states of all objects in the environment
        
        Args:
            env: Isaac Lab environment instance
            
        Returns:
            tuple: (object_states, object_names)
                - object_states: list of object states, each as [x,y,z,qx,qy,qz,qw]
                - object_names: list of object names
        """
        object_states_tensor, object_names = self.get_object_states_batch(env)
        
        # Return only the first environment's states
        object_states = []
        for i in range(object_states_tensor.shape[1]):
            state = object_states_tensor[0, i].cpu().numpy()
            object_states.append(state)
            
        return object_states, object_names
    
    def print_object_states(self, env, env_idx: int = 0):
        """
        Print the states of all objects in a specified environment
        """
        object_states_tensor, object_names = self.get_object_states_batch(env)
        
        print(f"\n==== Environment {env_idx} Object States ====")
        if object_states_tensor.shape[1] == 0:
            print("No objects found - please ensure the environment is loaded correctly")
        else:
            for i, name in enumerate(object_names):
                pos = object_states_tensor[env_idx, i, 0:3].cpu().numpy()
                print(f"{name}: Position X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
        print("==== End ====\n")
    
    def print_all_envs_summary(self, env):
        """
        Print summary information for all environments
        """
        object_states_tensor, object_names = self.get_object_states_batch(env)
        num_envs, num_objects = object_states_tensor.shape[:2]
        
        print(f"\n==== Multi-environment Summary ====")
        print(f"Number of environments: {num_envs}")
        print(f"Objects per environment: {num_objects}")
        print(f"Object names: {object_names}")
        
        if num_objects > 0:
            # Compute statistics for positions
            positions = object_states_tensor[..., :3]  # [num_envs, num_objects, 3]
            mean_pos = positions.mean(dim=0).cpu().numpy()  # [num_objects, 3]
            std_pos = positions.std(dim=0).cpu().numpy()    # [num_objects, 3]
            
            for i, name in enumerate(object_names):
                print(f"{name}: Mean position [{mean_pos[i, 0]:.3f}, {mean_pos[i, 1]:.3f}, {mean_pos[i, 2]:.3f}]")
                print(f"         Std deviation [{std_pos[i, 0]:.3f}, {std_pos[i, 1]:.3f}, {std_pos[i, 2]:.3f}]")
        print("==== End ====\n")

