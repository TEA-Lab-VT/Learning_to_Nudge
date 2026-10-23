import numpy as np
import torch
from typing import Tuple, List, Union

class GeometryCalculator:

    @staticmethod
    def quaternion_to_z_axis_batch(q: torch.Tensor) -> torch.Tensor:
        """
        从四元数提取物体的z轴方向（旋转矩阵的第三列）
        
        Parameters:
            q: quaternion tensor [..., 4] - [x, y, z, w]
            
        Returns:
            torch.Tensor: z axis direction [..., 3]
        """
        # x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # 旋转矩阵的第三列（z轴方向）
        z_axis_x = 2 * (x * z + w * y)
        z_axis_y = 2 * (y * z - w * x)
        z_axis_z = 1 - 2 * (x * x + y * y)
        
        z_axis = torch.stack([z_axis_x, z_axis_y, z_axis_z], dim=-1)
        return z_axis

    @staticmethod
    def calculate_tilt_angle_from_quaternion_batch(q: torch.Tensor) -> torch.Tensor:
        """
        使用四元数计算tilt angle（物体z轴与世界z轴的夹角）
        
        Parameters:
            q: quaternion tensor [..., 4] - [x, y, z, w]
            
        Returns:
            torch.Tensor: tilt angles in radians [...]
        """
        # 获取物体的z轴方向
        object_z_axis = GeometryCalculator.quaternion_to_z_axis_batch(q)
        
        # 世界坐标系的z轴方向 [0, 0, 1]
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
        
        # 计算点积（z轴分量）
        dot_product = object_z_axis[..., 2]  # 只需要z分量，因为world_z_axis是[0,0,1]
        
        # 限制dot_product的范围在[-1, 1]以避免数值误差
        dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)
        
        # 计算夹角
        tilt_angle = torch.acos(torch.abs(dot_product))  # 使用abs确保得到最小角度
        
        return tilt_angle


    @staticmethod
    def calculate_distance_2d_batch(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """
        Compute batch distances between points in the x,y plane
        
        Parameters:
            pos1: positions [..., 3] 
            pos2: positions [..., 3]
            
        Returns:
            torch.Tensor: distances [...]
        """
        diff = pos1[..., :2] - pos2[..., :2]
        return torch.linalg.norm(diff, dim=-1)

    @staticmethod
    def calculate_distance_2d(pos1: np.ndarray, pos2: np.ndarray) -> float:
        diff = np.array(pos1[:2]) - np.array(pos2[:2])
        return np.linalg.norm(diff)

    
    # @staticmethod
    # def quaternion_to_euler_batch(q: torch.Tensor) -> torch.Tensor:
    #     """
    #     Parameters:
    #         q: quaternion tensor [batch_size, 4] or [num_envs, num_objects, 4] - [x, y, z, w]
            
    #     Returns:
    #         torch.Tensor: Euler angles [batch_size, 3] or [num_envs, num_objects, 3] - [roll, pitch, yaw]
    #     """
    #     original_shape = q.shape
    #     if len(original_shape) > 2:
    #         # Flatten to [batch_size, 4]
    #         q = q.view(-1, 4)
        
    #     x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    #     # Roll (around X axis)
    #     sinr_cosp = 2 * (w * x + y * z)
    #     cosr_cosp = 1 - 2 * (x*x + y*y)
    #     roll = torch.atan2(sinr_cosp, cosr_cosp)

    #     # Pitch (around Y axis)
    #     sinp = 2 * (w * y - z * x)
    #     # Handle singularities
    #     pitch = torch.where(
    #         torch.abs(sinp) >= 1,
    #         torch.sign(sinp) * (torch.pi / 2),
    #         torch.asin(sinp)
    #     )

    #     # Yaw (around Z axis)
    #     siny_cosp = 2 * (w * z + x * y)
    #     cosy_cosp = 1 - 2 * (y*y + z*z)
    #     yaw = torch.atan2(siny_cosp, cosy_cosp)

    #     euler = torch.stack([roll, pitch, yaw], dim=-1)
        
    #     # Restore original shape
    #     if len(original_shape) > 2:
    #         target_shape = list(original_shape[:-1]) + [3]
    #         euler = euler.view(target_shape)
            
    #     return euler

    # @staticmethod
    # def calculate_total_tilt_batch(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    #     """     
    #     Parameters:
    #         roll: roll angle (radians) 
    #         pitch: pitch angle (radians) 
    #     """
    #     return torch.sqrt(roll**2 + pitch**2)

    # @staticmethod
    # def calculate_total_tilt(roll: float, pitch: float) -> float:
    #     return np.sqrt(roll**2 + pitch**2)


    # @staticmethod
    # def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    #     x, y, z, w = q

    #     # Roll (around X axis)
    #     sinr_cosp = 2 * (w * x + y * z)
    #     cosr_cosp = 1 - 2 * (x*x + y*y)
    #     roll = np.arctan2(sinr_cosp, cosr_cosp)

    #     # Pitch (around Y axis)
    #     sinp = 2 * (w * y - z * x)
    #     if abs(sinp) >= 1:
    #         pitch = np.sign(sinp) * (np.pi / 2)
    #     else:
    #         pitch = np.arcsin(sinp)

    #     # Yaw (around Z axis)
    #     siny_cosp = 2 * (w * z + x * y)
    #     cosy_cosp = 1 - 2 * (y*y + z*z)
    #     yaw = np.arctan2(siny_cosp, cosy_cosp)

    #     return roll, pitch, yaw

    # @staticmethod
    # def calculate_tilt_angle_from_quaternion(q: np.ndarray) -> float:
    #     """
    #     使用四元数计算单个物体的tilt angle
        
    #     Parameters:
    #         q: quaternion array [x, y, z, w]
            
    #     Returns:
    #         float: tilt angle in radians
    #     """
    #     x, y, z, w = q
        
    #     # 计算物体的z轴方向
    #     z_axis_x = 2 * (x * z + w * y)
    #     z_axis_y = 2 * (y * z - w * x)
    #     z_axis_z = 1 - 2 * (x * x + y * y)
        
    #     # 计算与世界z轴的夹角
    #     dot_product = z_axis_z  # 与[0,0,1]的点积就是z分量
    #     dot_product = np.clip(dot_product, -1.0, 1.0)  # 限制范围
        
    #     tilt_angle = np.arccos(np.abs(dot_product))
    #     return tilt_angle