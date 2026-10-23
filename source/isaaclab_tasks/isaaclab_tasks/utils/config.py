from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SafetyConfig:
    """Safety calculation configuration class"""
    max_closeness_cost: float = 15.0
    # max_closeness_cost: float = 5.0
    use_closeness_cost: bool = False
    max_tipping_angle_degrees: float = 20.0
    
    @property
    def max_tipping_angle_radians(self) -> float:
        """Maximum tipping angle (radians)"""
        return np.radians(self.max_tipping_angle_degrees)

@dataclass
class RobotConfig:
    """Robot configuration class"""
    ee_frame_name: str = "ee_frame"
    target_frame_name: str = "end_effector"

@dataclass
class EnvironmentConfig:
    """Environment configuration class"""
    skip_target_area: bool = True
    coordinate_transform: bool = False