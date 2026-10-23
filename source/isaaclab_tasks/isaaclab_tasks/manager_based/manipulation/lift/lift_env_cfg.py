# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.schemas import MassPropertiesCfg, CollisionPropertiesCfg

from . import mdp
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sim.spawners import CylinderCfg, CuboidCfg
import random
##
# Scene definition
##

# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
#     randomize_objects = EventTerm(
#         func=mdp.randomize_object_positions,
#         mode="reset",
#         params={"params": None}
#     )

    # initialize four cylinders
cylinder_positions = [
    [0.4, 0.1, 0.055],  # left front
    [0.6, 0.1, 0.055],  # right front
    [0.4, -0.1, 0.055], # left back
    [0.6, -0.1, 0.055],  # right back
    [0.6, -0.1, 0.055],
    [0.6, -0.1, 0.055],
    [0.6, -0.1, 0.055],
    [0.7, -0.1, 0.055],
    [0.8, -0.1, 0.055],
    [0.9, -0.1, 0.055]
]

# cylinder_colors = [
#     (0.8, 0.2, 0.2), # red
#     (0.2, 0.2, 0.8),  # green
#     (0.2, 0.2, 0.8),  # blue
#     (0.8, 0.8, 0.2), # yellow
#     (0.0, 0.0, 0.5) , #dark green
#     (0.5, 0.0, 0.0) #dark red

# ]
#color below is corresct ignore the above
cylinder_colors = [
    (0.8, 0.2, 0.2), # red
    (0.2, 0.8, 0.2),  # green
    (0.2, 0.2, 0.8),  # blue
    (0.8, 0.8, 0.2), # yellow
    (0.0, 0.2, 0.0) , #dark green
    (0.5, 0.0, 0.0)  #dark red
]
marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

@configclass
class FrankaCubeSceneLiftEnvCfg(InteractiveSceneCfg):
    # def __post_init__(self):
    #     print("==== FrankaCubeLiftEnvCfg __post_init__ called ====")
    #     # post init of parent
    #     super().__post_init__()


    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Set Franka as robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_positions[0], rot=[0, 0, 0, 1]),
        spawn=CylinderCfg(
            radius=0.05, #0.04
            height=0.2, #0.20
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=1.0), #2.0
            # mass_props = MassPropertiesCfg(density=1000.0),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cylinder_colors[3])
        ),
    )

    # create additional cylinders
    object2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_positions[1], rot=[0, 0, 0, 1]),
        spawn=CylinderCfg(
            radius=0.05, #0.05
            height=0.2,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=2000.0,
                max_linear_velocity=2000.0,
                max_depenetration_velocity=0.1,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=1.0),#1.5
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cylinder_colors[4])
        ),
    )

    object3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_positions[2], rot=[0, 0, 0, 1]),
        spawn=CylinderCfg(
            radius=0.05, #0.055
            height=0.2,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=2000.0,
                max_linear_velocity=2000.0,
                max_depenetration_velocity=0.1,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=1.0),#1.0
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cylinder_colors[3])
        ),
    )


    object4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cylinder_positions[3], rot=[0, 0, 0, 1]),
        spawn=CylinderCfg(
            radius=0.05,   #0.06
            height=0.2,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=2000.0,
                max_linear_velocity=2000.0,
                max_depenetration_velocity=0.1,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=1.0),#
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cylinder_colors[3])
        ),
    )


    # target area
    target_area = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetArea",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.8, -0.02, 0.22], rot=[0, 0, 0, 1]),#0.78, -0.036, 0.22
        spawn=CylinderCfg(
            radius=0.01,
            height=0.01,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=0,
                solver_velocity_iteration_count=0,
                max_angular_velocity=0.0,
                max_linear_velocity=0.0,
            ),
            mass_props=MassPropertiesCfg(mass=0.0),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.2, 0.0)
            )
        ),
    )

    # Listens to the required transforms

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    randomize_objects = EventTerm(
        func=mdp.randomize_object_positions,
        mode="reset",
        params={"params": None}
    )

    randomize_object1_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.3,2.0),  
            "operation": "abs",
            "distribution": "uniform"
        }
    )
    

    robot_physics_material_obj1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.5, 0.7),
            "dynamic_friction_range": (0.3, 0.49),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 50,
        },
    )



    randomize_object2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object2"),
            "mass_distribution_params": (1.3,2.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True
        }
    )
    
    robot_physics_material_obj2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object2"),
            "static_friction_range": (0.5, 0.7),
            "dynamic_friction_range": (0.3, 0.49),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 50,
        },
    )

    randomize_object3_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object3"),
            "mass_distribution_params": (1.3,2.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True
        }
    )
    

    robot_physics_material_obj3 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object3"),
            "static_friction_range": (0.5, 0.7),
            "dynamic_friction_range": (0.3, 0.49),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 50,
        },
    )

    randomize_object4_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object4"),
            "mass_distribution_params": (1.3,2.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True
        }
    )

    robot_physics_material_obj4 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object4"),
            "static_friction_range": (0.5, 0.7),
            "dynamic_friction_range": (0.3, 0.49),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 50,
        },
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

#     # domain randomization for object position
#     reset_object_position = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             # "pose_range": {"x": (-0.15, -0.1), "y": (-0.25, -0.2), "z": (0.0, 0.0)},
#             # "pose_range": {"x": (-0.05, -0.03), "y": (0.0,0.02), "z": (0.0, 0.0)},
#             "pose_range": {"x": (-0.03, 0.03), "y": (-0.05,0.05), "z": (0.0, 0.0)},
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object", body_names="Object"),
#         },
#     )
    
#     # domain randomization for second object
#     reset_object2_position = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             # "pose_range": {"x": (0.1, 0.15), "y": (-0.25, -0.2), "z": (0.0, 0.0)},
#             # "pose_range": {"x": (-0.03, -0.01), "y": (0.0,0.02), "z": (0.0, 0.0)},
#             "pose_range": {"x": (-0.05, 0.0), "y": (-0.1,0.02), "z": (0.0, 0.0)},
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object2", body_names="Object2"),
#         },
#     )
    
#     # domain randomization for object position
#     reset_object3_position = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             # "pose_range": {"x": (-0.15, -0.1), "y": (0.2, 0.25), "z": (0.0, 0.0)},
#             # "pose_range": {"x": (-0.05, -0.03), "y": (-0.1, -0.05), "z": (0.0, 0.0)},
#             "pose_range": {"x": (0.05, 0.07), "y": (-0.1,0.02), "z": (0.0, 0.0)},
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object3", body_names="Object3"),
#         },
#     )
    
#     # domain randomization for fourth object
#     reset_object4_position = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             # "pose_range": {"x": (0.1, 0.15), "y": (0.2, 0.25), "z": (0.0, 0.0)},
#             # "pose_range": {"x": (-0.03, -0.01), "y": (-0.1, -0.05), "z": (0.0, 0.0)},
#             "pose_range": {"x": (-0.05, 0.0), "y": (-0.1,0.02), "z": (0.0, 0.0)},
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object4", body_names="Object4"),
#         },
#     )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: FrankaCubeSceneLiftEnvCfg = FrankaCubeSceneLiftEnvCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # myObjectTableSceneCfg(ObjectTableSceneCfg),scene=myObjectTableSceneCfg
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625