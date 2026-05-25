# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import (
    ForgeGearMesh,
    ForgeNutThread,
    ForgePegInsert,
    ForgeTask,
)
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from .policy.configuration_pi0remote import PI0RemoteConfig,PI0RemoteTAVLAConfig


OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    
    # pos_action_threshold = [0.01, 0.01, 0.01]
    # pos_action_threshold = [0.005, 0.005, 0.005]
    pos_action_threshold = [0.01, 0.01, 0.01]
    


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    decimation = 4
    # seed = 0 # 采数据时用 0
    seed = 1 # 测试时用 1
    # control_mode: str = "position"
    control_mode: str = "torque"
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]
    
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd",
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                # stiffness=800.0,
                # damping=40.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                # stiffness=800.0,
                # damping=40.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    
    rl_training = False
    if not rl_training:
        wrist_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 1.0e5)
            ),
            # 使用你提供的特定 offset
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.07813, -0.00845, -0.0073), 
                rot=(0.12057, 0.71266, 0.68644, 0.07985), 
                convention="opengl"
            ),
        )

        # 2. 固定位相机配置 (Static/Fixed Camera)
        tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 1.0e5)
            ),
            # 使用你提供的特定 offset
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.29, -0.09, 0.4), 
                rot=(0.61, 0.4278, 0.347, 0.569), 
                convention="opengl"
            )
        )
    
    disable_xy_rot = True
    # Maximum random rotation magnitude (degrees) for peg_insert held asset initialization.
    peg_insert_rot_noise_deg: float = 0.0
    # Motion-planner debugging helpers.
    debug_motion_planner: bool = False
    debug_motion_planner_print_interval: int = 20
    debug_motion_planner_visualize: bool = False
    planner_to_rl_handoff: bool = False
    planner_to_rl_pos_tol: float = 0.005
    # planner_to_rl_rot_tol_deg: float = 0.5
    planner_to_rl_rot_tol_deg: float = 5
    
    collect_data = False if rl_training else True
    immediate_stop = False if rl_training else True
    data_collect_cfg = {
        "collect_data": collect_data,          # Enable data collection during execution
        "num_trajectories": 100,            # Number of trajectories to collect
        "save_failed_trajectory": True,  # Save trajectories even if the task fails
        "immediate_stop": immediate_stop        # Reset the environment immediately once the task succeeds
    }
    
    # policy_cfg = None
    # policy_cfg = PI0RemoteConfig(n_action_steps = 32)
    # policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=32, history_step_interval=1, n_action_steps=32)
    policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=10, history_step_interval=4, n_action_steps=10)

    def __post_init__(self,stiffness=800.0,damping=40.0):
        
        super().__post_init__()
        control_mode = str(self.control_mode).lower()
        if control_mode not in {"position", "torque"}:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}. Expected 'position' or 'torque'.")

        arm_stiffness = stiffness if control_mode == "position" else 0.0
        arm_damping = damping if control_mode == "position" else 0.0

        self.robot.actuators["panda_arm1"].stiffness = arm_stiffness
        self.robot.actuators["panda_arm1"].damping = arm_damping
        self.robot.actuators["panda_arm2"].stiffness = arm_stiffness
        self.robot.actuators["panda_arm2"].damping = arm_damping


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task_prompt = "place a peg in a hole"
    task = ForgePegInsert(
        # peg_shape="round",
        # peg_diameter_mm=8,
        use_industreal_obj_assets=False,
        success_threshold=0.04,
    )
    disable_xy_rot = False
    peg_insert_rot_noise_deg = 30
    episode_length_s = 10.0  # 测试时用 20，RL agent 训练时用 10

    def __post_init__(self):
        super().__post_init__()
        self.task.success_threshold = 0.04 if self.rl_training else 0.2
        self.episode_length_s = 10.0 if self.rl_training else 20.0
        # self.episode_length_s = 20.0 if self.rl_training else 20.0

    
@configclass
class ForgeTaskPegSquareInsertCfg(ForgeEnvCfg):
    task_name = "peg_square_insert"
    # task_prompt = "place a rectangle peg in a hole"
    task_prompt = "insert a peg into a hole"
    task = ForgePegInsert(
        peg_shape="rectangular",
        peg_diameter_mm=8,
        use_industreal_obj_assets=True,
        success_threshold=0.04,
    )
    
    disable_xy_rot = False
    peg_insert_rot_noise_deg = 0
    episode_length_s = 10.0  # 测试时用 20，RL agent 训练时用 10
    max_force = 25
    
    def __post_init__(self):
        super().__post_init__()
        self.task.success_threshold = 0.04 if self.rl_training else 0.2
        # self.task.success_threshold = 0.2 if self.rl_training else 0.2
        self.episode_length_s = 10.0 if self.rl_training else 20.0
        # self.episode_length_s = 20.0 if self.rl_training else 15.0 # 采集数据时设置成15确保质量


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task_prompt = "Install the gear between the two gears."
    task = ForgeGearMesh()
    episode_length_s = 20.0

@configclass
class ForgeTaskGearAssemblyCfg(ForgeTaskGearMeshCfg):
    task_name = "gear_assembly"
    task_prompt = "Pick up the gear on the table and install it between the two gears."
    planner_to_rl_handoff = False
    # episode_length_s = 30.0
    episode_length_s = 20.0
    gear_init_min_distance = 0.12 # 齿轮会被随机放置到距离底座中心 15 cm - 20 cm 的范围内
    gear_init_max_distance = 0.15
    gear_length = 0.15 # 齿轮底座的长度
    gear_width = 0.075 # 齿轮底座的宽度
    
@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task_prompt = "Thread the nut onto the bolt until it is fully tightened."
    # task = ForgeNutThread(success_threshold = 0.1)
    task = ForgeNutThread()
    episode_length_s = 30.0
    
