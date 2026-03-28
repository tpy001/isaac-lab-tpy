# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .events import randomize_dead_zone
from .realsim_tasks_cfg import RealSimTask,RealSimPegInsert,RealSimGearMesh, RealSimNutThread
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from .policy.configuration_pi0remote import PI0RemoteConfig, PI0RemoteTAVLAConfig



OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class RealSimCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class RealSimObsRandCfg(ObsRandCfg):
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
            "static_friction_range": (0.25, 1.25),
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
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)
    )


@configclass
class RealSimEnvCfg(FactoryEnvCfg):
    decimation = 4
    seed = 0
    action_space: int = 7
    obs_rand: RealSimObsRandCfg = RealSimObsRandCfg()
    ctrl: RealSimCtrlCfg = RealSimCtrlCfg()
    task: RealSimTask = RealSimTask()
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
        prim_path="/World/envs/env_.*/franka_env/Robot/franka",
        spawn = None,
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=True,
        #         max_depenetration_velocity=5.0,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=3666.0,
        #         enable_gyroscopic_forces=True,
        #         solver_position_iteration_count=192,
        #         solver_velocity_iteration_count=1,
        #         max_contact_impulse=1e32,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False,
        #         solver_position_iteration_count=192,
        #         solver_velocity_iteration_count=1,
        #     ),
        #     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        # ),
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "panda_joint1": 0.00871,
            #     "panda_joint2": -0.10368,
            #     "panda_joint3": -0.00794,
            #     "panda_joint4": -1.49139,
            #     "panda_joint5": -0.00083,
            #     "panda_joint6": 1.38774,
            #     "panda_joint7": 0.0,
            #     "panda_finger_joint2": 0.04,
            # },
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2":  -0.7853981634,
                "panda_joint3": 0.0,
                "panda_joint4": -2.3561944902,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5707963268,
                "panda_joint7":  0.7853981634,
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
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    
    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/franka_env/Robot/franka/panda_link7/panda_link8/panda_hand/wrist_camera",
        update_period=0,
        height=360,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.07813, -0.00845, -0.0073), 
            rot=(0.12057, 0.71266, 0.68644, 0.07985), 
            convention="opengl"
        ),
    )

    tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/franka_env/front_camera",  # 对应你USD里相机的实际路径
        update_period=0,
        height=360,
        width=640,
        data_types=["rgb"],
        spawn=None,  # ← 关键：不重新创建，直接用USD里的
    )
    
    disable_xy_rot = True
    data_collect_cfg = {
        "collect_data": False,          # Enable data collection during execution
        "num_trajectories": 100,           # Number of trajectories to collect
        "save_failed_trajectory": False,  # Save trajectories even if the task fails
        "immediate_stop": False        # Reset the environment immediately once the task succeeds
    }
    policy_cfg = None
    # policy_cfg = PI0RemoteConfig(n_action_steps = 32)
    # policy_cfg = PI0RemoteTAVLAConfig(num_history_steps=32, history_step_interval=1, n_action_steps=32)
    


@configclass
class RealSimTaskPegInsertCfg(RealSimEnvCfg):
    task_name = "peg_insert"
    task_prompt = "place a peg in a hole"
    task = RealSimPegInsert()
    disable_xy_rot = False
    episode_length_s = 20.0


@configclass
class RealSimTaskGearMeshCfg(RealSimEnvCfg):
    task_name = "gear_mesh"
    task_prompt = "Install the gear between the two gears."
    task = RealSimGearMesh()
    episode_length_s = 20.0


@configclass
class RealSimTaskNutThreadCfg(RealSimEnvCfg):
    task_name = "nut_thread"
    task_prompt = "Thread the nut onto the bolt until it is fully tightened."
    task = RealSimNutThread()
    episode_length_s = 30.0
