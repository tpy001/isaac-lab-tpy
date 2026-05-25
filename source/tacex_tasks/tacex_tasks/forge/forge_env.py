# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import os
import shutil
from PIL import Image
import csv
import cv2
import time

from .isaac_forge_env import ForgeEnv as IsaacForgeEnv
from .forge_env_cfg import ForgeEnvCfg
from isaaclab.sensors import TiledCamera
import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_control, factory_utils
import isaaclab.sim as sim_utils
import carb
from .utils import MotionPolicy,PoseTrajectoryPlanner
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers



class ForgeEnv(IsaacForgeEnv):
    """
    ForgeEnv extension for data collection.
    
    Inherits from ForgeEnv and adds functionality to collect and save:
    - Camera images (front camera, wrist camera)
    - Joint states
    - Gripper states
    - End-effector poses
    - Force/torque sensor data
    - Actions
    """
    
    cfg: ForgeEnvCfg

    def __init__(
        self, 
        cfg: ForgeEnvCfg, 
        render_mode: str | None = None,
        output_dir: str = "./data",
        **kwargs
    ):
        """
        Initialize the data collection environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode
            collect_data: Whether to collect data during runtime
            output_dir: Directory to save collected data
            **kwargs: Additional arguments
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # 创建一个专用的随机数生成器
        # 这里的 seed 可以从 cfg 读取，或者固定
        seed = self.cfg.seed if hasattr(self.cfg, "seed") and self.cfg.seed is not None else 42
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        
        
        # ========== 新增: Policy初始化 ==========
        if cfg.policy_cfg:
            # 根据配置类型选择对应的远程策略：
            # - PI0RemoteConfig -> PI0RemotePolicy（无历史 effort）
            # - PI0RemoteTAVLAConfig -> PI0RemotePolicyTAVLA（带历史 effort）
            from .policy.configuration_pi0remote import PI0RemoteConfig as _CfgBase, PI0RemoteTAVLAConfig as _CfgTavla
            from .policy.modeling_pi0remote import PI0RemotePolicy, PI0RemotePolicyTAVLA
            if isinstance(cfg.policy_cfg, _CfgTavla):
                self.policy = PI0RemotePolicyTAVLA(cfg.policy_cfg)
                print("Using Pi0 TAVLA Policy")
            elif isinstance(cfg.policy_cfg, _CfgBase):
                self.policy = PI0RemotePolicy(cfg.policy_cfg)
                print("Using Pi0 Policy")
            else:
                # 兜底：未知配置类型时，仍按基础 PI0RemotePolicy 处理
                self.policy = PI0RemotePolicy(cfg.policy_cfg)
                print("Using Pi0 Policy (fallback)")
        else:
            self.policy = None
        
        self.next_action = []  # 存储policy输出的action
        # ========================================
    
        self.collect_data = cfg.data_collect_cfg["collect_data"]
        self.immediate_stop = cfg.data_collect_cfg["immediate_stop"]
        self.save_failed_trajectory = cfg.data_collect_cfg["save_failed_trajectory"]
        self.num_trajectories = cfg.data_collect_cfg["num_trajectories"]
        self.cur_num_traj = 0

        self.output_dir = output_dir

        self.motion_planner = None
        self.motion_planner_last_action = None
        self.motion_planner_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.motion_planner_target_ee = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.debug_motion_planner = bool(getattr(self.cfg, "debug_motion_planner", False))
        self.debug_motion_planner_print_interval = int(getattr(self.cfg, "debug_motion_planner_print_interval", 20))
        self.debug_motion_planner_visualize = bool(getattr(self.cfg, "debug_motion_planner_visualize", True))
        self.planner_to_rl_handoff = bool(getattr(self.cfg, "planner_to_rl_handoff", True))
        self.planner_to_rl_pos_tol = float(getattr(self.cfg, "planner_to_rl_pos_tol", 0.05))
        self.planner_to_rl_rot_tol = float(
            np.deg2rad(getattr(self.cfg, "planner_to_rl_rot_tol_deg", 60.0))
        )
        self.motion_planner_debug = {
            "valid": torch.zeros((self.num_envs,), dtype=torch.bool, device="cpu"),
            "cur_ee_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "target_ee_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "target_held_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "fixed_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "held_to_tcp_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "next_action": torch.zeros((self.num_envs, 8), dtype=torch.float32, device="cpu"),
            "ctrl_target_pose": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "joint_pos": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
            "ctrl_target_joint_pos": torch.zeros((self.num_envs, 7), dtype=torch.float32, device="cpu"),
        }
        self._motion_planner_visualizers = {}
        if self.collect_data:
            # Initialize data buffers for each environment
            self.data_buffers = [
                {
                    "camera": {
                        "front": [],
                        "wrist": [],
                    },
                    "joints": [],
                    "gripper": [],
                    "ee_pose": [],
                    "force": [],  # Force/torque sensor data
                    "force_world": [],  # Force in world frame
                    "actions": []
                }
                for _ in range(self.num_envs)
            ]
            self.reset_data_buffer()
            self.motion_planner = [MotionPolicy() for _ in range(self.num_envs)]
            self.motion_planner_last_action = [None for _ in range(self.num_envs)]
            self.motion_planner_active[:] = True
        if self.debug_motion_planner and self.debug_motion_planner_visualize:
            self._init_motion_planner_debug_vis()
        
        self.success_times = 0
        self.total_times = 0
        self._init_experiment_logging()
        # episode_start flag for episode-streaming style policies (CPU-side, one bool per env)
        self._episode_start = torch.ones((self.num_envs,), dtype=torch.bool, device="cpu")
        self.control_mode = str(getattr(self.cfg, "control_mode", "position")).lower()
        if self.control_mode not in {"position", "torque"}:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}. Expected 'position' or 'torque'.")
        
        
    def _setup_scene(self):
        super()._setup_scene()
        # sensors
        if hasattr(self.cfg, "wrist_camera") and self.cfg.wrist_camera is not None:
            self.wrist_tiled_camera = TiledCamera(self.cfg.wrist_camera)
            self.scene.sensors["wrist_tiled_camera"] = self.wrist_tiled_camera

        if hasattr(self.cfg, "tiled_camera") and self.cfg.tiled_camera is not None:
            self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self.tiled_camera

    def _init_motion_planner_debug_vis(self):
        marker_specs = {
            "current_ee": "/Visuals/MotionPlanner/current_ee",
            "target_ee": "/Visuals/MotionPlanner/target_ee",
            "target_held": "/Visuals/MotionPlanner/target_held",
            "fixed_frame": "/Visuals/MotionPlanner/fixed_frame",
        }
        for key, prim_path in marker_specs.items():
            marker_cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
            marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
            self._motion_planner_visualizers[key] = VisualizationMarkers(marker_cfg)

    def _update_motion_planner_debug_vis(self):
        if not self.debug_motion_planner or not self._motion_planner_visualizers:
            return

        valid_mask = self.motion_planner_debug["valid"]
        if not torch.any(valid_mask):
            return

        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        for key, visualizer in self._motion_planner_visualizers.items():
            pose_key = {
                "current_ee": "cur_ee_pose",
                "target_ee": "target_ee_pose",
                "target_held": "target_held_pose",
                "fixed_frame": "fixed_pose",
            }[key]
            poses = self.motion_planner_debug[pose_key][valid_indices]
            visualizer.visualize(translations=poses[:, :3], orientations=poses[:, 3:])

    def _print_motion_planner_debug(self):
        if not self.debug_motion_planner:
            return
        interval = max(1, self.debug_motion_planner_print_interval)
        if self.common_step_counter > 20 and self.common_step_counter % interval != 0:
            return

        valid_mask = self.motion_planner_debug["valid"]
        if not torch.any(valid_mask):
            return

        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        current_pose = torch.cat(
            [
                self.fingertip_midpoint_pos[valid_indices].to("cpu"),
                self.fingertip_midpoint_quat[valid_indices].to("cpu"),
            ],
            dim=1,
        )
        self.motion_planner_debug["cur_ee_pose"][valid_indices] = current_pose

        env_idx = int(valid_mask.nonzero(as_tuple=False)[0].item())
        cur_pose = self.motion_planner_debug["cur_ee_pose"][env_idx].tolist()
        target_ee_pose = self.motion_planner_debug["target_ee_pose"][env_idx].tolist()
        target_held_pose = self.motion_planner_debug["target_held_pose"][env_idx].tolist()
        fixed_pose = self.motion_planner_debug["fixed_pose"][env_idx].tolist()
        held_to_tcp_pose = self.motion_planner_debug["held_to_tcp_pose"][env_idx].tolist()
        next_action = self.motion_planner_debug["next_action"][env_idx].tolist()
        ctrl_target_pose = self.motion_planner_debug["ctrl_target_pose"][env_idx].tolist()
        joint_pos = self.motion_planner_debug["joint_pos"][env_idx].tolist()
        ctrl_target_joint_pos = self.motion_planner_debug["ctrl_target_joint_pos"][env_idx].tolist()
        pos_err = (
            self.motion_planner_debug["ctrl_target_pose"][env_idx, :3]
            - self.motion_planner_debug["cur_ee_pose"][env_idx, :3]
        ).tolist()
        rot_err_deg = float(
            torch.rad2deg(
                self._quat_angle_error_wxyz(
                    self.motion_planner_debug["cur_ee_pose"][env_idx, 3:].unsqueeze(0),
                    self.motion_planner_debug["ctrl_target_pose"][env_idx, 3:].unsqueeze(0),
                )
            ).item()
        )
        print(
            f"[MotionPlanner][env={env_idx}][step={int(self.common_step_counter)}]\n"
            f"  current_ee  pos={cur_pose[:3]} quat={cur_pose[3:]}\n"
            f"  target_ee   pos={target_ee_pose[:3]} quat={target_ee_pose[3:]}\n"
            f"  target_held pos={target_held_pose[:3]} quat={target_held_pose[3:]}\n"
            f"  fixed_frame pos={fixed_pose[:3]} quat={fixed_pose[3:]}\n"
            f"  held_to_tcp pos={held_to_tcp_pose[:3]} quat={held_to_tcp_pose[3:]}\n"
            f"  next_action pos={next_action[:3]} quat={next_action[3:7]} gripper={next_action[7]}\n"
            f"  ctrl_target pos={ctrl_target_pose[:3]} quat={ctrl_target_pose[3:]}\n"
            f"  ee_err      dpos={pos_err} rot_err_deg={rot_err_deg:.2f}\n"
            f"  joint_pos   {joint_pos}\n"
            f"  joint_tgt   {ctrl_target_joint_pos}"
        )

    @staticmethod
    def _quat_angle_error_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        dot = torch.sum(q1 * q2, dim=-1).abs().clamp(-1.0, 1.0)
        return 2.0 * torch.acos(dot)

    def _init_experiment_logging(self):
        """Create logging artifacts for the current run."""
        os.makedirs(self.output_dir, exist_ok=True)

        self.config_output_dir = os.path.join(self.output_dir, "config")
        os.makedirs(self.config_output_dir, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_files = ("forge_env_cfg.py", "forge_tasks_cfg.py")
        for config_name in config_files:
            src_path = os.path.join(current_dir, config_name)
            dst_path = os.path.join(self.config_output_dir, config_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        self.success_log_path = os.path.join(self.output_dir, "success_rate.csv")
        if not os.path.exists(self.success_log_path):
            with open(self.success_log_path, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    [
                        "common_step",
                        "env_id",
                        "episode_length",
                        "success",
                        "success_times",
                        "total_times",
                        "success_rate_percent",
                    ]
                )

    def _log_success_rate(self, env_id: int, success: bool):
        """Append episode-level success statistics to disk."""
        success_rate = (self.success_times / self.total_times) * 100 if self.total_times > 0 else 0.0
        with open(self.success_log_path, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    int(self.common_step_counter),
                    int(env_id),
                    int(self.episode_length_buf[env_id].item()),
                    int(bool(success)),
                    int(self.success_times),
                    int(self.total_times),
                    f"{success_rate:.6f}",
                ]
            )
        return success_rate
        
    def record_data(self, env_idx=None):
        """
        Record simulation data for one or all environments.
        
        Args:
            env_idx (int, optional): Index of the environment to record data for.
                                    If None, record data for all environments.
        """
        if not self.collect_data:
            return
            
        if env_idx is not None:
            buf = self.data_buffers[env_idx]
            
            # Record camera data
            if hasattr(self, "tiled_camera") and self.tiled_camera is not None:
                buf["camera"]["front"].append(
                    self.tiled_camera.data.output["rgb"][env_idx].to("cpu").clone()
                )
            if hasattr(self, "wrist_tiled_camera") and self.wrist_tiled_camera is not None:
                buf["camera"]["wrist"].append(
                    self.wrist_tiled_camera.data.output["rgb"][env_idx].to("cpu").clone()
                )
            
            # Record joint states
            buf["joints"].append(self.joint_pos[env_idx].to("cpu").clone())
            
            # Record gripper state
            ctrl_target_gripper_dof_pos = 0.0
            buf["gripper"].append(ctrl_target_gripper_dof_pos)
            
            # Record end-effector pose
            ee_pose = torch.cat([
                self.fingertip_midpoint_pos[env_idx], 
                self.fingertip_midpoint_quat[env_idx]
            ], dim=0).to("cpu")
            buf["ee_pose"].append(ee_pose)
            
            # Record force/torque sensor data (local frame)
            buf["force"].append(self.force_sensor_smooth[env_idx].to("cpu").clone())
            
            # Record force/torque sensor data (world frame)
            buf["force_world"].append(self.force_sensor_world_smooth[env_idx].to("cpu").clone())
            
        else:
            # Record for all environments
            for i in range(self.num_envs):
                self.record_data(i)

    def reset_data_buffer(self, env_idx=None):
        """
        Clear the data buffer for one or all environments.
        
        Args:
            env_idx (int, optional): Index of the environment to reset.
                                    If None, reset all environments.
        """
        if not self.collect_data:
            return
            
        if env_idx is not None:
            self.data_buffers[env_idx] = {
                "camera": {
                    "front": [],
                    "wrist": [],
                },
                "joints": [],
                "gripper": [],
                "ee_pose": [],
                "force": [],
                "force_world": [],
                "actions": []
            }
        else:
            for i in range(self.num_envs):
                self.reset_data_buffer(i)

    def save_data_to_disk(self, env_idx=None):
        """
        Save the buffered data to disk for one or more environments.
        
        Args:
            env_idx (int, list, np.ndarray, optional): Index or indices of environments to save.
                                                    If None, save all environments.
        """
        if not self.collect_data:
            return
            
        def preprocess_frame(frame):
            """Convert frame to proper format for saving."""
            # Convert to numpy
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            elif not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"Unsupported data type: {type(frame)}. "
                    f"Expected torch.Tensor or numpy.ndarray."
                )

            # Normalize [0,1] -> [0,255] if needed
            if frame.min() >= 0.0 and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

            # Handle channel order: (C, H, W) -> (H, W, C)
            if frame.ndim == 3 and frame.shape[0] in [1, 3]:
                frame = np.transpose(frame, (1, 2, 0))

            return frame

        base_output_dir = self.output_dir
        os.makedirs(base_output_dir, exist_ok=True)

        # Normalize env indices
        if env_idx is None:
            env_indices = list(range(self.num_envs))
        elif isinstance(env_idx, (list, tuple, np.ndarray)):
            env_indices = [int(i) for i in env_idx]
        else:
            env_indices = [int(env_idx)]

        for idx in env_indices:
            # Find next available episode directory
            episode_idx = 0
            while True:
                episode_dir = os.path.join(base_output_dir, f'episode_{episode_idx}')
                if not os.path.exists(episode_dir):
                    print(f"Saving new data folder for env {idx}: {episode_dir}")
                    break
                episode_idx += 1

            buf = self.data_buffers[idx]

            # Save camera data as video
            for key, camera_list in buf["camera"].items():
                if len(camera_list) <= 1:
                    continue

                save_dir = os.path.join(episode_dir, key)
                os.makedirs(save_dir, exist_ok=True)

                # Get frame properties
                first_frame = preprocess_frame(camera_list[0])
                last_frame = preprocess_frame(camera_list[-1])
                height, width, channels = first_frame.shape

                # Create video writer
                fps = int(1 / (self.physics_dt * self.cfg.decimation))
                video_path = os.path.join(save_dir, f'{key}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                # Write all frames (skip first)
                for i in range(1, len(camera_list)):
                    frame = preprocess_frame(camera_list[i])
                    
                    # Convert to BGR for OpenCV
                    if frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    video_writer.write(frame)

                video_writer.release()
                print(f"Saved {key} video to {video_path}")

                # Save last frame
                Image.fromarray(last_frame).save(os.path.join(save_dir, 'last_frame.png'))

            # Save numerical data to CSV
            self._save_array_to_csv(buf["joints"], episode_dir, 'joint_states.csv', 'joint')
            self._save_array_to_csv(buf["force"], episode_dir, 'force_local.csv', 'force')
            self._save_array_to_csv(buf["force_world"], episode_dir, 'force_world.csv', 'force')
            self._save_array_to_csv(buf["gripper"], episode_dir, 'gripper.csv', 'gripper')
            self._save_array_to_csv(buf["actions"], episode_dir, 'actions.csv', 'action')
            self._save_array_to_csv(buf["ee_pose"], episode_dir, 'ee_pose.csv', 'ee_pose')

    def _save_array_to_csv(self, data_list, episode_dir, filename, column_prefix):
        """
        Helper function to save a time-series data list to CSV.
        
        Args:
            data_list (list): List of tensors to save
            episode_dir (str): Directory to save CSV
            filename (str): CSV file name
            column_prefix (str): Prefix for CSV column names
        """
        if not data_list or len(data_list) <= 1:
            print(f"No data or insufficient data ({column_prefix}), skipping save.")
            return

        file_path = os.path.join(episode_dir, filename)

        try:
            # Handle scalar gripper values
            if isinstance(data_list[0], (int, float)):
                header = [column_prefix]
                with open(file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(header)
                    for i in range(1, len(data_list)):
                        csv_writer.writerow([data_list[i]])
            else:
                # Handle tensor/array data
                num_columns = data_list[0].shape[0]
                header = [f'{column_prefix}_{j}' for j in range(num_columns)]

                with open(file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(header)

                    for i in range(1, len(data_list)):
                        row_data = data_list[i]
                        if hasattr(row_data, 'cpu'):
                            row_data = row_data.detach().cpu().numpy()
                        csv_writer.writerow(row_data)

            print(f"Data ({column_prefix}) saved to {file_path}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    def _get_observations(self):
        """Override to record data after computing observations."""
        obs = super()._get_observations()
        
        # ========== 新增: Policy推理 ==========
        if self.policy:
            next_action = self.select_action(self.policy)
            self.next_action = torch.stack(next_action).to(obs["policy"].device)

        # ======================================
        
        # Record data if collection is enabled
        if self.collect_data:
            self.record_data()
        
        return obs

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.immediate_stop :
            curr_successes = self._get_curr_successes(
                success_threshold=self.cfg_task.success_threshold, check_rot=self.cfg_task.name == "nut_thread"
            )
            terminated = time_out | curr_successes
        else:
            terminated = time_out

        return terminated, time_out
    
    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # Use pi0 (zero actions) for inference instead of RL actions
        if hasattr(self, 'policy') and self.policy is not None:
            action = torch.zeros_like(action)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        start = time.time()
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        end = time.time()
        # print(f"Cost time:{end-start}")
        # print(self.episode_length_buf)

        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # 保存数据
            for env_ids in reset_env_ids.to("cpu").numpy().tolist():
                success = self._get_curr_successes(
                    success_threshold=self.cfg_task.success_threshold, check_rot=False
                )
                self.success_times = self.success_times + 1 if success[env_ids] else self.success_times
                self.total_times += 1
                success_rate = self._log_success_rate(env_ids, bool(success[env_ids]))
                if self.collect_data and self.reset_terminated[env_ids]:
                    if success[env_ids]:
                        print("Task success!")
                        self.cur_num_traj += 1
                        self.save_data_to_disk(env_ids)
                        self.reset_data_buffer(env_ids)
                    elif self.save_failed_trajectory:
                        self.cur_num_traj += 1
                        print("Task Failed!")
                        self.save_data_to_disk(env_ids)
                        self.reset_data_buffer(env_ids)
                    else:
                        # Failed trajectory is intentionally not saved, but buffer
                        # must still be cleared to avoid mixing with next episode.
                        self.reset_data_buffer(env_ids)

            self._reset_idx(reset_env_ids)
            avg_reward = self.reward_buf.mean()
            print(f"Current Success rate: {self.success_times} / {self.total_times} = {success_rate:.2f}%")
            print(f"Average Reward: {avg_reward.item():.6f}")

            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.cur_num_traj >= self.num_trajectories:
            exit(0)
            
        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 保存数据
        # post-step:
         # 保存 action 的数据
        if self.collect_data:
            env_num = self.fingertip_midpoint_pos.shape[0]
            ctrl_target_gripper_dof_pos = 0.0
            gripper = torch.tensor(ctrl_target_gripper_dof_pos, device="cuda")
            gripper = gripper.expand(env_num, 1)
            for i in range(self.num_envs):
                self.data_buffers[i]["actions"].append(self.next_action[i])

        if self.debug_motion_planner:
            if self.debug_motion_planner_visualize:
                self._update_motion_planner_debug_vis()
            self._print_motion_planner_debug()

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _apply_action_torque_control(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.

        if self.cfg.disable_xy_rot:
            rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        # ========== 新增: 判断使用RL action还是Policy action ==========
        if hasattr(self, 'policy') and self.policy is not None:
            # 使用外部policy的action (绝对位置和四元数)
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=self.next_action[:, :3],
                ctrl_target_fingertip_midpoint_quat=self.next_action[:, 3:7],
                # ctrl_target_fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                # ctrl_target_fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                # ctrl_target_fingertip_midpoint_quat= torch.tensor(
                #     [[0.0, 1.0, 0.0, 0.0]], device=self.device
                # ),
                ctrl_target_gripper_dof_pos=self.next_action[:,-1:],
            )
        elif self.motion_planner is not None and self.planner_to_rl_handoff:
            gripper = torch.zeros((self.num_envs, 1), dtype=self.fingertip_midpoint_pos.dtype, device=self.device)
            rl_next_action = torch.cat(
                [
                    ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat,
                    gripper,
                ],
                dim=1,
            )
            next_action = rl_next_action.clone()
            planner_target_pos = ctrl_target_fingertip_midpoint_pos.clone()
            planner_target_quat = ctrl_target_fingertip_midpoint_quat.clone()
            planner_target_gripper = gripper.clone()

            for env_idx in range(self.num_envs):
                if not self.motion_planner_active[env_idx]:
                    continue

                target_pose = self.motion_planner_target_ee[env_idx]
                pos_err = torch.linalg.norm(self.fingertip_midpoint_pos[env_idx] - target_pose[:3])
                rot_err = self._quat_angle_error_wxyz(
                    self.fingertip_midpoint_quat[env_idx].unsqueeze(0),
                    target_pose[3:].unsqueeze(0),
                ).squeeze(0)

                if self.planner_to_rl_handoff and pos_err <= self.planner_to_rl_pos_tol and rot_err <= self.planner_to_rl_rot_tol:
                    self.motion_planner_active[env_idx] = False
                    self.motion_planner_debug["valid"][env_idx] = False
                    print(
                        f"[MotionPlanner->RL][env={env_idx}][step={int(self.common_step_counter)}] "
                        f"handoff pos_err={float(pos_err):.4f} rot_err_deg={float(torch.rad2deg(rot_err)):.2f}"
                    )
                    continue

                planner_action = self.motion_planner[env_idx].get_action()
                if planner_action is None:
                    planner_action = self.motion_planner_last_action[env_idx]
                else:
                    self.motion_planner_last_action[env_idx] = planner_action

                if planner_action is None:
                    continue

                planner_action = torch.tensor(
                    planner_action, dtype=self.fingertip_midpoint_pos.dtype, device=self.device
                )
                next_action[env_idx] = planner_action
                planner_target_pos[env_idx] = planner_action[:3]
                planner_target_quat[env_idx] = planner_action[3:7]
                planner_target_gripper[env_idx] = planner_action[-1]

            self.next_action = next_action
            if self.debug_motion_planner:
                self.motion_planner_debug["next_action"] = self.next_action.detach().to("cpu", dtype=torch.float32)
                self.motion_planner_debug["ctrl_target_pose"] = torch.cat(
                    [planner_target_pos, planner_target_quat], dim=1
                ).detach().to("cpu", dtype=torch.float32)
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=planner_target_pos,
                ctrl_target_fingertip_midpoint_quat=planner_target_quat,
                # ctrl_target_gripper_dof_pos=0.0,
                ctrl_target_gripper_dof_pos=planner_target_gripper,
            )
            if self.debug_motion_planner:
                self.motion_planner_debug["joint_pos"] = self.joint_pos[:, :7].detach().to("cpu", dtype=torch.float32)
                self.motion_planner_debug["ctrl_target_joint_pos"] = (
                    self.ctrl_target_joint_pos[:, :7].detach().to("cpu", dtype=torch.float32)
                )
        else:
            ctrl_target_gripper_dof_pos = 0.0
            gripper = torch.tensor(ctrl_target_gripper_dof_pos, device="cuda")
            gripper = gripper.expand(self.fingertip_midpoint_pos.shape[0], 1)

            self.next_action = torch.cat(
                [
                    ctrl_target_fingertip_midpoint_pos,
                    ctrl_target_fingertip_midpoint_quat,
                    gripper,
                ],
                dim=1,
            )

            # 使用RL计算的action (相对增量)
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                ctrl_target_gripper_dof_pos=0.0,
            )
        # ===========================================================
        
    def _apply_action_position_control(self):
        """Position-control variant of _apply_action using DLS IK to produce joint targets."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)
        joint_dtype = self.joint_pos.dtype

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions

        if self.cfg.disable_xy_rot:
            rot_actions[:, 0:2] = 0.0

        rot_actions[:, 2] = (
            torch.tensor(np.deg2rad(-180.0), device=self.device, dtype=joint_dtype)
            + torch.tensor(np.deg2rad(270.0), device=self.device, dtype=joint_dtype) * (rot_actions[:, 2] + 1.0) / 2.0
        )
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )
        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)
        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)
        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        if hasattr(self, "policy") and self.policy is not None:
            target_pos = self.next_action[:, :3].to(dtype=joint_dtype)
            target_quat = self.next_action[:, 3:7].to(dtype=joint_dtype)
            target_gripper = self.next_action[:, -1:].to(dtype=joint_dtype)
        elif self.motion_planner is not None and self.planner_to_rl_handoff:
            gripper = torch.zeros((self.num_envs, 1), dtype=self.fingertip_midpoint_pos.dtype, device=self.device)
            rl_next_action = torch.cat(
                [ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, gripper], dim=1
            )
            next_action = rl_next_action.clone()
            target_pos = ctrl_target_fingertip_midpoint_pos.clone()
            target_quat = ctrl_target_fingertip_midpoint_quat.clone()
            target_gripper = gripper.clone()

            for env_idx in range(self.num_envs):
                if not self.motion_planner_active[env_idx]:
                    continue

                target_pose = self.motion_planner_target_ee[env_idx]
                pos_err = torch.linalg.norm(self.fingertip_midpoint_pos[env_idx] - target_pose[:3])
                rot_err = self._quat_angle_error_wxyz(
                    self.fingertip_midpoint_quat[env_idx].unsqueeze(0),
                    target_pose[3:].unsqueeze(0),
                ).squeeze(0)

                if self.planner_to_rl_handoff and pos_err <= self.planner_to_rl_pos_tol and rot_err <= self.planner_to_rl_rot_tol:
                    self.motion_planner_active[env_idx] = False
                    self.motion_planner_debug["valid"][env_idx] = False
                    print(
                        f"[MotionPlanner->RL][env={env_idx}][step={int(self.common_step_counter)}] "
                        f"handoff pos_err={float(pos_err):.4f} rot_err_deg={float(torch.rad2deg(rot_err)):.2f}"
                    )
                    continue

                planner_action = self.motion_planner[env_idx].get_action()
                if planner_action is None:
                    planner_action = self.motion_planner_last_action[env_idx]
                else:
                    self.motion_planner_last_action[env_idx] = planner_action

                if planner_action is None:
                    continue

                planner_action = torch.tensor(planner_action, dtype=joint_dtype, device=self.device)
                next_action[env_idx] = planner_action
                target_pos[env_idx] = planner_action[:3]
                target_quat[env_idx] = planner_action[3:7]
                target_gripper[env_idx] = planner_action[-1]

            self.next_action = next_action
            if self.debug_motion_planner:
                self.motion_planner_debug["next_action"] = self.next_action.detach().to("cpu", dtype=torch.float32)
                self.motion_planner_debug["ctrl_target_pose"] = torch.cat(
                    [target_pos, target_quat], dim=1
                ).detach().to("cpu", dtype=torch.float32)
        else:
            ctrl_target_gripper_dof_pos = 0.0
            gripper = torch.full(
                (self.fingertip_midpoint_pos.shape[0], 1),
                ctrl_target_gripper_dof_pos,
                dtype=self.fingertip_midpoint_pos.dtype,
                device=self.device,
            )
            self.next_action = torch.cat(
                [ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, gripper], dim=1
            )
            target_pos = ctrl_target_fingertip_midpoint_pos.to(dtype=joint_dtype)
            target_quat = ctrl_target_fingertip_midpoint_quat.to(dtype=joint_dtype)
            target_gripper = gripper.to(dtype=joint_dtype)

        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos.to(dtype=joint_dtype),
            fingertip_midpoint_quat=self.fingertip_midpoint_quat.to(dtype=joint_dtype),
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1).to(dtype=joint_dtype)
        delta_dof_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_hand_pose,
            ik_method="dls",
            jacobian=self.fingertip_midpoint_jacobian.to(dtype=joint_dtype),
            device=self.device,
        )

        joint_target = self.joint_pos.clone()
        joint_target[:, 0:7] = joint_target[:, 0:7] + delta_dof_pos[:, 0:7]
        joint_target[:, 7:9] = target_gripper

        self.ctrl_target_joint_pos[:, :] = joint_target
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

        if self.debug_motion_planner:
            self.motion_planner_debug["joint_pos"] = self.joint_pos[:, :7].detach().to("cpu", dtype=torch.float32)
            self.motion_planner_debug["ctrl_target_joint_pos"] = (
                self.ctrl_target_joint_pos[:, :7].detach().to("cpu", dtype=torch.float32)
            )

    def _apply_action(self):
        """Dispatch to the configured arm control mode."""
        desired_mode = "position" if self.motion_planner_active[0] else "torque" # motion plan 的时候用位置控制，切换为 rl policy 的时候用扭矩控制
        if self.motion_planner is not None and self.planner_to_rl_handoff and self.control_mode != desired_mode:
            self.control_mode = desired_mode
            self.cfg.control_mode = desired_mode

            arm_joint_ids = self._robot.find_joints("panda_joint[1-7]")[0]
            arm_stiffness = 800.0 if desired_mode == "position" else 0.0
            arm_damping = 40.0 if desired_mode == "position" else 0.0

            self._robot.write_joint_stiffness_to_sim(arm_stiffness, joint_ids=arm_joint_ids)
            self._robot.write_joint_damping_to_sim(arm_damping, joint_ids=arm_joint_ids)

            for actuator_name in ("panda_arm1", "panda_arm2"):
                actuator = self._robot.actuators.get(actuator_name)
                if actuator is not None:
                    actuator.stiffness[:] = arm_stiffness
                    actuator.damping[:] = arm_damping

        if self.control_mode == "torque":
            self._apply_action_torque_control()
        else:
            self._apply_action_position_control()
                
        
            
    def select_action(self, policy):
        """
        使用policy为每个环境生成action。
        
        Args:
            policy: Policy模型实例
            
        Returns:
            action_list: 每个环境的action列表
        """
        action_list = []
        for env_idx in range(self.num_envs):
            # 准备当前末端执行器位姿
            cur_ee_pose = torch.cat([
                self.fingertip_midpoint_pos[env_idx], 
                self.fingertip_midpoint_quat[env_idx]
            ], dim=0)
            
            # 准备相机图像
            head_img_tensor = self.tiled_camera.data.output["rgb"][env_idx].to("cpu").clone().unsqueeze(0)
            wrist_img_tensor = self.wrist_tiled_camera.data.output["rgb"][env_idx].to("cpu").clone().unsqueeze(0)
            
            # 准备状态信息 (末端位姿 + 夹爪状态)
            _state = torch.cat([
                cur_ee_pose, 
                self.joint_pos[env_idx][-1].unsqueeze(0)
            ], dim=-1).to("cpu").clone()
            _state = _state.view(1, -1)
            
            # 准备任务提示
            prompt_data = self.cfg.task_prompt
            
            # 准备力/力矩传感器数据
            effort = self.force_sensor_smooth[env_idx].unsqueeze(0).to("cpu")
            # effort = torch.nn.functional.pad(effort, (0, 2, 0, 0), mode='constant', value=0)
            # print("effort type", type(effort))
            # print("effort ",effort)
            
            # 打包输入字典 (键名必须与模型内部映射一致)
            batch_input = {
                "observation.images.front": head_img_tensor,
                "observation.images.left_wrist": wrist_img_tensor,
                # Backward/remote-server compatible aliases:
                "observation.images.head_camera": head_img_tensor,
                "observation.images.wrist_left_camera": wrist_img_tensor,
                "observation.state": _state,
                "observation.effort": effort,
                "task": prompt_data,
                # For episode streaming: True only on the first step after reset (per env).
                "episode_start": torch.tensor([bool(self._episode_start[env_idx].item())], dtype=torch.bool),
            }
            
            # 调用policy推理
            next_action = policy.select_action(batch_input)
            action_list.append(next_action)
            # consume episode_start after first use
            if self._episode_start[env_idx]:
                self._episode_start[env_idx] = False
        
        return action_list
    
    def _build_motion_planner(self,env_ids):
        # 构建 MotionPlanner
        if self.motion_planner is not None:
            for env_idx in env_ids.tolist():
                cur_ee_pose = torch.cat([
                    self.fingertip_midpoint_pos[env_idx], 
                    self.fingertip_midpoint_quat[env_idx]
                ], dim=0).to("cpu")
                
                if self.cfg.task_name == "peg_insert":
                    held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose() # 实际上获取的位置是不对的，因为这个函数里面只考虑了peg没有旋转的情形，但我们的peg有30度的旋转，但是不影响

                    target_held_pos = self.fixed_pos_obs_frame[env_idx].unsqueeze(0).clone()
                    target_held_pos[:, 2] += 0.03
                    target_held_quat = self.fixed_quat[env_idx].unsqueeze(0)
                    
                    rot_180_euler = torch.tensor([[np.pi, 0.0, 0.0]], device=self.device)
                    quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
                        roll=rot_180_euler[:, 0],
                        pitch=rot_180_euler[:, 1],
                        yaw=rot_180_euler[:, 2],
                    )

                    held_to_tcp_quat = quat_bolt_to_ee
                    held_to_tcp_pos = held_asset_relative_pos[env_idx].unsqueeze(0)
                    
                    target_ee_quat, target_ee_pos = torch_utils.tf_combine(
                        target_held_quat,
                        target_held_pos,
                        held_to_tcp_quat,
                        held_to_tcp_pos,
                    )
                    # Debug experiment: keep the current EE orientation and only test
                    # whether the controller can track the planned position target.
                    target_ee_quat = cur_ee_pose[3:].unsqueeze(0).to(self.device)
                    target_ee_pose = torch.cat([target_ee_pos.squeeze(0), target_ee_quat.squeeze(0)], dim=0).to("cpu")
                    target_held_pose = torch.cat([target_held_pos.squeeze(0), target_held_quat.squeeze(0)], dim=0).to("cpu")
                    held_to_tcp_pose = torch.cat([held_to_tcp_pos.squeeze(0), held_to_tcp_quat.squeeze(0)], dim=0).to("cpu")
                    fixed_pose = torch.cat([
                        self.fixed_pos_obs_frame[env_idx].to("cpu"),
                        self.fixed_quat[env_idx].to("cpu"),
                    ], dim=0)
                    
                    pose_planner = PoseTrajectoryPlanner()
                    move_action = pose_planner.plan( 
                        cur_pose = cur_ee_pose,
                        target_pose = target_ee_pose,
                        num_steps = 100, # 50 步后到达目标点
                        gripper = 0.0,
                    )
                    self.motion_planner[env_idx]._append_segment(move_action)
                    self.motion_planner_active[env_idx] = True
                    self.motion_planner_target_ee[env_idx] = torch.cat(
                        [target_ee_pos.squeeze(0), target_ee_quat.squeeze(0)], dim=0
                    )
                    if self.debug_motion_planner:
                        self.motion_planner_debug["valid"][env_idx] = True
                        self.motion_planner_debug["cur_ee_pose"][env_idx] = cur_ee_pose
                        self.motion_planner_debug["target_ee_pose"][env_idx] = target_ee_pose
                        self.motion_planner_debug["target_held_pose"][env_idx] = target_held_pose
                        self.motion_planner_debug["fixed_pose"][env_idx] = fixed_pose
                        self.motion_planner_debug["held_to_tcp_pose"][env_idx] = held_to_tcp_pose
                elif self.cfg.task_name == "gear_assembly":
                    held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
                        self.held_pos, self.held_quat, "gear_mesh", self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
                    )
                     
                    target_ee_pos = held_base_pos[env_idx][0:3].unsqueeze(0).clone()
                    target_ee_pos[:, 2] += 0.017
                    # target_ee_quat = self.fixed_quat[env_idx].unsqueeze(0)
                    target_ee_quat = self.fingertip_midpoint_quat
                    
                    # Debug experiment: keep the current EE orientation and only test
                    # whether the controller can track the planned position target.
                    target_ee_pose = torch.cat([target_ee_pos.squeeze(0), target_ee_quat.squeeze(0)], dim=0).to("cpu")
                    waypoint_1 = target_ee_pose.clone()
                    waypoint_1[2] += 0.06
                    pose_planner = PoseTrajectoryPlanner()
                    move_action = pose_planner.plan(  # 移到gear的上方
                        cur_pose = cur_ee_pose,
                        target_pose = waypoint_1,
                        num_steps = 150, # 50 步后到达目标点
                        gripper = 0.03,
                    )
                    self.motion_planner[env_idx]._append_segment(move_action)
                    move_action = pose_planner.plan(  # 下降
                        cur_pose = waypoint_1,
                        target_pose = target_ee_pose,
                        num_steps = 100, # 50 步后到达目标点
                        gripper = 0.03,
                    )
                    
                    self.motion_planner[env_idx]._append_segment(move_action)
                    
                    
                    self.motion_planner[env_idx]._append_segment(
                        pose_planner.plan(  # 稳定
                        cur_pose = target_ee_pose,
                        target_pose = target_ee_pose,
                        num_steps = 30, # 50 步后到达目标点
                        gripper = 0.03,
                        )
                    )
                    
                    move_action = pose_planner.plan(  # 闭合gripper
                        cur_pose = target_ee_pose,
                        target_pose = target_ee_pose,
                        num_steps = 200, # 50 步后到达目标点
                        gripper = 0.0,
                    )
                    self.motion_planner[env_idx]._append_segment(move_action)
                    
                    waypoint_2 = target_ee_pose.clone() 
                    waypoint_2[2] = waypoint_2[2] + 0.06 
                    self.motion_planner[env_idx]._append_segment(
                        pose_planner.plan(  # 下降
                            cur_pose = target_ee_pose,
                            target_pose = waypoint_2,
                            num_steps = 150, # 50 步后到达目标点
                            gripper = 0.0,
                        )
                    )
                    
                    target_gear_pos, target_gear_quat =  factory_utils.get_target_held_base_pose(
                        self.fixed_pos,
                        self.fixed_quat,
                        "gear_mesh",
                        self.cfg_task.fixed_asset_cfg,
                        self.num_envs,
                        self.device,
                    )
                    target_gear_pose =  torch.cat([target_gear_pos.squeeze(0),  self.fingertip_midpoint_quat.squeeze(0)], dim=0).to("cpu")
                    target_gear_pose[2] = waypoint_2[2]
                    self.motion_planner[env_idx]._append_segment(
                        pose_planner.plan(  # 下降
                            cur_pose = waypoint_2,
                            target_pose = target_gear_pose,
                            num_steps = 150, # 50 步后到达目标点
                            gripper = 0.0,
                        )
                    )
                     
                    waypoint_4 = target_gear_pose.clone()
                    waypoint_4[2] = target_gear_pose[2] - 0.04
                    self.motion_planner[env_idx]._append_segment(
                        pose_planner.plan(  # 下降
                            cur_pose = target_gear_pose,
                            target_pose = waypoint_4,
                            num_steps = 150, # 50 步后到达目标点
                            gripper = 0.0,
                        )
                    )
                    self.motion_planner_active[env_idx] = True
                    self.motion_planner_target_ee[env_idx] = waypoint_4
                    
                    

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)
        # print("fingertip_midpoint_pos:",self.fingertip_midpoint_pos)
        # print("fingertip_midpoint_quat:",self.fingertip_midpoint_quat)
        # print("fixed_pos:",self.fixed_pos)
        # print("fixed_quat:",self.fixed_quat)
        
        # ========== 新增: 重置policy状态 ==========
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
            
        if self.motion_planner is not None and self.planner_to_rl_handoff:
            self.motion_planner = [MotionPolicy() for _ in range(self.num_envs)]
            self.motion_planner_last_action = [None for _ in range(self.num_envs)]
            self.motion_planner_active[env_ids] = True
            self.motion_planner_debug["valid"][env_ids.to(device="cpu")] = False
            self._build_motion_planner(env_ids)
        # =========================================
        # mark episode start for reset envs (CPU-side flag consumed in select_action)
        if env_ids is not None and len(env_ids) > 0:
            self._episode_start[env_ids.to(device="cpu")] = True
        
        
        
    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        # [MODIFIED] 使用局部生成器 self.rng
        rand_sample = torch.rand((len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        
        # [MODIFIED] 使用局部生成器 self.rng
        rand_sample = torch.rand((len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        # [MODIFIED] 使用局部生成器 self.rng
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            # [MODIFIED] 使用局部生成器 self.rng
            rand_sample = torch.rand((n_bad, 3), generator=self.rng, dtype=torch.float32, device=self.device)
            
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            # [MODIFIED] 使用局部生成器 self.rng
            rand_sample = torch.rand((n_bad, 3), generator=self.rng, dtype=torch.float32, device=self.device)
            
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            ) 

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        # [MODIFIED] 使用局部生成器 self.rng
        # 注意：原文这里是对 self.num_envs 进行随机，建议保持原逻辑以防维度不匹配
        # 如果只想对 env_ids 随机，需要修改这部分逻辑，但为了稳妥起见，我们只替换生成器
        rand_sample = torch.rand((self.num_envs, 3), generator=self.rng, dtype=torch.float32, device=self.device)
        
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        
        # Apply configurable peg-in-gripper rotation noise for peg insertion.
        if self.cfg_task.name == "peg_insert":
            rot_noise_deg = float(getattr(self.cfg, "peg_insert_rot_noise_deg", 0.0))
            rot_noise_rad = np.deg2rad(rot_noise_deg)
            
            # [MODIFIED] 使用局部生成器 self.rng
            rand_rot_sample = torch.rand((self.num_envs, 3), generator=self.rng, dtype=torch.float32, device=self.device)
            
            held_asset_rpy_noise = 2 * (rand_rot_sample - 0.5) * rot_noise_rad 

            # 转换为四元数
            held_asset_rot_noise_quat = torch_utils.quat_from_euler_xyz(
                held_asset_rpy_noise[:, 0], 
                held_asset_rpy_noise[:, 1], 
                held_asset_rpy_noise[:, 2]
            )

            # 3. 第一步结合：将噪声施加到夹爪坐标系上
            # 这一步得到的是一个“带有误差的夹爪中心位姿”
            noisy_gripper_quat, noisy_gripper_pos = torch_utils.tf_combine(
                q1=fingertip_flipped_quat, 
                t1=fingertip_flipped_pos, 
                q2=held_asset_rot_noise_quat,   # 在夹爪中心施加旋转
                t2=held_asset_pos_noise         # 在夹爪中心施加位移
            )

            # 4. 第二步结合：加上物体相对于夹爪的固定偏移
            # 因为前一步旋转了坐标系，这一步的偏移向量会跟着旋转，从而实现“绕点旋转”
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=noisy_gripper_quat, 
                t1=noisy_gripper_pos, 
                q2=asset_in_hand_quat, 
                t2=asset_in_hand_pos
            )
            print("随机位置和旋转")
        else:
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=translated_held_asset_quat,
                t1=translated_held_asset_pos,
                q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                t2=held_asset_pos_noise,
            )
            
        if self.cfg.task_name == "gear_assembly": # 对于这个任务，需要将gear在桌面上随机放置
            held_base_pos_local = factory_utils.get_held_base_pos_local( "gear_mesh", self.cfg_task.fixed_asset_cfg, self.num_envs, self.device) # gear 的中心点相对于其基坐标系的偏移
            held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            held_inverse_quat_local , held_inverse_pos_local  = torch_utils.tf_inverse(
                held_base_quat_local, held_base_pos_local
            )
            
            dist_rand = 2 * torch.rand((len(env_ids)), generator=self.rng, dtype=torch.float32, device=self.device) -1 # 随机距离
            theta_rand = 2 * torch.rand((len(env_ids)), generator=self.rng, dtype=torch.float32, device=self.device) -1 # 随机角度
            
            dist = self.cfg.gear_init_min_distance + dist_rand * (self.cfg.gear_init_max_distance - self.cfg.gear_init_min_distance)
            theta = theta_rand * np.pi
            
            target_gear_pos = fixed_state[:,0:3].clone()
            target_gear_pos[:,0] += dist * torch.cos(theta)
            target_gear_pos[:,1] += dist * torch.sin(theta)
            target_gear_quat = fixed_state[:,3:7]
            
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                target_gear_quat,
                target_gear_pos,
                held_inverse_quat_local,
                held_inverse_pos_local,
            )
            
            # 设置重力
            # physics_sim_view.set_gravity(carb.Float3(9.8, 0.0, 0.0))
            
            
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        if self.cfg.task_name != "gear_assembly": # gear assembly 需要先夹起 gear 然后装配，因此不需要闭合夹爪
            grasp_time = 0.0
            while grasp_time < 0.25:
                self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
                self.close_gripper_in_place()
                self.step_sim_no_action()
                grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
