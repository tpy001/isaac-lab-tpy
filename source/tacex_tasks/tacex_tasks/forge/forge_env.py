# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import os
from PIL import Image
import csv
import cv2
import time

from .isaac_forge_env import ForgeEnv as IsaacForgeEnv
from .forge_env_cfg import ForgeEnvCfg
from isaaclab.sensors import TiledCamera
import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_utils



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
        
        # ========== 新增: Policy初始化 ==========
        if cfg.policy_cfg:
            from .policy.modeling_pi0remote import PI0RemotePolicy, PI0RemotePolicyTAVLA
            # 根据配置选择policy类型
            if hasattr(cfg.policy_cfg, 'num_history_steps'):
                self.policy = PI0RemotePolicyTAVLA(cfg.policy_cfg)
                print("Using Pi0 Policy")
            else:
                self.policy = PI0RemotePolicy(cfg.policy_cfg)
                print("Using TA-VLA Policy")
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
        
        self.success_times = 0
        self.total_times = 0

    def _setup_scene(self):
        super()._setup_scene()
        # sensors
        if hasattr(self.cfg, "wrist_camera") and self.cfg.wrist_camera is not None:
            self.wrist_tiled_camera = TiledCamera(self.cfg.wrist_camera)
            self.scene.sensors["wrist_tiled_camera"] = self.wrist_tiled_camera

        if hasattr(self.cfg, "tiled_camera") and self.cfg.tiled_camera is not None:
            self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self.tiled_camera
        
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
                success_rate = (self.success_times / self.total_times) * 100 if self.total_times > 0 else 0.0
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

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _apply_action(self):
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
                ctrl_target_gripper_dof_pos=0.0,
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
                "observation.state": _state,
                "observation.effort": effort,
                "task": prompt_data,
            }
            
            # 调用policy推理
            next_action = policy.select_action(batch_input)
            action_list.append(next_action)
        
        return action_list
    
    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)
        
        # ========== 新增: 重置policy状态 ==========
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
        # =========================================
