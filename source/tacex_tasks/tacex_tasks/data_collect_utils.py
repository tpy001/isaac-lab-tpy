import os
from PIL import Image
import csv
import torch
import numpy as np
import cv2


def reset_data_buffer(self, env_idx=None):
    """
    Clear the data buffer for one or all environments.

    Args:
        env_idx (int, optional): Index of the environment to reset.
                                If None, reset all environments.
    """
    if env_idx is not None:
        # Reset buffer for a single environment
        self.data_buffers[env_idx] = {
            "camera": {
                "front": [],
                "wrist": [],
            },
            "joints": [],
            "gripper": [],
            "force": [],
            "ee_pose": [],
            "actions": []
        }
    else:
        # Reset buffers for all environments
        for i in range(self.num_envs):
            reset_data_buffer(self,i)
            
def get_force(self, link_name, idx):
    # 获取link索引
    link_idx = self._robot.data.body_names.index(link_name)
    # 获取局部六维力
    force_local = self._robot.data.body_incoming_joint_wrench_b[idx, link_idx]
    return force_local
    
def record_data(self, env_idx=None, force_links=["panda_hand"]):
        """
        Record simulation data for one or all environments.

        Args:
            env_idx (int, optional): Index of the environment to record data for.
                                    If None, record data for all environments.
            force_links (list of str, optional): List of link names to record six-dimensional forces.
        """
        if env_idx is not None:
            buf = self.data_buffers[env_idx]
            # 原有数据记录
            if hasattr(self, "tiled_camera") and self.tiled_camera is not None:
                buf["camera"]["front"].append(self.tiled_camera.data.output["rgb"][env_idx].to("cpu").clone())
            if hasattr(self, "wrist_tiled_camera") and self.wrist_tiled_camera is not None:
                buf["camera"]["wrist"].append(self.wrist_tiled_camera.data.output["rgb"][env_idx].to("cpu").clone())
            buf["joints"].append(self.joint_pos[env_idx].to("cpu").clone())
            ctrl_target_gripper_dof_pos=0.0,
            buf["gripper"].append(ctrl_target_gripper_dof_pos)
            buf["ee_pose"].append(torch.cat([self.fingertip_midpoint_pos[env_idx], self.fingertip_midpoint_quat[env_idx]], dim=0).to("cpu"))

            # 记录指定 link 的六维力
            if force_links is not None:
                forces = []
                for link_name in force_links:
                    forces.append(get_force(self,link_name, env_idx).to("cpu"))
                forces = torch.cat(forces, dim=0)
                buf["force"].append(forces)
        else:
            for i in range(self.num_envs):
                record_data(self,i, force_links)
 
def _save_array_to_csv(data_list, episode_dir, filename, column_prefix):
        """
        Helper function to save a time-series data list to CSV.

        Args:
            data_list (list): List of tensors to save (e.g., joints, actions)
            episode_dir (str): Directory to save CSV
            filename (str): CSV file name
            column_prefix (str): Prefix for CSV column names
        """
        if not data_list or len(data_list) <= 1:
            print(f"No data or insufficient data ({column_prefix}), skipping save.")
            return

        file_path = os.path.join(episode_dir, filename)

        try:
            num_columns = data_list[0].shape[0]
            header = [f'{column_prefix}_{j}' for j in range(num_columns)]

            with open(file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(header)

                for i in range(1, len(data_list)):  # skip first frame
                    row_data = data_list[i]
                    if hasattr(row_data, 'cpu'):
                        row_data = row_data.detach().cpu().numpy()
                    csv_writer.writerow(row_data)

            print(f"Data ({column_prefix}) saved to {file_path}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            
            
            
def save_data_to_disk(self, env_idx=None):
    """
    Save the buffered data to disk for one or more environments.

    Args:
        env_idx (int, list, np.ndarray, optional): Index or indices of environments to save.
                                                If None, save all environments.
    """
    def preprocess_frame(frame):
        # Convert to numpy
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        elif not isinstance(frame, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(frame)}. Expected torch.Tensor or numpy.ndarray.")

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
        env_indices = [int(env_idx)]   # single int or numpy scalar

    for idx in env_indices:
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
            if len(camera_list) <= 1:  # skip if no data or only first frame
                continue

            save_dir = os.path.join(episode_dir, key)
            os.makedirs(save_dir, exist_ok=True)

            # Get first frame size (C, H, W)
            first_frame = preprocess_frame(camera_list[0])
            last_frame = preprocess_frame(camera_list[-1])
            height, width, channels = first_frame.shape

            # Define video writer (fps can be your data_save_rate, e.g., 20)
            fps = int(1 / (self.physics_dt * self.cfg.decimation))
            video_path = os.path.join(save_dir, f'{key}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write all frames (skip first)
            for i in range(1, len(camera_list)):
                frame = preprocess_frame(camera_list[i])
                if frame.min() >= 0.0 and frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

                if frame.shape[0] in [1, 3]:  # C,H,W
                    frame = np.transpose(frame, (1, 2, 0))

                # If grayscale, convert to 3-channel
                if frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:  # 转成 BGR格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                video_writer.write(frame)

            video_writer.release()
            print(f"Saved {key} video to {video_path}")

            # save last frame
            Image.fromarray(last_frame).save(os.path.join(save_dir, 'last_frame.png'))

        # Save joints, actions, EE pose
        _save_array_to_csv(buf["joints"], episode_dir, 'joint_states.csv', 'joint')
        _save_array_to_csv(buf["force"], episode_dir, 'force.csv', 'joint')
        _save_array_to_csv(buf["gripper"], episode_dir, 'gripper.csv', 'gripper')
        _save_array_to_csv(buf["actions"], episode_dir, 'actions.csv', 'action')
        _save_array_to_csv(buf["ee_pose"], episode_dir, 'ee_pose.csv', 'ee_pose')