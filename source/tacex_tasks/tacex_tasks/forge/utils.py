import numpy as np
from typing import Sequence, Union, List, Optional
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List, Optional, Protocol



ArrayLike = Union[Sequence[float], np.ndarray]

class PoseTrajectoryPlanner:
    """
    Pose format:
        [x, y, z, qw, qx, qy, qz]

    Output action format:
        [x, y, z, qw, qx, qy, qz, gripper]
    """

    def __init__(self):
        pass

    @staticmethod
    def _wxyz_to_xyzw(q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    @staticmethod
    def _xyzw_to_wxyz(q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

    def _slerp(self, q0_wxyz: ArrayLike, q1_wxyz: ArrayLike, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        q0_xyzw = self._wxyz_to_xyzw(q0_wxyz)
        q1_xyzw = self._wxyz_to_xyzw(q1_wxyz)

        rotations = R.from_quat([q0_xyzw, q1_xyzw])
        slerp = Slerp([0.0, 1.0], rotations)
        q_interp_xyzw = slerp([alpha])[0].as_quat()

        return self._xyzw_to_wxyz(q_interp_xyzw)

    def plan(
        self,
        cur_pose: ArrayLike,
        target_pose: ArrayLike,
        num_steps: int,
        gripper: float = 0.0,
        include_start: bool = True,
    ) -> List[List[float]]:
        """
        Generate a pose trajectory segment in exactly num_steps interpolation steps.

        Args:
            cur_pose: [x, y, z, qw, qx, qy, qz]
            target_pose: [x, y, z, qw, qx, qy, qz]
            num_steps: total interpolation steps
            gripper: gripper value
            include_start:
                - True: trajectory starts from cur_pose and ends at target_pose
                - False: trajectory excludes cur_pose, but still ends at target_pose

        Returns:
            List of actions:
            [x, y, z, qw, qx, qy, qz, gripper]
        """
        cur_pose = np.asarray(cur_pose, dtype=np.float64)
        target_pose = np.asarray(target_pose, dtype=np.float64)

        if cur_pose.shape != (7,) or target_pose.shape != (7,):
            raise ValueError("Pose must have shape (7,), i.e. [x, y, z, qw, qx, qy, qz].")

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        cur_pos = cur_pose[:3]
        tgt_pos = target_pose[:3]

        cur_quat = cur_pose[3:]
        tgt_quat = target_pose[3:]

        trajectory: List[List[float]] = []

        q0_xyzw = self._wxyz_to_xyzw(cur_quat)
        q1_xyzw = self._wxyz_to_xyzw(tgt_quat)
        rotations = R.from_quat([q0_xyzw, q1_xyzw])
        slerp = Slerp([0.0, 1.0], rotations)

        if include_start:
            alphas = np.linspace(0.0, 1.0, num_steps)
        else:
            alphas = np.linspace(0.0, 1.0, num_steps + 1)[1:]

        for alpha in alphas:
            pos = (1.0 - alpha) * cur_pos + alpha * tgt_pos
            quat_xyzw = slerp([alpha])[0].as_quat()
            quat_wxyz = self._xyzw_to_wxyz(quat_xyzw)

            trajectory.append([
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat_wxyz[0]), float(quat_wxyz[1]),
                float(quat_wxyz[2]), float(quat_wxyz[3]),
                float(gripper),
            ])

        return trajectory


class BaseMotionPlanner(Protocol):
    def plan(self) -> List[List[float]]:
        """
        Return a trajectory:
        [x, y, z, qw, qx, qy, qz, gripper]
        """
        ...


class MotionPolicy:
    """
    Merge trajectories from multiple motion planners.

    Each action is:
        [x, y, z, qw, qx, qy, qz, gripper]
    """

    def __init__(self, motion_planners: Optional[List[BaseMotionPlanner]] = None, eps: float = 1e-6):
        self.motion_planners: List[BaseMotionPlanner] = motion_planners or []
        self.eps = float(eps)

        self.trajectory: List[List[float]] = []
        self.cursor: int = 0

    def reset(self) -> None:
        self.trajectory = []
        self.cursor = 0

    def set_motion_planners(self, motion_planners: List[BaseMotionPlanner]) -> None:
        self.motion_planners = motion_planners

    def add_motion_planner(self, motion_planner: BaseMotionPlanner) -> None:
        self.motion_planners.append(motion_planner)

    def _append_segment(self, segment: List[List[float]]) -> None:
        if not segment:
            return

        if not self.trajectory:
            self.trajectory.extend(segment)
            return

        prev = np.asarray(self.trajectory[-1], dtype=np.float64)
        first = np.asarray(segment[0], dtype=np.float64)

        # 若前一段末尾和后一段开头一样，则去掉重复点
        if np.allclose(prev, first, atol=self.eps):
            self.trajectory.extend(segment[1:])
        else:
            self.trajectory.extend(segment)

    def build(self) -> List[List[float]]:
        """
        Merge all motion planners into one trajectory.
        """
        self.reset()

        for planner in self.motion_planners:
            segment = planner.plan()
            self._append_segment(segment)

        return self.trajectory

    def get_action(self) -> Optional[List[float]]:
        """
        Output one action at a time.
        Returns None when trajectory is exhausted.
        """
        if self.cursor >= len(self.trajectory):
            return None

        action = self.trajectory[self.cursor]
        self.cursor += 1
        return action

    def get_all_actions(self) -> List[List[float]]:
        return self.trajectory.copy()

    def is_done(self) -> bool:
        return self.cursor >= len(self.trajectory)

    def remaining_steps(self) -> int:
        return max(0, len(self.trajectory) - self.cursor)