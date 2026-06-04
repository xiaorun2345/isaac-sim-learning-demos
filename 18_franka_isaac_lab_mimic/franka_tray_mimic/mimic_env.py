"""Mimic environment adapter for the Demo 17 Franka pick-to-tray task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv

from .scene_contract import TRAY_CENTER_XY


class FrankaTrayIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """Add the fixed tray reference frame and custom automatic subtask signals."""

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        scene_state = self.scene.get_state(is_relative=True)
        robot_root_pose = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]
        object_poses: dict[str, torch.Tensor] = {}

        for object_name, object_state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                object_state["root_pose"][env_ids, :3],
                object_state["root_pose"][env_ids, 3:7],
            )
            object_poses[object_name] = PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quaternion))

        tray_position = torch.tensor(
            [TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], 0.0],
            dtype=torch.float32,
            device=self.device,
        ).repeat(root_pos.shape[0], 1)
        tray_rotation = torch.eye(3, dtype=torch.float32, device=self.device).repeat(root_pos.shape[0], 1, 1)
        object_poses["tray"] = PoseUtils.make_pose(tray_position, tray_rotation)
        return object_poses

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        terms = self.obs_buf["subtask_terms"]
        return {
            "grasp": terms["grasp"][env_ids],
            "placed_in_tray": terms["placed_in_tray"][env_ids],
        }

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """The red cube should remain attached during the transfer subtask."""

        return "cube_2" if subtask_index == 1 else None
