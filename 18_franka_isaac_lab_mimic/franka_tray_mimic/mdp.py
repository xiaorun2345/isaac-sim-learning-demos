"""Custom observation and termination terms for the Franka-to-tray task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .scene_contract import (
    TRAY_CENTER_XY,
    TRAY_OUTER_X,
    TRAY_OUTER_Y,
    TRAY_SUCCESS_MAX_Z,
    TRAY_WALL_THICKNESS,
)


def tray_position(env) -> torch.Tensor:
    """Return the fixed tray reference position in each environment frame."""

    position = torch.tensor(
        [TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], 0.0],
        dtype=torch.float32,
        device=env.device,
    )
    return position.repeat(env.num_envs, 1)


def target_cube_in_tray(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Return one boolean per environment using Demo 17's success geometry."""

    target_cube: RigidObject = env.scene[object_cfg.name]
    position = target_cube.data.root_pos_w - env.scene.env_origins
    inner_half_x = TRAY_OUTER_X / 2.0 - TRAY_WALL_THICKNESS
    inner_half_y = TRAY_OUTER_Y / 2.0 - TRAY_WALL_THICKNESS
    within_x = torch.abs(position[:, 0] - TRAY_CENTER_XY[0]) < inner_half_x
    within_y = torch.abs(position[:, 1] - TRAY_CENTER_XY[1]) < inner_half_y
    within_z = position[:, 2] < TRAY_SUCCESS_MAX_Z
    return within_x & within_y & within_z
