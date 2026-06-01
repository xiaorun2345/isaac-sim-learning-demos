"""Helpers for loading and commanding Franka."""

import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path


FRANKA_PRIM_PATH = "/World/Franka"


def get_franka_usd_path():
    return get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def load_franka(prim_path=FRANKA_PRIM_PATH, name="franka"):
    add_reference_to_stage(get_franka_usd_path(), prim_path)
    return SingleArticulation(prim_path=prim_path, name=name)


def home_positions():
    return np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)


def home_action():
    return ArticulationAction(joint_positions=home_positions())


def gripper_open_action():
    return ArticulationAction(
        joint_positions=np.array([0.04, 0.04], dtype=np.float32),
        joint_indices=np.array([7, 8], dtype=np.int32),
    )


def gripper_close_action():
    return ArticulationAction(
        joint_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_indices=np.array([7, 8], dtype=np.int32),
    )


def arm_pose_action(joint_values):
    return ArticulationAction(joint_positions=np.array(joint_values, dtype=np.float32))
