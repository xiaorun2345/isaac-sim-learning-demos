"""Route ROS 2 links and run passive or scripted collection for selected Franka workcells.

Run with Isaac Sim's Python environment:

    python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py

这个示例建立在 `19_franka_multi_station_scene` 之上，但重点不再只是“复制场景”，
而是进一步演示“如何路由”和“如何并行采集”：

1. 让 ROS 2 只连某一个机械臂，或者同时连全部机械臂
2. 让 ROS 2 只发布某一个工位的某一路相机，或者发布全部工位相机
3. 让本地采集器只记录某一个工位，或者把全部工位分别采下来
4. 按 `17_franka_smolvla_data_collection` 的抓取放置策略，让多个工位并行采集成功轨迹

因此它既可以继续当“多工位选择器 demo”，也可以直接当“多机械臂并行采集 demo”。
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析演示所需参数。"""

    # 这个 demo 的核心不是机器人控制算法本身，而是“路由”：
    # 同一份多工位场景里，到底哪一个 env 要开放给 ROS 2，哪一个 env 要被采集。
    #
    # 所以参数设计上故意采用：
    # - `off`   : 完全关闭某类能力
    # - `single`: 只选择一个 env_XX
    # - `all`   : 选择全部 env
    #
    # 这样后续不管是继续接手动控制、策略控制还是批量采集，选择规则都能复用。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无界面运行。")
    parser.add_argument("--num-envs", type=int, default=4, choices=(4, 5, 6), help="工位数量。")

    parser.add_argument(
        "--ros2-arm",
        choices=("off", "single", "all"),
        default="off",
        help="机械臂 ROS 2 桥模式。",
    )
    parser.add_argument("--ros2-arm-env", type=int, default=0, help="当 `--ros2-arm single` 时选择哪一个工位。")

    parser.add_argument(
        "--ros2-camera",
        choices=("off", "single", "all"),
        default="off",
        help="相机 ROS 2 发布模式。",
    )
    parser.add_argument("--ros2-camera-env", type=int, default=0, help="当 `--ros2-camera single` 时选择哪一个工位。")
    parser.add_argument(
        "--ros2-camera-name",
        choices=("front", "wrist", "both"),
        default="front",
        help="发布哪一路相机。",
    )

    parser.add_argument(
        "--collect",
        choices=("off", "single", "all"),
        default="off",
        help="本地采集模式。",
    )
    parser.add_argument(
        "--collect-policy",
        choices=("expert", "passive"),
        default="expert",
        help="采集策略：`expert` 使用 17_demo 的脚本化抓放策略，`passive` 只被动记录。",
    )
    parser.add_argument("--collect-env", type=int, default=0, help="当 `--collect single` 时选择哪一个工位。")
    parser.add_argument("--episodes", type=int, default=20, help="expert 采集模式下，要成功保存多少条轨迹。")
    parser.add_argument("--collect-steps", type=int, default=240, help="每个工位记录多少帧。")
    parser.add_argument("--collect-every", type=int, default=1, help="每隔多少个 sim step 记录一次。")
    parser.add_argument(
        "--compress",
        action="store_true",
        help="expert 采集模式下是否使用 np.savez_compressed 压缩保存。默认关闭以提升速度。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "raw",
        help="采集输出目录。",
    )
    return parser.parse_args()


ARGS = parse_args()

# Isaac Sim 的启动顺序要求比较严格：
# 必须先创建 `SimulationApp`，后面才能安全导入 Isaac / Omniverse 相关模块。
simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1600,
        "height": 900,
    }
)


from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux

from common.ros2_utils import decode_joint_command


# ----------------------------
# 多工位场景基础尺寸
# ----------------------------
# 这部分坐标和 19_demo 保持一致，目的不是重新设计场景，
# 而是把“路由 / ROS2 / 采集”问题隔离出来单独演示。
TABLE_H = 0.40
TABLE_CENTER_LOCAL = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H

FRONT_CAMERA_EYE_LOCAL = np.array([1.15, -1.10, 1.10], dtype=np.float32)
FRONT_CAMERA_TARGET_LOCAL = np.array([0.40, 0.0, 0.65], dtype=np.float32)
FRONT_CAMERA_ROTATION = (-35.0, 0.0, 45.0)
WRIST_CAMERA_LOCAL_POS = (0.06, 0.0, 0.03)
WRIST_CAMERA_LOCAL_ROT = (-95.0, 0.0, -90.0)
FRONT_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 20
CAMERA_WARMUP_STEPS = 20
CAMERA_CAPTURE_RETRIES = 6
DEBUG_PRINT_EVERY_STEPS = 8

CUBES = (
    ("cube_red", np.array([0.35, -0.18, TABLE_SURFACE_Z + 0.0275], dtype=np.float32), np.array([0.055, 0.055, 0.055], dtype=np.float32), np.array([0.90, 0.15, 0.10], dtype=np.float32)),
    ("cube_blue", np.array([0.50, -0.05, TABLE_SURFACE_Z + 0.0450], dtype=np.float32), np.array([0.040, 0.040, 0.090], dtype=np.float32), np.array([0.20, 0.40, 0.90], dtype=np.float32)),
    ("cube_green", np.array([0.62, 0.10, TABLE_SURFACE_Z + 0.0250], dtype=np.float32), np.array([0.070, 0.070, 0.050], dtype=np.float32), np.array([0.15, 0.80, 0.25], dtype=np.float32)),
    ("cube_yellow", np.array([0.75, -0.20, TABLE_SURFACE_Z + 0.0375], dtype=np.float32), np.array([0.055, 0.055, 0.075], dtype=np.float32), np.array([0.95, 0.80, 0.10], dtype=np.float32)),
)

PLACE_BOX_CENTER_LOCAL = np.array([0.82, 0.16], dtype=np.float32)
PLACE_BOX_OUTER_X = 0.24
PLACE_BOX_OUTER_Y = 0.20
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.018
PLACE_BOX_WALL_H = 0.13

ENV_SPACING_X = 1.80
ENV_SPACING_Y = 1.50
WORKCELL_HALF_EXTENT_X = 0.95
WORKCELL_HALF_EXTENT_Y = 0.95

EE_FEEDBACK_Z_BIAS = 0.0985
TARGET_CUBE_NAME = "cube_red"
TARGET_CUBE_SCALE = np.array([0.055, 0.055, 0.055], dtype=np.float32)
TARGET_CUBE_HALF_Z = float(TARGET_CUBE_SCALE[2] / 2.0)
TARGET_CUBE_SPAWN_REGIONS_LOCAL = [
    ("left_front", (0.28, 0.40), (0.10, 0.24)),
    ("left_mid", (0.28, 0.42), (-0.08, 0.08)),
    ("left_back", (0.28, 0.40), (-0.26, -0.12)),
    ("center_front", (0.44, 0.58), (0.00, 0.16)),
    ("center_back", (0.44, 0.62), (-0.26, -0.08)),
    ("right_back", (0.58, 0.72), (-0.28, -0.12)),
]
TARGET_CUBE_BOX_EXCLUSION_MARGIN = 0.050
TARGET_CUBE_DISTRACTOR_CLEARANCE_XY = 0.090
TARGET_CUBE_SAMPLE_MAX_TRIES = 80
DISTRACTOR_CUBE_LAYOUT_LOCAL = {
    cube_name: local_position
    for cube_name, local_position, _, _ in CUBES
    if cube_name != TARGET_CUBE_NAME
}
PLACE_GOAL_POSITION_LOCAL = np.array(
    [PLACE_BOX_CENTER_LOCAL[0], PLACE_BOX_CENTER_LOCAL[1], TABLE_SURFACE_Z + TARGET_CUBE_HALF_Z],
    dtype=np.float32,
)
HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)
ACTION_NAMES = [
    "target_ee_pos_x",
    "target_ee_pos_y",
    "target_ee_pos_z",
    "target_gripper_closed",
]
TASK_DESCRIPTION = "Pick up the red cube with Franka and place it into the tray in a multi-station scene."
CAPTURE_EVERY_STEPS = max(1, int(ARGS.collect_every))
EE_PICK_APPROACH_Z_OFFSET = 0.060
EE_PICK_ACTION_Z_OFFSET = 0.013
EE_PLACE_APPROACH_Z_OFFSET = 0.110
EE_PLACE_ACTION_Z_OFFSET = 0.085
EE_HOVER_MARGIN = 0.140
GRASP_HOLD_STEPS = 26
RELEASE_HOLD_STEPS = 26
PHASE_STABLE_FRAMES = 2
PRE_GRASP_STABLE_FRAMES = 6
CLOSE_GRIPPER_STABLE_FRAMES = 4
PRE_RELEASE_STABLE_FRAMES = 4
OPEN_GRIPPER_STABLE_FRAMES = 4
GRASP_CLOSE_COMMAND_STEPS = 10
RELEASE_OPEN_COMMAND_STEPS = 10
POST_RELEASE_WAIT_STEPS = 12
EE_GENERAL_REACH_XY_THRESHOLD = 0.030
EE_GENERAL_REACH_Z_THRESHOLD = 0.030
EE_PICK_REACH_XY_THRESHOLD = 0.015
EE_PICK_REACH_Z_THRESHOLD = 0.020
EE_GRASP_ALIGN_XY_THRESHOLD = 0.008
EE_GRASP_ALIGN_Z_THRESHOLD = 0.012
EE_PLACE_REACH_XY_THRESHOLD = 0.040
EE_PLACE_REACH_Z_THRESHOLD = 0.150
GRIPPER_CLOSE_WIDTH_THRESHOLD = 0.065
GRIPPER_OPEN_WIDTH_THRESHOLD = 0.085
GRASP_VERIFY_XY_THRESHOLD = 0.020
GRASP_VERIFY_Z_THRESHOLD = 0.030
GRASP_LIFT_MIN_HEIGHT = 0.010
EPISODE_MAX_STEPS = 520
EPISODE_SETTLE_STEPS = 20
STATE_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "ee_quat_w",
    "ee_quat_x",
    "ee_quat_y",
    "ee_quat_z",
    "gripper_width",
]
COMMAND_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "finger_1",
    "finger_2",
]
DEFAULT_COMMAND = np.full(9, np.nan, dtype=np.float32)
EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))


def env_name(env_index: int) -> str:
    # 所有外部接口都统一基于 `env_XX` 命名。
    # 这样：
    # 1. USD prim 路径稳定
    # 2. ROS2 话题稳定
    # 3. 采集文件名也稳定
    return f"env_{env_index:02d}"


def env_root(env_index: int) -> str:
    return f"/World/Envs/{env_name(env_index)}"


def franka_prim_path(env_index: int) -> str:
    return f"{env_root(env_index)}/Franka"


def front_camera_path(env_index: int) -> str:
    return f"{env_root(env_index)}/front_camera"


def wrist_camera_path(env_index: int) -> str:
    return f"{franka_prim_path(env_index)}/panda_hand/wrist_camera"


def translated(origin: np.ndarray, local_xyz: np.ndarray) -> np.ndarray:
    return np.array([origin[0] + local_xyz[0], origin[1] + local_xyz[1], local_xyz[2]], dtype=np.float32)


def env_layout(num_envs: int) -> list[np.ndarray]:
    """生成多工位原点布局。"""

    # 这里延续 19_demo 的布局规则：
    # - 4 个工位：2x2
    # - 5/6 个工位：3x2
    #
    # 返回的是每个工位在世界坐标系中的 XY 原点。
    # 后面每个工位内部的桌子、机器人、方块、相机，都会在这个原点基础上叠加局部坐标。
    columns = 2 if num_envs <= 4 else 3
    rows = int(math.ceil(num_envs / columns))
    y_offsets = ((rows - 1) / 2.0 - np.arange(rows, dtype=np.float32)) * ENV_SPACING_Y

    origins: list[np.ndarray] = []
    for row_index in range(rows):
        remaining = num_envs - len(origins)
        row_count = min(columns, remaining)
        x_offsets = (np.arange(row_count, dtype=np.float32) - (row_count - 1) / 2.0) * ENV_SPACING_X
        for column_index in range(row_count):
            origins.append(np.array([x_offsets[column_index], y_offsets[row_index]], dtype=np.float32))
            if len(origins) >= num_envs:
                return origins
    return origins


def require_valid_env_index(env_index: int, num_envs: int, arg_name: str) -> None:
    if not 0 <= env_index < num_envs:
        raise ValueError(f"{arg_name}={env_index} is out of range for num_envs={num_envs}.")


def selected_env_indices(mode: str, single_env: int, num_envs: int) -> list[int]:
    """把 off/single/all 三种模式统一转换成 env 下标列表。"""

    # 这个函数是整个 demo 的“路由分发核心”之一。
    # 后面：
    # - 机械臂 ROS2 桥
    # - 相机 ROS2 发布
    # - 本地采集器
    # 都复用同一套选择逻辑，避免每一块代码各自写一套 if/else。
    if mode == "off":
        return []
    if mode == "all":
        return list(range(num_envs))
    require_valid_env_index(single_env, num_envs, "env index")
    return [single_env]


def create_camera_prim(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """直接在 USD stage 上创建相机 prim。

    注意这里创建的是“相机对象本体”，不是传感器包装器。
    后面如果要本地采图，会再用 `isaacsim.sensors.camera.Camera`
    去包一层传感器接口。
    """

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights(origins: list[np.ndarray]) -> None:
    """创建全局环境光和每个工位单独的顶灯。"""

    stage = get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1000.0)

    for env_index, origin in enumerate(origins):
        key = UsdLux.RectLight.Define(stage, f"/World/Lights/EnvKey_{env_index:02d}")
        key.CreateIntensityAttr(3600.0)
        key.CreateWidthAttr(1.4)
        key.CreateHeightAttr(1.0)
        xform = UsdGeom.XformCommonAPI(key.GetPrim())
        xform.SetTranslate(Gf.Vec3d(float(origin[0] + 0.65), float(origin[1] - 0.20), 1.80))
        xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World, origins: list[np.ndarray]) -> None:
    """根据工位分布动态生成一个足够大的房间包围盒。"""

    # 房间尺寸不写死，而是根据当前工位布局自动扩展。
    # 这样 4 / 5 / 6 个工位都能共用同一个构建函数，不会出现相机或墙体把场景裁掉。
    x_values = np.array([origin[0] for origin in origins], dtype=np.float32)
    y_values = np.array([origin[1] for origin in origins], dtype=np.float32)

    min_x = float(np.min(x_values) - WORKCELL_HALF_EXTENT_X - 0.60)
    max_x = float(np.max(x_values) + WORKCELL_HALF_EXTENT_X + 0.60)
    min_y = float(np.min(y_values) - WORKCELL_HALF_EXTENT_Y - 0.60)
    max_y = float(np.max(y_values) + WORKCELL_HALF_EXTENT_Y + 0.60)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    room_size_x = max_x - min_x
    room_size_y = max_y - min_y

    world.scene.add(
        FixedCuboid(
            name="room_floor",
            prim_path="/World/Room/Floor",
            position=np.array([center_x, center_y, -0.025], dtype=np.float32),
            scale=np.array([room_size_x, room_size_y, 0.05], dtype=np.float32),
            size=1.0,
            color=np.array([0.34, 0.35, 0.36], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_back_wall",
            prim_path="/World/Room/BackWall",
            position=np.array([center_x, max_y, 1.20], dtype=np.float32),
            scale=np.array([room_size_x, 0.04, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_front_wall",
            prim_path="/World/Room/FrontWall",
            position=np.array([center_x, min_y, 1.20], dtype=np.float32),
            scale=np.array([room_size_x, 0.04, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_left_wall",
            prim_path="/World/Room/LeftWall",
            position=np.array([min_x, center_y, 1.20], dtype=np.float32),
            scale=np.array([0.04, room_size_y, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_right_wall",
            prim_path="/World/Room/RightWall",
            position=np.array([max_x, center_y, 1.20], dtype=np.float32),
            scale=np.array([0.04, room_size_y, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )


def add_table(world: World, env_index: int, origin: np.ndarray) -> None:
    """给某个工位放一张桌子。"""

    world.scene.add(
        FixedCuboid(
            name=f"table_{env_index:02d}",
            prim_path=f"{env_root(env_index)}/Table",
            position=translated(origin, TABLE_CENTER_LOCAL),
            scale=TABLE_SIZE,
            size=1.0,
            color=np.array([0.55, 0.35, 0.15], dtype=np.float32),
        )
    )


def add_place_box(world: World, env_index: int, origin: np.ndarray) -> None:
    """给某个工位放一个托盘。

    托盘不是导入复杂 mesh，而是继续沿用底板 + 四面墙的拼装方式。
    这样更便于理解坐标，也方便你后面自己改尺寸。
    """

    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)
    root = f"{env_root(env_index)}/PlaceBox"

    world.scene.add(
        FixedCuboid(
            name=f"place_box_bottom_{env_index:02d}",
            prim_path=f"{root}/Bottom",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1], bottom_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_left_{env_index:02d}",
            prim_path=f"{root}/WallLeft",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] - PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_right_{env_index:02d}",
            prim_path=f"{root}/WallRight",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] + PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_front_{env_index:02d}",
            prim_path=f"{root}/WallFront",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_back_{env_index:02d}",
            prim_path=f"{root}/WallBack",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )


def add_franka(world: World, env_index: int, origin: np.ndarray) -> SingleManipulator:
    """把某个工位的 Franka 机器人加入世界。"""

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    robot_path = franka_prim_path(env_index)
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=robot_path)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{robot_path}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    robot = world.scene.add(
        SingleManipulator(
            prim_path=robot_path,
            name=f"franka_{env_index:02d}",
            end_effector_prim_path=f"{robot_path}/panda_hand",
            gripper=gripper,
            position=np.array([origin[0], origin[1], TABLE_H], dtype=np.float32),
        )
    )
    robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    return robot


def add_cubes(world: World, env_index: int, origin: np.ndarray) -> DynamicCuboid:
    """给某个工位添加红色目标方块和固定干扰方块。"""

    target_cube: DynamicCuboid | None = None
    for cube_name, local_position, scale, color in CUBES:
        cube_world_position = translated(origin, local_position)
        if cube_name == TARGET_CUBE_NAME:
            target_cube = world.scene.add(
                DynamicCuboid(
                    name=f"{cube_name}_{env_index:02d}",
                    prim_path=f"{env_root(env_index)}/{cube_name}",
                    position=cube_world_position,
                    scale=scale,
                    size=1.0,
                    color=color,
                )
            )
        else:
            world.scene.add(
                FixedCuboid(
                    name=f"{cube_name}_{env_index:02d}",
                    prim_path=f"{env_root(env_index)}/{cube_name}",
                    position=cube_world_position,
                    scale=scale,
                    size=1.0,
                    color=color,
                )
            )
    if target_cube is None:
        raise RuntimeError(f"Target cube {TARGET_CUBE_NAME} is missing from CUBES definition.")
    return target_cube


def add_station_cameras(env_index: int, origin: np.ndarray) -> None:
    """为某个工位创建前视和手腕相机 prim。"""

    create_camera_prim(
        path=front_camera_path(env_index),
        position=tuple(translated(origin, FRONT_CAMERA_EYE_LOCAL).tolist()),
        rotation_xyz_deg=FRONT_CAMERA_ROTATION,
        focal_length=10.0,
    )
    create_camera_prim(
        path=wrist_camera_path(env_index),
        position=WRIST_CAMERA_LOCAL_POS,
        rotation_xyz_deg=WRIST_CAMERA_LOCAL_ROT,
        focal_length=4.0,
    )


def overview_camera_pose(origins: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """计算默认透视视口的总览位置。"""

    # 这里返回的是 `/OmniverseKit_Persp` 的总览视角，不是某个工位自己的前视相机。
    # 目标是脚本一打开就能在 UI 里直接看到整片工作区。
    x_values = np.array([origin[0] for origin in origins], dtype=np.float32)
    y_values = np.array([origin[1] for origin in origins], dtype=np.float32)

    min_x = float(np.min(x_values) - WORKCELL_HALF_EXTENT_X - 0.60)
    max_x = float(np.max(x_values) + WORKCELL_HALF_EXTENT_X + 0.60)
    min_y = float(np.min(y_values) - WORKCELL_HALF_EXTENT_Y - 0.60)
    max_y = float(np.max(y_values) + WORKCELL_HALF_EXTENT_Y + 0.60)

    center_x = float((min_x + max_x) / 2.0)
    center_y = float((min_y + max_y) / 2.0)
    room_size_x = max_x - min_x
    room_size_y = max_y - min_y

    target = np.array([center_x + 0.45, center_y, 0.55], dtype=np.float32)
    eye = np.array(
        [
            center_x + min(1.20, room_size_x * 0.16),
            min_y + min(1.10, room_size_y * 0.20),
            2.55,
        ],
        dtype=np.float32,
    )
    return eye, target


def build_scene(num_envs: int) -> tuple[World, dict[int, SingleManipulator], dict[int, DynamicCuboid], list[np.ndarray]]:
    """构建多工位场景，并返回世界对象、机器人句柄和目标方块句柄。"""

    # 返回值里把机器人放进 `dict[int, SingleManipulator]`，
    # 是因为后面 ROS2 路由和采集器都天然是“按 env_index 索引”的。
    origins = env_layout(num_envs)
    world = World(stage_units_in_meters=1.0)
    create_lights(origins)
    add_room(world, origins)
    world.scene.add_default_ground_plane()

    robots: dict[int, SingleManipulator] = {}
    target_cubes: dict[int, DynamicCuboid] = {}
    for env_index, origin in enumerate(origins):
        add_table(world, env_index, origin)
        add_place_box(world, env_index, origin)
        robots[env_index] = add_franka(world, env_index, origin)
        target_cubes[env_index] = add_cubes(world, env_index, origin)
        add_station_cameras(env_index, origin)

    world.reset()

    for env_index, origin in enumerate(origins):
        set_camera_view(
            eye=translated(origin, FRONT_CAMERA_EYE_LOCAL),
            target=translated(origin, FRONT_CAMERA_TARGET_LOCAL),
            camera_prim_path=front_camera_path(env_index),
        )

    if not ARGS.headless:
        eye, target = overview_camera_pose(origins)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    return world, robots, target_cubes, origins


def enable_ros2_bridge_extension() -> None:
    """确保 Isaac Sim 的 ROS 2 bridge 扩展已经启用。"""

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    enabled_names = {ext["name"] for ext in manager.get_extensions() if ext.get("enabled")}
    if "isaacsim.ros2.bridge" not in enabled_names:
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        omni.kit.app.get_app().update()


def create_ros2_camera_graph(
    graph_path: str,
    camera_prim: str,
    namespace: str,
    frame_id: str,
    width: int,
    height: int,
) -> str:
    """为某一路相机创建一个独立的 ROS2 图像发布 OmniGraph。"""

    # 这里每一路相机都建自己的 graph，而不是所有相机共用一个 graph。
    # 好处是：
    # 1. 每个 env / 每路相机的话题更清晰
    # 2. 后续想只开某几路相机时，控制粒度更细
    import omni.graph.core as og

    enable_ros2_bridge_extension()
    stage = get_current_stage()
    if stage.GetPrimAtPath(graph_path).IsValid():
        return graph_path

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "PublishRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "PublishRgb.inputs:renderProductPath"),
                ("ROS2Context.outputs:context", "PublishRgb.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", [Sdf.Path(camera_prim)]),
                ("CreateRenderProduct.inputs:width", max(1, width)),
                ("CreateRenderProduct.inputs:height", max(1, height)),
                ("ROS2Context.inputs:useDomainIDEnvVar", False),
                ("PublishRgb.inputs:frameId", frame_id),
                ("PublishRgb.inputs:nodeNamespace", namespace),
                ("PublishRgb.inputs:queueSize", 1),
                ("PublishRgb.inputs:topicName", "rgb"),
                ("PublishRgb.inputs:type", "rgb"),
            ],
        },
    )
    return graph_path


def enable_selected_camera_publishers(selected_envs: list[int], camera_name: str) -> list[str]:
    """根据选择规则创建相机 ROS 2 发布图。"""

    # `selected_envs` 已经是上面统一选择函数算好的最终结果，
    # 这里不再关心用户到底传的是 `single` 还是 `all`。
    graph_paths: list[str] = []
    for env_index in selected_envs:
        if camera_name in {"front", "both"}:
            graph_paths.append(
                create_ros2_camera_graph(
                    graph_path=f"/World/ROS2Graphs/{env_name(env_index)}_front",
                    camera_prim=front_camera_path(env_index),
                    namespace=f"{env_name(env_index)}/front_camera",
                    frame_id=f"{env_name(env_index)}_front_camera",
                    width=FRONT_CAMERA_RESOLUTION[0],
                    height=FRONT_CAMERA_RESOLUTION[1],
                )
            )
        if camera_name in {"wrist", "both"}:
            graph_paths.append(
                create_ros2_camera_graph(
                    graph_path=f"/World/ROS2Graphs/{env_name(env_index)}_wrist",
                    camera_prim=wrist_camera_path(env_index),
                    namespace=f"{env_name(env_index)}/wrist_camera",
                    frame_id=f"{env_name(env_index)}_wrist_camera",
                    width=WRIST_CAMERA_RESOLUTION[0],
                    height=WRIST_CAMERA_RESOLUTION[1],
                )
            )
    return graph_paths


def capture_rgb(camera: Camera) -> np.ndarray:
    """从 Isaac 相机传感器读取一帧 RGB，并对启动初期的空帧做重试。"""

    for _ in range(CAMERA_CAPTURE_RETRIES):
        rgb = camera.get_rgb()
        if rgb is not None:
            return np.asarray(rgb, dtype=np.uint8)

        rgba_getter = getattr(camera, "get_rgba", None)
        if callable(rgba_getter):
            rgba = rgba_getter()
            if rgba is not None:
                return np.asarray(rgba, dtype=np.uint8)[..., :3]

        simulation_app.update()

    raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")


def current_sim_time(world: World, frame_index: int) -> float:
    """兼容不同 Isaac 版本的仿真时间读取。"""

    # 某些版本直接有 `world.current_time`，
    # 如果没有，就退化成基于采样频率的近似时间轴。
    time_value = getattr(world, "current_time", None)
    if time_value is None:
        return float(frame_index) / float(CAMERA_FREQUENCY)
    return float(time_value)


def get_task_space_ee_pose(franka: SingleManipulator) -> tuple[np.ndarray, np.ndarray]:
    """读取与任务空间动作更一致的末端位姿。"""

    # 这里沿用 17_demo 里的修正：
    # `panda_hand` 反馈点和控制用的任务空间参考点在 Z 上有稳定偏差。
    # 如果不减掉这个偏差，保存出来的状态和控制语义会对不上。
    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32).copy()
    ee_position[2] -= EE_FEEDBACK_Z_BIAS
    ee_orientation = np.asarray(ee_orientation, dtype=np.float32)
    return ee_position, ee_orientation


def get_robot_state(franka: SingleManipulator) -> np.ndarray:
    """组装采集时使用的 15 维状态向量。"""

    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    ee_position, ee_orientation = get_task_space_ee_pose(franka)
    gripper_width = float(joint_positions[7] + joint_positions[8])
    return np.concatenate(
        [
            joint_positions[:7],
            ee_position[:3],
            ee_orientation[:4],
            np.array([gripper_width], dtype=np.float32),
        ]
    ).astype(np.float32)


def rounded_list(array: np.ndarray, decimals: int = 4) -> list[float]:
    """把数组整理成更适合终端查看的短列表。"""

    return np.round(np.asarray(array, dtype=np.float32), decimals).tolist()


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个三维点在 XY 平面上的距离。"""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a[:2] - b[:2]))


def vertical_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个三维点在 Z 方向上的绝对距离。"""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(abs(a[2] - b[2]))


def is_position_close(
    current_position: np.ndarray,
    target_position: np.ndarray,
    xy_threshold: float,
    z_threshold: float,
) -> bool:
    """按 XY / Z 分开的阈值判断末端是否到位。"""

    return bool(
        planar_distance(current_position, target_position) <= xy_threshold
        and vertical_distance(current_position, target_position) <= z_threshold
    )


def phase_reach_thresholds(phase_name: str) -> tuple[float, float]:
    """为不同阶段返回不同的末端到位阈值。"""

    if phase_name == "pre_grasp_settle":
        return EE_GRASP_ALIGN_XY_THRESHOLD, EE_GRASP_ALIGN_Z_THRESHOLD
    if phase_name in {"descend_pick", "grasp_close", "grasp_hold"}:
        return EE_PICK_REACH_XY_THRESHOLD, EE_PICK_REACH_Z_THRESHOLD
    if phase_name in {"descend_place", "pre_release_settle", "release_open", "post_release_settle"}:
        return EE_PLACE_REACH_XY_THRESHOLD, EE_PLACE_REACH_Z_THRESHOLD
    return EE_GENERAL_REACH_XY_THRESHOLD, EE_GENERAL_REACH_Z_THRESHOLD


def sample_target_cube_local_position(rng: np.random.Generator) -> np.ndarray:
    """在当前工位桌面上为红色 cube 采样一个更分散的初始位置。"""

    for _ in range(TARGET_CUBE_SAMPLE_MAX_TRIES):
        _, x_range, y_range = TARGET_CUBE_SPAWN_REGIONS_LOCAL[rng.integers(len(TARGET_CUBE_SPAWN_REGIONS_LOCAL))]
        local_position = np.array(
            [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                TABLE_SURFACE_Z + TARGET_CUBE_HALF_Z,
            ],
            dtype=np.float32,
        )
        inside_box_x = abs(float(local_position[0] - PLACE_BOX_CENTER_LOCAL[0])) <= (
            PLACE_BOX_OUTER_X / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        inside_box_y = abs(float(local_position[1] - PLACE_BOX_CENTER_LOCAL[1])) <= (
            PLACE_BOX_OUTER_Y / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        if inside_box_x and inside_box_y:
            continue

        blocked = False
        for distractor_position in DISTRACTOR_CUBE_LAYOUT_LOCAL.values():
            if planar_distance(local_position, distractor_position) < TARGET_CUBE_DISTRACTOR_CLEARANCE_XY:
                blocked = True
                break
        if blocked:
            continue
        return local_position

    return np.array([0.46, -0.18, TABLE_SURFACE_Z + TARGET_CUBE_HALF_Z], dtype=np.float32)


def reset_robot(franka: SingleManipulator) -> None:
    """把某一台 Franka 直接复位到统一 home 姿态。"""

    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_target_cube(cube: DynamicCuboid, origin: np.ndarray, local_position: np.ndarray) -> None:
    """把当前工位的目标红色方块复位到新位置。"""

    cube.set_world_pose(
        position=translated(origin, local_position),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
    cube.set_angular_velocity(np.zeros(3, dtype=np.float32))


def merge_joint_actions(num_dof: int, *actions: ArticulationAction) -> ArticulationAction:
    """把机械臂动作和夹爪动作合并成一条控制指令。"""

    merged_positions: list[float | None] = [None] * num_dof
    merged_velocities: list[float | None] = [None] * num_dof
    merged_efforts: list[float | None] = [None] * num_dof

    for action in actions:
        if action is None:
            continue
        _merge_single_field(merged_positions, action.joint_positions, action.joint_indices)
        _merge_single_field(merged_velocities, action.joint_velocities, action.joint_indices)
        _merge_single_field(merged_efforts, action.joint_efforts, action.joint_indices)

    return ArticulationAction(
        joint_positions=merged_positions,
        joint_velocities=merged_velocities,
        joint_efforts=merged_efforts,
    )


def _merge_single_field(
    target: list[float | None],
    values: list[float] | np.ndarray | None,
    indices: list[int] | np.ndarray | None,
) -> None:
    """把一个动作字段写入合并缓冲区。"""

    if values is None:
        return
    if indices is None:
        for index, value in enumerate(values):
            if value is not None:
                target[index] = float(value)
        return
    for index, value in zip(indices, values):
        if value is not None:
            target[int(index)] = float(value)


def build_phase_targets(
    start_ee_position: np.ndarray,
    target_cube_position: np.ndarray,
    place_goal_position: np.ndarray,
) -> list[dict[str, np.ndarray | bool | str]]:
    """生成和 17_demo 一致的抓取放置阶段目标。"""

    pick_approach_position = np.array(
        [
            float(target_cube_position[0]),
            float(target_cube_position[1]),
            float(target_cube_position[2] + EE_PICK_APPROACH_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    pick_grasp_position = np.array(
        [
            float(target_cube_position[0]),
            float(target_cube_position[1]),
            float(target_cube_position[2] + EE_PICK_ACTION_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    pick_hover_position = pick_approach_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    place_approach_position = np.array(
        [
            float(place_goal_position[0]),
            float(place_goal_position[1]),
            float(place_goal_position[2] + EE_PLACE_APPROACH_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    place_release_position = np.array(
        [
            float(place_goal_position[0]),
            float(place_goal_position[1]),
            float(place_goal_position[2] + EE_PLACE_ACTION_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    place_hover_position = place_approach_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    return [
        {"name": "approach_pick_hover", "target": pick_hover_position, "gripper_closed": False},
        {"name": "descend_pick", "target": pick_approach_position, "gripper_closed": False},
        {"name": "pre_grasp_settle", "target": pick_grasp_position, "gripper_closed": False},
        {"name": "grasp_close", "target": pick_grasp_position, "gripper_closed": True},
        {"name": "grasp_hold", "target": pick_grasp_position, "gripper_closed": True},
        {"name": "lift", "target": pick_hover_position, "gripper_closed": True},
        {"name": "transfer", "target": place_hover_position, "gripper_closed": True},
        {"name": "descend_place", "target": place_approach_position, "gripper_closed": True},
        {"name": "pre_release_settle", "target": place_release_position, "gripper_closed": True},
        {"name": "release_open", "target": place_release_position, "gripper_closed": False},
        {"name": "post_release_settle", "target": place_release_position, "gripper_closed": False},
        {"name": "retreat", "target": place_hover_position, "gripper_closed": False},
        {"name": "return_idle", "target": np.asarray(start_ee_position, dtype=np.float32), "gripper_closed": False},
    ]


def make_task_space_action(target_position: np.ndarray, gripper_closed: bool) -> np.ndarray:
    """把当前阶段目标编码成 4 维任务空间动作。"""

    return np.concatenate(
        [
            np.asarray(target_position, dtype=np.float32),
            np.array([1.0 if gripper_closed else 0.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def is_cube_inside_box(cube: DynamicCuboid, origin: np.ndarray) -> bool:
    """判断目标方块是否已经落进当前工位托盘内部。"""

    cube_position, _ = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)
    box_center_world = np.array(
        [origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1], PLACE_GOAL_POSITION_LOCAL[2]],
        dtype=np.float32,
    )
    inner_half_x = PLACE_BOX_OUTER_X / 2.0 - PLACE_BOX_WALL_T
    inner_half_y = PLACE_BOX_OUTER_Y / 2.0 - PLACE_BOX_WALL_T
    within_x = abs(float(cube_position[0] - box_center_world[0])) < inner_half_x
    within_y = abs(float(cube_position[1] - box_center_world[1])) < inner_half_y
    within_z = abs(float(cube_position[2] - box_center_world[2])) < 0.040
    return bool(within_x and within_y and within_z)


def episode_metadata(env_index: int) -> str:
    """生成 expert 采集用的每条轨迹元数据。"""

    return json.dumps(
        {
            "schema_version": 1,
            "task": TASK_DESCRIPTION,
            "env_index": env_index,
            "env_name": env_name(env_index),
            "num_envs": ARGS.num_envs,
            "collect_policy": ARGS.collect_policy,
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
            "front_camera_resolution": FRONT_CAMERA_RESOLUTION,
            "wrist_camera_resolution": WRIST_CAMERA_RESOLUTION,
            "episode_max_steps": EPISODE_MAX_STEPS,
            "capture_every_steps": CAPTURE_EVERY_STEPS,
            "save_compressed": bool(ARGS.compress),
        },
        ensure_ascii=False,
        indent=2,
    )


@dataclass
class RecorderBuffers:
    front_images: list[np.ndarray] = field(default_factory=list)
    wrist_images: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


class EnvRecorder:
    """记录单个工位的图像、状态和最近收到的关节指令。"""

    def __init__(self, env_index: int, front_camera: Camera, wrist_camera: Camera):
        self.env_index = env_index
        self.front_camera = front_camera
        self.wrist_camera = wrist_camera
        self.buffers = RecorderBuffers()

    def record(self, robot: SingleManipulator, latest_command: np.ndarray | None, sim_time: float) -> None:
        # 这里每采一帧，就把两路图像、机器人状态和“最近一次收到的命令”并排存起来。
        # 这样后面分析单工位数据时，不需要再去额外对齐时间轴。
        self.buffers.front_images.append(capture_rgb(self.front_camera))
        self.buffers.wrist_images.append(capture_rgb(self.wrist_camera))
        self.buffers.states.append(get_robot_state(robot))
        self.buffers.commands.append(
            np.asarray(latest_command, dtype=np.float32).copy() if latest_command is not None else DEFAULT_COMMAND.copy()
        )
        self.buffers.timestamps.append(float(sim_time))

    def num_frames(self) -> int:
        return len(self.buffers.states)

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> Path:
        """把单个工位当前缓存的数据保存成一个独立的 NPZ。"""

        # 这里故意按工位拆文件，而不是把所有 env 混在一个大文件里。
        # 原因是后续最常见的需求是：
        # - 只训练 / 回放 / 检查某一个工位
        # - 某一个工位失败时单独丢弃
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{env_name(self.env_index)}.npz"
        np.savez_compressed(
            file_path,
            **{
                "observation.images.front": np.asarray(self.buffers.front_images, dtype=np.uint8),
                "observation.images.wrist": np.asarray(self.buffers.wrist_images, dtype=np.uint8),
                "observation.state": np.asarray(self.buffers.states, dtype=np.float32),
                "command.joint_position": np.asarray(self.buffers.commands, dtype=np.float32),
                "timestamp_sec": np.asarray(self.buffers.timestamps, dtype=np.float64),
                "state_names": np.asarray(STATE_NAMES),
                "command_names": np.asarray(COMMAND_NAMES),
                "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2)),
            },
        )
        return file_path


def create_collection_cameras(selected_envs: list[int]) -> dict[int, tuple[Camera, Camera]]:
    """为要采集的工位创建 Isaac 传感器包装器。"""

    # 注意：
    # - ROS2 相机发布只需要相机 prim + OmniGraph
    # - 本地采图则需要 `Camera(...)` 传感器对象
    #
    # 所以这里的传感器只为 `collect` 模式创建，避免无谓开销。
    cameras: dict[int, tuple[Camera, Camera]] = {}
    for env_index in selected_envs:
        front_camera = Camera(
            prim_path=front_camera_path(env_index),
            name=f"{env_name(env_index)}_front_camera_sensor",
            frequency=CAMERA_FREQUENCY,
            resolution=FRONT_CAMERA_RESOLUTION,
        )
        wrist_camera = Camera(
            prim_path=wrist_camera_path(env_index),
            name=f"{env_name(env_index)}_wrist_camera_sensor",
            frequency=CAMERA_FREQUENCY,
            resolution=WRIST_CAMERA_RESOLUTION,
        )
        front_camera.initialize()
        wrist_camera.initialize()
        cameras[env_index] = (front_camera, wrist_camera)
    return cameras


class MultiEnvCollector:
    """按 single/all 选择规则记录多工位数据。"""

    def __init__(
        self,
        env_indices: list[int],
        robots: dict[int, SingleManipulator],
        output_dir: Path,
        max_steps: int,
        capture_every: int,
    ):
        # `run_dir` 以时间戳分目录，目的是让每次运行都有独立输出，
        # 避免多次采集相互覆盖。
        self.env_indices = env_indices
        self.robots = robots
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.capture_every = max(1, capture_every)
        self.recorders: dict[int, EnvRecorder] = {}
        self.saved = False
        self.run_dir = output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def attach_cameras(self, cameras: dict[int, tuple[Camera, Camera]]) -> None:
        for env_index in self.env_indices:
            front_camera, wrist_camera = cameras[env_index]
            self.recorders[env_index] = EnvRecorder(env_index, front_camera, wrist_camera)

    def maybe_record(
        self,
        frame_index: int,
        world: World,
        latest_commands: dict[int, np.ndarray],
    ) -> None:
        """按采样间隔记录一帧。

        这里的 `frame_index` 是主循环里的仿真步数，不是 episode 步数。
        `collect_every` 可以让你按更稀疏的频率采图或采状态。
        """
        if self.saved or frame_index % self.capture_every != 0:
            return
        sim_time = current_sim_time(world, frame_index)
        for env_index in self.env_indices:
            self.recorders[env_index].record(
                robot=self.robots[env_index],
                latest_command=latest_commands.get(env_index),
                sim_time=sim_time,
            )

    def is_complete(self) -> bool:
        if not self.recorders:
            return False
        return all(recorder.num_frames() >= self.max_steps for recorder in self.recorders.values())

    def save_all(self) -> list[Path]:
        """把当前所有目标工位的数据一次性写盘。"""

        saved_paths: list[Path] = []
        for env_index in self.env_indices:
            metadata = {
                "schema_version": 1,
                "env_index": env_index,
                "env_name": env_name(env_index),
                "num_envs": ARGS.num_envs,
                "collect_mode": ARGS.collect,
                "collect_steps": self.max_steps,
                "collect_every": self.capture_every,
                "front_camera_resolution": FRONT_CAMERA_RESOLUTION,
                "wrist_camera_resolution": WRIST_CAMERA_RESOLUTION,
                "task": "Record selected Franka workcells under ROS2 routing or direct simulation.",
            }
            saved_paths.append(self.recorders[env_index].save(self.run_dir, metadata))
        self.saved = True
        return saved_paths


@dataclass
class ExpertEpisodeBuffers:
    front_images: list[np.ndarray] = field(default_factory=list)
    wrist_images: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


@dataclass
class ExpertEnvState:
    env_index: int
    origin: np.ndarray
    robot: SingleManipulator
    target_cube: DynamicCuboid
    front_camera: Camera
    wrist_camera: Camera
    controller: RMPFlowController
    buffers: ExpertEpisodeBuffers = field(default_factory=ExpertEpisodeBuffers)
    current_seed: int = 0
    target_cube_local_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    place_goal_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    phase_targets: list[dict[str, np.ndarray | bool | str]] = field(default_factory=list)
    phase_index: int = 0
    phase_reached_frames: int = 0
    grasp_hold_frames: int = 0
    release_hold_frames: int = 0
    debug_step_index: int = 0
    phase_elapsed_frames: int = 0
    previous_phase_name: str = ""
    episode_step_index: int = 0
    warmup_steps_remaining: int = 0
    latest_action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    active: bool = False


class ExpertMultiEnvCollector:
    """按 17_demo 的脚本化抓取放置策略并行采集多个工位的成功轨迹。"""

    def __init__(
        self,
        env_indices: list[int],
        robots: dict[int, SingleManipulator],
        target_cubes: dict[int, DynamicCuboid],
        origins: list[np.ndarray],
        output_dir: Path,
        cameras: dict[int, tuple[Camera, Camera]],
    ):
        self.env_indices = env_indices
        self.robots = robots
        self.target_cubes = target_cubes
        self.origins = origins
        self.output_dir = output_dir
        self.run_dir = output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.success_count = 0
        self.attempt_count = 0
        self.max_attempts = max(ARGS.episodes * 12, ARGS.episodes + len(env_indices) * 4)
        self.saved_paths: list[Path] = []
        self.env_states: dict[int, ExpertEnvState] = {}

        for env_index in env_indices:
            front_camera, wrist_camera = cameras[env_index]
            self.env_states[env_index] = ExpertEnvState(
                env_index=env_index,
                origin=np.asarray(origins[env_index], dtype=np.float32),
                robot=robots[env_index],
                target_cube=target_cubes[env_index],
                front_camera=front_camera,
                wrist_camera=wrist_camera,
                controller=RMPFlowController(
                    name=f"{env_name(env_index)}_scripted_rmpflow",
                    robot_articulation=robots[env_index],
                ),
                place_goal_world=translated(origins[env_index], PLACE_GOAL_POSITION_LOCAL),
            )
            self.start_new_episode(env_index)

    def start_new_episode(self, env_index: int) -> None:
        """为某个工位启动一条新的采集尝试。"""

        state = self.env_states[env_index]
        if self.success_count >= ARGS.episodes or self.attempt_count >= self.max_attempts:
            state.active = False
            return

        seed = 20260609 + self.attempt_count
        self.attempt_count += 1
        rng = np.random.default_rng(seed)
        target_cube_local_position = sample_target_cube_local_position(rng)

        reset_robot(state.robot)
        reset_target_cube(state.target_cube, state.origin, target_cube_local_position)
        state.controller.reset()
        state.buffers = ExpertEpisodeBuffers()
        state.current_seed = seed
        state.target_cube_local_position = target_cube_local_position
        state.phase_targets = []
        state.phase_index = 0
        state.phase_reached_frames = 0
        state.grasp_hold_frames = 0
        state.release_hold_frames = 0
        state.debug_step_index = 0
        state.phase_elapsed_frames = 0
        state.previous_phase_name = ""
        state.episode_step_index = 0
        state.warmup_steps_remaining = EPISODE_SETTLE_STEPS
        state.latest_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state.active = True

        print(
            f"[{env_name(env_index)}] start_attempt={self.attempt_count:04d}"
            f" seed={seed}"
            f" target_cube_local={rounded_list(target_cube_local_position)}",
            flush=True,
        )

    def before_step(self) -> None:
        """在每个仿真 step 前，为所有活跃工位下发动作。"""

        for env_index in self.env_indices:
            state = self.env_states[env_index]
            if not state.active:
                continue

            if state.warmup_steps_remaining > 0:
                state.robot.apply_action(ArticulationAction(joint_positions=HOME_JOINT_POSITIONS.copy()))
                continue

            if not state.phase_targets:
                start_ee_position, _ = get_task_space_ee_pose(state.robot)
                cube_position, _ = state.target_cube.get_world_pose()
                state.phase_targets = build_phase_targets(
                    start_ee_position=start_ee_position,
                    target_cube_position=np.asarray(cube_position, dtype=np.float32),
                    place_goal_position=state.place_goal_world,
                )
                state.phase_index = 0

            if state.phase_index >= len(state.phase_targets):
                continue

            phase = state.phase_targets[state.phase_index]
            phase_name = str(phase["name"])
            if phase_name != state.previous_phase_name:
                state.phase_elapsed_frames = 0
                print(
                    f"[{env_name(env_index)}] enter_phase={phase_name}"
                    f" target={rounded_list(np.asarray(phase['target'], dtype=np.float32))}",
                    flush=True,
                )
                state.previous_phase_name = phase_name

            target_position = np.asarray(phase["target"], dtype=np.float32)
            gripper_closed = bool(phase["gripper_closed"])
            task_space_action = make_task_space_action(target_position, gripper_closed)
            arm_action = state.controller.forward(
                target_end_effector_position=np.asarray(task_space_action[:3], dtype=np.float32),
                target_end_effector_orientation=EE_TARGET_ORIENTATION,
            )
            gripper_action = state.robot.gripper.forward("close" if gripper_closed else "open")
            joint_action = merge_joint_actions(state.robot.num_dof, arm_action, gripper_action)
            state.robot.apply_action(joint_action)
            state.latest_action = task_space_action

    def after_step(self, frame_index: int, world: World) -> None:
        """在仿真 step 后读取观测、推进阶段，并在成功时立即保存。"""

        for env_index in self.env_indices:
            state = self.env_states[env_index]
            if not state.active:
                continue

            if state.warmup_steps_remaining > 0:
                state.warmup_steps_remaining -= 1
                continue

            if not state.phase_targets or state.phase_index >= len(state.phase_targets):
                continue

            state.debug_step_index += 1
            state.episode_step_index += 1
            state.phase_elapsed_frames += 1

            phase = state.phase_targets[state.phase_index]
            phase_name = str(phase["name"])
            target_position = np.asarray(phase["target"], dtype=np.float32)
            ee_position, _ = get_task_space_ee_pose(state.robot)
            cube_position, _ = state.target_cube.get_world_pose()
            cube_position = np.asarray(cube_position, dtype=np.float32)
            joint_positions = np.asarray(state.robot.get_joint_positions(), dtype=np.float32)
            gripper_width = float(joint_positions[7] + joint_positions[8])
            attach_offset = np.array([0.0, 0.0, -EE_PICK_ACTION_Z_OFFSET], dtype=np.float32)
            desired_cube_position = ee_position + attach_offset
            ee_reach_xy_threshold, ee_reach_z_threshold = phase_reach_thresholds(phase_name)
            ee_reached_target = is_position_close(
                ee_position,
                target_position,
                ee_reach_xy_threshold,
                ee_reach_z_threshold,
            )
            cube_following_ee = is_position_close(
                cube_position,
                desired_cube_position,
                GRASP_VERIFY_XY_THRESHOLD,
                GRASP_VERIFY_Z_THRESHOLD,
            )
            cube_lifted_from_table = float(cube_position[2] - (TABLE_SURFACE_Z + TARGET_CUBE_HALF_Z)) >= GRASP_LIFT_MIN_HEIGHT
            cube_grasped_by_physics = (
                gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD
                and cube_following_ee
                and cube_lifted_from_table
            )
            success = is_cube_inside_box(state.target_cube, state.origin)

            if state.debug_step_index % CAPTURE_EVERY_STEPS == 0:
                state.buffers.front_images.append(capture_rgb(state.front_camera))
                state.buffers.wrist_images.append(capture_rgb(state.wrist_camera))
                state.buffers.states.append(get_robot_state(state.robot))
                state.buffers.actions.append(state.latest_action.copy())
                state.buffers.rewards.append(1.0 if success else 0.0)
                state.buffers.dones.append(False)

            if phase_name in {
                "pre_grasp_settle",
                "grasp_close",
                "grasp_hold",
                "pre_release_settle",
                "release_open",
                "post_release_settle",
            } and state.debug_step_index % DEBUG_PRINT_EVERY_STEPS == 0:
                print(
                    f"[{env_name(env_index)}]"
                    f" phase={phase_name}"
                    f" ee={rounded_list(ee_position)}"
                    f" cube={rounded_list(cube_position)}"
                    f" target={rounded_list(target_position)}"
                    f" gripper_width={round(gripper_width, 4)}"
                    f" phase_elapsed={state.phase_elapsed_frames}"
                    f" phase_frames={state.phase_reached_frames}"
                    f" grasp_hold_frames={state.grasp_hold_frames}"
                    f" release_hold_frames={state.release_hold_frames}"
                    f" ee_reached={ee_reached_target}"
                    f" cube_following={cube_following_ee}"
                    f" cube_lifted={cube_lifted_from_table}"
                    f" cube_grasped={cube_grasped_by_physics}"
                    f" success={success}",
                    flush=True,
                )

            if phase_name == "grasp_close":
                if ee_reached_target:
                    if gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD:
                        state.phase_reached_frames += 1
                    else:
                        state.phase_reached_frames = 0
                    if (
                        state.phase_reached_frames >= CLOSE_GRIPPER_STABLE_FRAMES
                        or state.phase_elapsed_frames >= GRASP_CLOSE_COMMAND_STEPS
                    ):
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.phase_reached_frames = 0
            elif phase_name == "grasp_hold":
                if ee_reached_target:
                    state.grasp_hold_frames += 1
                    if state.grasp_hold_frames >= GRASP_HOLD_STEPS:
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.grasp_hold_frames = 0
            elif phase_name == "pre_grasp_settle":
                if ee_reached_target:
                    state.phase_reached_frames += 1
                    if state.phase_reached_frames >= PRE_GRASP_STABLE_FRAMES:
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.phase_reached_frames = 0
            elif phase_name == "pre_release_settle":
                if ee_reached_target:
                    state.phase_reached_frames += 1
                    if state.phase_reached_frames >= PRE_RELEASE_STABLE_FRAMES:
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.phase_reached_frames = 0
            elif phase_name == "release_open":
                if ee_reached_target:
                    if gripper_width >= GRIPPER_OPEN_WIDTH_THRESHOLD:
                        state.phase_reached_frames += 1
                    else:
                        state.phase_reached_frames = 0
                    if (
                        state.phase_reached_frames >= OPEN_GRIPPER_STABLE_FRAMES
                        or state.phase_elapsed_frames >= RELEASE_OPEN_COMMAND_STEPS
                    ):
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.phase_reached_frames = 0
            elif phase_name == "post_release_settle":
                if gripper_width >= GRIPPER_OPEN_WIDTH_THRESHOLD and success:
                    state.release_hold_frames += 1
                    if state.release_hold_frames >= RELEASE_HOLD_STEPS:
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                elif state.phase_elapsed_frames >= POST_RELEASE_WAIT_STEPS:
                    state.phase_index += 1
                    state.phase_reached_frames = 0
                else:
                    state.release_hold_frames = 0
            else:
                if ee_reached_target:
                    state.phase_reached_frames += 1
                    if state.phase_reached_frames >= PHASE_STABLE_FRAMES:
                        state.phase_index += 1
                        state.phase_reached_frames = 0
                else:
                    state.phase_reached_frames = 0

            if state.phase_index != 0 and state.phase_index >= len(state.phase_targets):
                self.finish_episode(state, success, cube_grasped_by_physics, gripper_width)
                continue

            if state.episode_step_index >= EPISODE_MAX_STEPS:
                self.finish_episode(state, success, cube_grasped_by_physics, gripper_width)

    def finish_episode(
        self,
        state: ExpertEnvState,
        success: bool,
        cube_grasped_by_physics: bool,
        gripper_width: float,
    ) -> None:
        """结束某个工位当前 episode：成功就保存，失败就丢弃，然后启动下一条。"""

        final_cube_position, _ = state.target_cube.get_world_pose()
        final_cube_position = np.asarray(final_cube_position, dtype=np.float32)
        final_ee_position, _ = get_task_space_ee_pose(state.robot)
        final_phase_name = (
            "finished"
            if state.phase_index >= len(state.phase_targets)
            else str(state.phase_targets[state.phase_index]["name"])
        )
        final_phase_target = (
            final_ee_position.copy()
            if state.phase_index >= len(state.phase_targets)
            else np.asarray(state.phase_targets[state.phase_index]["target"], dtype=np.float32)
        )

        if state.buffers.dones:
            state.buffers.dones[-1] = True

        if success and state.buffers.states:
            save_path = self.save_episode(state)
            self.saved_paths.append(save_path)
            self.success_count += 1
            print(
                f"[{env_name(state.env_index)}] success_count={self.success_count:04d}/{ARGS.episodes}"
                f" saved={save_path}",
                flush=True,
            )
        else:
            print(
                f"[{env_name(state.env_index)}] failed"
                f" phase={final_phase_name}"
                f" ee={rounded_list(final_ee_position)}"
                f" target={rounded_list(final_phase_target)}"
                f" cube={rounded_list(final_cube_position)}"
                f" cube_grasped={cube_grasped_by_physics}"
                f" gripper_width={round(gripper_width, 4)}",
                flush=True,
            )

        if self.success_count >= ARGS.episodes:
            state.active = False
            return
        self.start_new_episode(state.env_index)

    def save_episode(self, state: ExpertEnvState) -> Path:
        """把某个成功轨迹保存成和 17_demo 一致的 NPZ 格式。"""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        episode_index = self.success_count
        file_path = self.run_dir / f"episode_{episode_index:05d}_{env_name(state.env_index)}.npz"
        payload = {
            "observation.images.front": np.asarray(state.buffers.front_images, dtype=np.uint8),
            "observation.images.wrist": np.asarray(state.buffers.wrist_images, dtype=np.uint8),
            "observation.state": np.asarray(state.buffers.states, dtype=np.float32),
            "action": np.asarray(state.buffers.actions, dtype=np.float32),
            "next.reward": np.asarray(state.buffers.rewards, dtype=np.float32),
            "next.done": np.asarray(state.buffers.dones, dtype=np.bool_),
            "state_names": np.asarray(STATE_NAMES),
            "action_names": np.asarray(ACTION_NAMES),
            "task": np.asarray(TASK_DESCRIPTION),
            "success": np.asarray(True, dtype=np.bool_),
            "episode_index": np.asarray(episode_index, dtype=np.int32),
            "episode_seed": np.asarray(state.current_seed, dtype=np.int32),
            "metadata_json": np.asarray(episode_metadata(state.env_index)),
        }
        if ARGS.compress:
            np.savez_compressed(file_path, **payload)
        else:
            np.savez(file_path, **payload)
        return file_path

    def is_complete(self) -> bool:
        """判断 expert 并行采集是否已经达到停止条件。"""

        if self.success_count >= ARGS.episodes:
            return True
        if self.attempt_count < self.max_attempts:
            return False
        return not any(state.active for state in self.env_states.values())


def import_ros2_modules() -> tuple[Any, Any, Any, Any]:
    """按需导入 ROS2 Python 模块。

    只有在用户真的开启 `--ros2-arm` 时，才要求当前环境里存在 rclpy。
    这样纯本地采集或纯场景查看模式不会被 ROS2 依赖卡住。
    """

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are required for this mode. Expected modules: rclpy, sensor_msgs, std_msgs."
        ) from exc
    return rclpy, Node, JointState, Float64MultiArray


def create_multi_env_ros2_bridge(selected_envs: list[int], robots: dict[int, SingleManipulator]):
    """创建一个按 env 路由的多机械臂 ROS2 bridge。"""

    # 这里的设计重点是“同一个 Node 管多个 env”：
    # - 每个 env 各自有 command / state 话题
    # - 同时额外支持一个广播 command 话题
    #
    # 这样既能精确控制某一台机械臂，也能快速给全部工位发同一条命令。
    rclpy, Node, JointState, Float64MultiArray = import_ros2_modules()

    class MultiEnvRos2Bridge(Node):
        def __init__(self):
            super().__init__("multi_franka_ros2_bridge")
            self.robots = robots
            self.latest_commands: dict[int, np.ndarray] = {}
            self.command_subs = []
            self.state_pubs: dict[int, Any] = {}

            for env_index in selected_envs:
                # 每个工位独立订阅自己的命令话题、独立发布自己的状态话题。
                env_topic = f"/{env_name(env_index)}/franka/joint_command"
                self.command_subs.append(
                    self.create_subscription(
                        Float64MultiArray,
                        env_topic,
                        lambda msg, env_index=env_index: self.handle_command(env_index, msg.data),
                        10,
                    )
                )
                self.state_pubs[env_index] = self.create_publisher(
                    JointState,
                    f"/{env_name(env_index)}/franka/joint_state",
                    10,
                )

            self.broadcast_sub = self.create_subscription(
                Float64MultiArray,
                "/all_envs/franka/joint_command",
                self.handle_broadcast_command,
                10,
            )

        def handle_command(self, env_index: int, values: Any) -> None:
            # 每个 env 的最近命令单独缓存。
            try:
                self.latest_commands[env_index] = decode_joint_command(values, expected_len=9)
            except ValueError as exc:
                self.get_logger().error(f"{env_name(env_index)}: {exc}")

        def handle_broadcast_command(self, msg: Any) -> None:
            # 广播命令直接复制给所有已选 env。
            try:
                command = decode_joint_command(msg.data, expected_len=9)
            except ValueError as exc:
                self.get_logger().error(f"broadcast: {exc}")
                return
            for env_index in selected_envs:
                self.latest_commands[env_index] = command.copy()

        def apply_pending_actions(self) -> None:
            # 这里不在 callback 里直接控机器人，而是统一放到主循环里应用。
            # 这样 Isaac Sim 的物理 step 顺序更清楚，也更容易调试。
            for env_index in selected_envs:
                command = self.latest_commands.get(env_index)
                if command is None:
                    continue
                self.robots[env_index].apply_action(ArticulationAction(joint_positions=command))

        def publish_states(self) -> None:
            # 每一步仿真后都发布一次状态，保持和主循环节奏一致。
            stamp_msg = self.get_clock().now().to_msg()
            for env_index in selected_envs:
                robot = self.robots[env_index]
                joint_state = JointState()
                joint_state.header.stamp = stamp_msg
                joint_state.header.frame_id = franka_prim_path(env_index)
                joint_state.name = COMMAND_NAMES
                joint_state.position = [float(x) for x in np.asarray(robot.get_joint_positions(), dtype=np.float32)]
                joint_state.velocity = [float(x) for x in np.asarray(robot.get_joint_velocities(), dtype=np.float32)]
                self.state_pubs[env_index].publish(joint_state)

    rclpy.init()
    return rclpy, MultiEnvRos2Bridge()


def print_selection_summary(
    arm_envs: list[int],
    camera_envs: list[int],
    collect_envs: list[int],
) -> None:
    """在终端打印这次运行实际启用了哪些路由。"""

    print(f"Built {ARGS.num_envs} Franka workcells.", flush=True)

    if arm_envs:
        print("ROS2 arm topics:", flush=True)
        for env_index in arm_envs:
            print(f"  /{env_name(env_index)}/franka/joint_command", flush=True)
            print(f"  /{env_name(env_index)}/franka/joint_state", flush=True)
        print("  /all_envs/franka/joint_command", flush=True)
    else:
        print("ROS2 arm bridge: off", flush=True)

    if camera_envs:
        print("ROS2 camera topics:", flush=True)
        for env_index in camera_envs:
            if ARGS.ros2_camera_name in {"front", "both"}:
                print(f"  /{env_name(env_index)}/front_camera/rgb", flush=True)
            if ARGS.ros2_camera_name in {"wrist", "both"}:
                print(f"  /{env_name(env_index)}/wrist_camera/rgb", flush=True)
    else:
        print("ROS2 camera bridge: off", flush=True)

    if collect_envs:
        print(f"Collector target envs: {[env_name(index) for index in collect_envs]}", flush=True)
        print(f"Collector policy: {ARGS.collect_policy}", flush=True)
        if ARGS.collect_policy == "expert":
            print(f"Target successful episodes: {ARGS.episodes}", flush=True)
            print(f"Capture every steps: {CAPTURE_EVERY_STEPS}", flush=True)
            print(f"Save compressed: {ARGS.compress}", flush=True)
        else:
            print(f"Collect steps per env: {ARGS.collect_steps}", flush=True)
            print(f"Collect every steps: {ARGS.collect_every}", flush=True)
        print(f"Collector output dir: {ARGS.output_dir}", flush=True)
    else:
        print("Collector: off", flush=True)


def main() -> None:
    try:
        # 第一步先把三类选择都转成最终 env 下标列表。
        # 到这一步之后，后面逻辑就都不再关心用户原始命令行传的是 single 还是 all。
        arm_envs = selected_env_indices(ARGS.ros2_arm, ARGS.ros2_arm_env, ARGS.num_envs)
        camera_envs = selected_env_indices(ARGS.ros2_camera, ARGS.ros2_camera_env, ARGS.num_envs)
        collect_envs = selected_env_indices(ARGS.collect, ARGS.collect_env, ARGS.num_envs)

        # 先搭完整场景，再决定哪些工位需要开放给 ROS2 或采集器。
        world, robots, target_cubes, origins = build_scene(ARGS.num_envs)
        for robot in robots.values():
            robot.initialize()

        # 如果用户打开了 ROS2 相机发布，这里只给被选中的工位建 graph。
        if camera_envs:
            enable_selected_camera_publishers(camera_envs, ARGS.ros2_camera_name)

        collector: MultiEnvCollector | None = None
        expert_collector: ExpertMultiEnvCollector | None = None
        if collect_envs and ARGS.collect_policy == "expert":
            if arm_envs:
                raise ValueError("expert collect mode does not support --ros2-arm at the same time.")
            expert_collector = ExpertMultiEnvCollector(
                env_indices=collect_envs,
                robots=robots,
                target_cubes=target_cubes,
                origins=origins,
                output_dir=ARGS.output_dir.resolve(),
                cameras=create_collection_cameras(collect_envs),
            )
        elif collect_envs:
            # 本地采集器只为被选中的工位创建传感器和缓存。
            collector = MultiEnvCollector(
                env_indices=collect_envs,
                robots=robots,
                output_dir=ARGS.output_dir.resolve(),
                max_steps=ARGS.collect_steps,
                capture_every=ARGS.collect_every,
            )
            collector.attach_cameras(create_collection_cameras(collect_envs))

        rclpy = None
        ros2_bridge = None
        if arm_envs:
            # 机械臂 ROS2 bridge 同样只绑定被选中的工位。
            rclpy, ros2_bridge = create_multi_env_ros2_bridge(arm_envs, robots)

        print_selection_summary(arm_envs=arm_envs, camera_envs=camera_envs, collect_envs=collect_envs)

        if ARGS.headless and not arm_envs and not camera_envs and not collect_envs:
            return

        world.play()
        if collect_envs:
            for _ in range(CAMERA_WARMUP_STEPS):
                world.step(render=True)
        frame_index = 0

        while simulation_app.is_running():
            # 主循环顺序刻意固定成：
            # 1. 先收 ROS2 命令
            # 2. 把命令应用到机器人
            # 3. 推进一步仿真
            # 4. 记录本地数据
            # 5. 向 ROS2 发布最新状态
            #
            # 这个顺序最容易理解，也方便后续做时序对齐。
            if ros2_bridge is not None and rclpy is not None:
                rclpy.spin_once(ros2_bridge, timeout_sec=0.0)
                ros2_bridge.apply_pending_actions()

            if expert_collector is not None:
                expert_collector.before_step()

            world.step(render=True)
            frame_index += 1

            latest_commands = ros2_bridge.latest_commands if ros2_bridge is not None else {}
            if expert_collector is not None:
                expert_collector.after_step(frame_index, world)
                if expert_collector.is_complete():
                    print(
                        f"Expert collection finished:"
                        f" success={expert_collector.success_count}/{ARGS.episodes}"
                        f" attempts={expert_collector.attempt_count}",
                        flush=True,
                    )
                    for path in expert_collector.saved_paths:
                        print(f"  {path}", flush=True)
                    break
            elif collector is not None:
                collector.maybe_record(frame_index, world, latest_commands)
                if collector.is_complete() and not collector.saved:
                    # headless 模式下，采满目标帧数后直接保存并退出。
                    saved_paths = collector.save_all()
                    print("Saved recordings:", flush=True)
                    for path in saved_paths:
                        print(f"  {path}", flush=True)
                    if ARGS.headless:
                        break

            if ros2_bridge is not None:
                ros2_bridge.publish_states()

        if ros2_bridge is not None:
            ros2_bridge.destroy_node()
        if rclpy is not None:
            rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # 无论正常退出还是异常退出，都显式关闭 SimulationApp。
        simulation_app.close()


if __name__ == "__main__":
    main()
