"""Franka SmolVLA 数据采集示例。

这个脚本会在 Isaac Sim 中创建一个 Franka 桌面抓取场景，并用内置的
PickPlaceController 作为“专家策略”自动采集示教数据。

采集内容包括：
1. 前视相机图像
2. 手腕相机图像
3. 机器人状态（7 关节、末端位置、末端姿态、夹爪开合宽度）
4. 专家动作（末端目标位置、夹爪开合真值）

数据会按 episode 保存为压缩 `.npz` 文件，字段命名尽量贴近后续
SmolVLA / LeRobot 常用格式，便于后续转换。

运行示例：

    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/demo.py
    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/demo.py --episodes 50 --headless
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析少量必要参数，避免把脚本做成参数堆。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20, help="采集多少个 episode。")
    parser.add_argument("--headless", action="store_true", help="无界面运行。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "raw",
        help="数据输出目录。",
    )
    return parser.parse_args()


ARGS = parse_args()

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
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
from pxr import Gf, UsdGeom, UsdLux


TABLE_H = 0.40
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H

FRANKA_PRIM_PATH = "/World/Franka"
FRONT_CAMERA_PATH = "/World/front_camera"
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"

FRONT_CAMERA_EYE = np.array([1.15, -1.10, 1.10], dtype=np.float32)
FRONT_CAMERA_TARGET = np.array([0.46, 0.00, 0.55], dtype=np.float32)

FRONT_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 20

CUBE_SIZE = np.array([0.045, 0.045, 0.045], dtype=np.float32)
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)

TARGET_CUBE_NAME = "cube_red"
TARGET_CUBE_COLOR = np.array([0.88, 0.15, 0.15], dtype=np.float32)
DISTRACTOR_CUBE_SPECS = [
    ("cube_green", np.array([0.18, 0.66, 0.24], dtype=np.float32)),
    ("cube_blue", np.array([0.12, 0.36, 0.86], dtype=np.float32)),
]

# 红色目标方块压在最稳的抓取区，优先保证成功率。
TARGET_CUBE_X_RANGE = (0.42, 0.44)
TARGET_CUBE_Y_RANGE = (-0.02, 0.02)

# 两个干扰方块固定在左右两侧，仅做视觉干扰，不进入主要抓取路径。
DISTRACTOR_CUBE_LAYOUT = {
    "cube_green": np.array([0.34, 0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
    "cube_blue": np.array([0.34, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
}
DISTRACTOR_CUBE_JITTER_X = 0.015
DISTRACTOR_CUBE_JITTER_Y = 0.020

PLACE_BOX_CENTER = np.array([0.64, 0.18], dtype=np.float32)
PLACE_BOX_OUTER_X = 0.18
PLACE_BOX_OUTER_Y = 0.18
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.016
PLACE_BOX_WALL_H = 0.10
PLACE_GOAL_POSITION = np.array(
    [PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], TABLE_SURFACE_Z + CUBE_HALF_Z],
    dtype=np.float32,
)

# 这个 home 姿态用来确保每个 episode 开始时机器人状态一致。
HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)
EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))

EE_PICK_Z_OFFSET = 0.092
EE_PLACE_Z_OFFSET = 0.110
EE_HOVER_MARGIN = 0.140

# `controller.forward()` 使用的任务空间控制点，与 `franka.end_effector.get_world_pose()`
# 读取到的 `panda_hand` 反馈点之间，Z 方向存在稳定的固定偏差。
# 如果不做这一步修正，状态机会一直误判“末端还没到位”，从而卡死在 hover 阶段。
EE_FEEDBACK_Z_BIAS = 0.0985

GRASP_HOLD_STEPS = 26
RELEASE_HOLD_STEPS = 26
PHASE_STABLE_FRAMES = 2
EE_GENERAL_REACH_XY_THRESHOLD = 0.030
EE_GENERAL_REACH_Z_THRESHOLD = 0.030
EE_PICK_REACH_XY_THRESHOLD = 0.015
EE_PICK_REACH_Z_THRESHOLD = 0.020
EE_PLACE_REACH_XY_THRESHOLD = 0.040
EE_PLACE_REACH_Z_THRESHOLD = 0.150
GRIPPER_CLOSE_WIDTH_THRESHOLD = 0.065
PICK_ATTACH_XY_THRESHOLD = 0.015
PICK_ATTACH_Z_THRESHOLD = 0.020
PLACE_RELEASE_XY_THRESHOLD = 0.040
PLACE_RELEASE_Z_THRESHOLD = 0.150
ATTACHED_CUBE_BLEND = 0.35
PLACE_SETTLE_STEPS = 20

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

ACTION_NAMES = [
    "target_ee_pos_x",
    "target_ee_pos_y",
    "target_ee_pos_z",
    "target_gripper_closed",
]

TASK_DESCRIPTION = "Pick up the red cube with Franka and place it into the wooden tray."


def create_camera_prim(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """在 USD 场景里创建相机 prim。"""

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    """创建基础光照，让两路图像的视觉质量更稳定。"""

    stage = get_current_stage()

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1200.0)

    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateWidthAttr(1.6)
    key.CreateHeightAttr(1.2)

    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.65, -0.20, 1.80))
    xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World) -> None:
    """添加简单房间，避免采图背景过空。"""

    world.scene.add(
        FixedCuboid(
            name="room_floor",
            prim_path="/World/Room/Floor",
            position=np.array([0.55, 0.0, -0.025], dtype=np.float32),
            scale=np.array([3.4, 3.0, 0.05], dtype=np.float32),
            size=1.0,
            color=np.array([0.34, 0.35, 0.36], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_back_wall",
            prim_path="/World/Room/BackWall",
            position=np.array([0.55, 1.50, 1.20], dtype=np.float32),
            scale=np.array([3.4, 0.04, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_left_wall",
            prim_path="/World/Room/LeftWall",
            position=np.array([-1.15, 0.0, 1.20], dtype=np.float32),
            scale=np.array([0.04, 3.0, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_right_wall",
            prim_path="/World/Room/RightWall",
            position=np.array([2.25, 0.0, 1.20], dtype=np.float32),
            scale=np.array([0.04, 3.0, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )


def add_table(world: World) -> None:
    """添加工作台。"""

    world.scene.add(
        FixedCuboid(
            name="table",
            prim_path="/World/Table",
            position=TABLE_CENTER,
            scale=TABLE_SIZE,
            size=1.0,
            color=np.array([0.55, 0.35, 0.15], dtype=np.float32),
        )
    )


def add_place_box(world: World) -> None:
    """创建放置托盘。"""

    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)

    world.scene.add(
        FixedCuboid(
            name="place_box_bottom",
            prim_path="/World/PlaceBox/Bottom",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], bottom_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_left",
            prim_path="/World/PlaceBox/WallLeft",
            position=np.array([PLACE_BOX_CENTER[0] - PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_right",
            prim_path="/World/PlaceBox/WallRight",
            position=np.array([PLACE_BOX_CENTER[0] + PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_front",
            prim_path="/World/PlaceBox/WallFront",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_back",
            prim_path="/World/PlaceBox/WallBack",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )


def add_franka(world: World) -> SingleManipulator:
    """把 Franka 以可控制机械臂的形式加入场景。"""

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H], dtype=np.float32),
        )
    )
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_training_cubes(world: World) -> dict[str, DynamicCuboid]:
    """添加一个可抓取的红色目标方块和两个固定干扰方块。"""

    target_cube = world.scene.add(
        DynamicCuboid(
            name=TARGET_CUBE_NAME,
            prim_path=f"/World/{TARGET_CUBE_NAME}",
            position=np.array([0.42, 0.00, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
            scale=CUBE_SIZE,
            size=1.0,
            color=TARGET_CUBE_COLOR,
        )
    )

    for cube_name, cube_color in DISTRACTOR_CUBE_SPECS:
        world.scene.add(
            FixedCuboid(
                name=cube_name,
                prim_path=f"/World/{cube_name}",
                position=DISTRACTOR_CUBE_LAYOUT[cube_name],
                scale=CUBE_SIZE,
                size=1.0,
                color=cube_color,
            )
        )

    return {TARGET_CUBE_NAME: target_cube}


def build_scene() -> tuple[World, SingleManipulator, dict[str, DynamicCuboid]]:
    """构建完整场景。"""

    world = World(stage_units_in_meters=1.0)
    create_lights()
    add_room(world)
    world.scene.add_default_ground_plane()
    add_table(world)
    add_place_box(world)

    franka = add_franka(world)
    cubes = add_training_cubes(world)

    create_camera_prim(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-35.0, 0.0, 45.0),
        focal_length=10.0,
    )
    create_camera_prim(
        path=WRIST_CAMERA_PATH,
        position=(0.06, 0.0, 0.03),
        rotation_xyz_deg=(-95.0, 0.0, -90.0),
        focal_length=4.0,
    )

    world.reset()

    set_camera_view(
        eye=FRONT_CAMERA_EYE,
        target=FRONT_CAMERA_TARGET,
        camera_prim_path=FRONT_CAMERA_PATH,
    )
    if not ARGS.headless:
        set_camera_view(
            eye=FRONT_CAMERA_EYE,
            target=FRONT_CAMERA_TARGET,
            camera_prim_path="/OmniverseKit_Persp",
        )

    return world, franka, cubes


def create_cameras() -> tuple[Camera, Camera]:
    """用传感器接口包装前视相机与手腕相机。"""

    front_camera = Camera(
        prim_path=FRONT_CAMERA_PATH,
        name="front_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=FRONT_CAMERA_RESOLUTION,
    )
    wrist_camera = Camera(
        prim_path=WRIST_CAMERA_PATH,
        name="wrist_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=WRIST_CAMERA_RESOLUTION,
    )
    front_camera.initialize()
    wrist_camera.initialize()
    return front_camera, wrist_camera


def sample_cube_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """采样三色方块的位置。

    红色方块是唯一抓取目标，因此放在最稳的抓取区。
    绿色和蓝色方块是固定干扰物，不参与采样。
    """

    return {
        TARGET_CUBE_NAME: np.array(
            [
                rng.uniform(*TARGET_CUBE_X_RANGE),
                rng.uniform(*TARGET_CUBE_Y_RANGE),
                TABLE_SURFACE_Z + CUBE_HALF_Z,
            ],
            dtype=np.float32,
        )
    }


def reset_robot(franka: SingleManipulator) -> None:
    """把机器人直接复位到一个统一 home 姿态。"""

    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_cubes(cubes: dict[str, DynamicCuboid], cube_positions: dict[str, np.ndarray]) -> None:
    """把三色方块统一复位到新位置。"""

    for cube_name, cube in cubes.items():
        cube_position = cube_positions[cube_name]
        cube.set_world_pose(
            position=cube_position,
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))


def settle_scene(world: World, steps: int) -> None:
    """给物理系统一点稳定时间。"""

    for _ in range(steps):
        world.step(render=True)


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
) -> list[dict[str, np.ndarray | bool | str]]:
    """生成按阶段推进的任务目标。

    这里不再预先展开成固定帧数的整条轨迹，而是只给出每个阶段的目标位姿。
    真正何时进入下一阶段，由机器人是否“真的走到位”来决定。
    这样可以避免：
    1. 机械臂还没到抓取位就提前吸附方块
    2. 机械臂还没到托盘上方就提前放置方块
    """

    pick_position = np.array(
        [
            float(target_cube_position[0]),
            float(target_cube_position[1]),
            float(target_cube_position[2] + EE_PICK_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    pick_hover_position = pick_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    place_position = np.array(
        [
            float(PLACE_GOAL_POSITION[0]),
            float(PLACE_GOAL_POSITION[1]),
            float(PLACE_GOAL_POSITION[2] + EE_PLACE_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    place_hover_position = place_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    current_position = np.asarray(start_ee_position, dtype=np.float32)
    return [
        {"name": "approach_pick_hover", "target": pick_hover_position, "gripper_closed": False},
        {"name": "descend_pick", "target": pick_position, "gripper_closed": False},
        {"name": "grasp", "target": pick_position, "gripper_closed": True},
        {"name": "lift", "target": pick_hover_position, "gripper_closed": True},
        {"name": "transfer", "target": place_hover_position, "gripper_closed": True},
        {"name": "descend_place", "target": place_position, "gripper_closed": True},
        {"name": "release", "target": place_position, "gripper_closed": False},
        {"name": "retreat", "target": place_hover_position, "gripper_closed": False},
        {"name": "return_idle", "target": current_position, "gripper_closed": False},
    ]


def make_task_space_action(target_position: np.ndarray, gripper_closed: bool) -> np.ndarray:
    """把当前阶段目标编码成一条训练动作。"""

    return np.concatenate(
        [
            np.asarray(target_position, dtype=np.float32),
            np.array([1.0 if gripper_closed else 0.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def position_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个三维点之间的欧氏距离。"""

    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算 XY 平面距离。

    抓取/放置质量更多取决于水平面对齐，因此这里把 XY 单独拆出来，
    避免只用三维欧氏距离时掩盖“还没对准就吸附/释放”的问题。
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a[:2] - b[:2]))


def vertical_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Z 方向绝对距离。"""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(abs(a[2] - b[2]))


def is_position_close(
    current_position: np.ndarray,
    target_position: np.ndarray,
    xy_threshold: float,
    z_threshold: float,
) -> bool:
    """按 XY 和 Z 分开阈值判断是否真正到位。"""

    return bool(
        planar_distance(current_position, target_position) <= xy_threshold
        and vertical_distance(current_position, target_position) <= z_threshold
    )


def phase_reach_thresholds(phase_name: str) -> tuple[float, float]:
    """为不同阶段返回不同的到位阈值。

    原因：
    1. 过渡阶段可以略松，保证 RMPFlow 容易推进
    2. 抓取下降和放置下降必须更严格，否则会在还没真正靠近物体时提前切阶段
    """

    if phase_name in {"descend_pick", "grasp"}:
        return EE_PICK_REACH_XY_THRESHOLD, EE_PICK_REACH_Z_THRESHOLD
    if phase_name in {"descend_place", "release"}:
        return EE_PLACE_REACH_XY_THRESHOLD, EE_PLACE_REACH_Z_THRESHOLD
    return EE_GENERAL_REACH_XY_THRESHOLD, EE_GENERAL_REACH_Z_THRESHOLD


def smoothstep(alpha: float) -> float:
    """把 0~1 的线性插值变成更平滑的过渡曲线。"""

    clipped = float(np.clip(alpha, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def capture_rgb(camera: Camera) -> np.ndarray:
    """读取一帧 RGB 图像，并统一成 uint8。"""

    rgb = camera.get_rgb()
    if rgb is None:
        raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")
    return np.asarray(rgb, dtype=np.uint8)


def get_task_space_ee_pose(franka: SingleManipulator) -> tuple[np.ndarray, np.ndarray]:
    """读取与 RMPFlow 目标位置同一参考系下的末端位姿。

    `franka.end_effector` 对应的 `panda_hand` 反馈点在 Z 方向上比控制点更高，
    因此这里对位置做固定偏差修正，让：
    1. 阶段推进逻辑
    2. 抓取/放置几何判断
    3. 保存到数据集里的末端位置
    都和动作真值使用同一套任务空间参考系。
    """

    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32).copy()
    ee_position[2] -= EE_FEEDBACK_Z_BIAS
    ee_orientation = np.asarray(ee_orientation, dtype=np.float32)
    return ee_position, ee_orientation


def get_robot_state(franka: SingleManipulator) -> np.ndarray:
    """读取训练时常用的状态向量。"""

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


def is_cube_inside_box(cube: DynamicCuboid) -> bool:
    """判断指定方块是否已经被放进托盘。"""

    cube_position, _ = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)

    inner_half_x = PLACE_BOX_OUTER_X / 2.0 - PLACE_BOX_WALL_T
    inner_half_y = PLACE_BOX_OUTER_Y / 2.0 - PLACE_BOX_WALL_T
    within_x = abs(float(cube_position[0] - PLACE_BOX_CENTER[0])) < inner_half_x
    within_y = abs(float(cube_position[1] - PLACE_BOX_CENTER[1])) < inner_half_y
    within_z = abs(float(cube_position[2] - PLACE_GOAL_POSITION[2])) < 0.035
    return bool(within_x and within_y and within_z)


def episode_metadata() -> str:
    """生成每个 episode 共享的元数据字符串。"""

    return json.dumps(
        {
            "schema_version": 1,
            "task": TASK_DESCRIPTION,
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
            "front_camera_resolution": FRONT_CAMERA_RESOLUTION,
            "wrist_camera_resolution": WRIST_CAMERA_RESOLUTION,
            "episode_max_steps": EPISODE_MAX_STEPS,
        },
        ensure_ascii=False,
        indent=2,
    )


def save_episode(
    output_dir: Path,
    episode_index: int,
    front_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    success: bool,
    seed: int,
) -> Path:
    """把单个 episode 保存成压缩 npz。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"episode_{episode_index:05d}.npz"

    np.savez_compressed(
        file_path,
        **{
            "observation.images.front": np.asarray(front_images, dtype=np.uint8),
            "observation.images.wrist": np.asarray(wrist_images, dtype=np.uint8),
            "observation.state": np.asarray(states, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.float32),
            "next.reward": np.asarray(rewards, dtype=np.float32),
            "next.done": np.asarray(dones, dtype=np.bool_),
            "state_names": np.asarray(STATE_NAMES),
            "action_names": np.asarray(ACTION_NAMES),
            "task": np.asarray(TASK_DESCRIPTION),
            "success": np.asarray(success, dtype=np.bool_),
            "episode_index": np.asarray(episode_index, dtype=np.int32),
            "episode_seed": np.asarray(seed, dtype=np.int32),
            "metadata_json": np.asarray(episode_metadata()),
        },
    )
    return file_path


def collect_episode(
    world: World,
    franka: SingleManipulator,
    cubes: dict[str, DynamicCuboid],
    front_camera: Camera,
    wrist_camera: Camera,
    controller: RMPFlowController,
    episode_index: int,
    output_dir: Path,
    seed: int,
) -> tuple[bool, Path | None]:
    """采集单个 episode。"""

    rng = np.random.default_rng(seed)
    cube_positions = sample_cube_positions(rng)
    target_cube = cubes[TARGET_CUBE_NAME]
    reset_robot(franka)
    reset_cubes(cubes, cube_positions)
    controller.reset()
    settle_scene(world, EPISODE_SETTLE_STEPS)

    front_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    done = False
    success = False
    cube_attached = False
    start_ee_position, _ = get_task_space_ee_pose(franka)
    phase_targets = build_phase_targets(
        start_ee_position=start_ee_position,
        target_cube_position=cube_positions[TARGET_CUBE_NAME],
    )
    phase_index = 0
    phase_reached_frames = 0
    grasp_hold_frames = 0
    release_hold_frames = 0
    place_settle_step = 0
    attach_offset = np.array([0.0, 0.0, -EE_PICK_Z_OFFSET], dtype=np.float32)
    release_start_position = PLACE_GOAL_POSITION.copy()

    for _ in range(EPISODE_MAX_STEPS):
        if phase_index >= len(phase_targets):
            break

        phase = phase_targets[phase_index]
        phase_name = str(phase["name"])
        target_position = np.asarray(phase["target"], dtype=np.float32)
        gripper_closed = bool(phase["gripper_closed"])
        task_space_action = make_task_space_action(target_position, gripper_closed)
        target_position = np.asarray(task_space_action[:3], dtype=np.float32)

        arm_action = controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=EE_TARGET_ORIENTATION,
        )
        gripper_action = franka.gripper.forward("close" if gripper_closed else "open")
        joint_action = merge_joint_actions(franka.num_dof, arm_action, gripper_action)
        franka.apply_action(joint_action)
        world.step(render=True)

        ee_position, _ = get_task_space_ee_pose(franka)
        cube_position, _ = target_cube.get_world_pose()
        cube_position = np.asarray(cube_position, dtype=np.float32)
        joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        gripper_width = float(joint_positions[7] + joint_positions[8])
        desired_cube_position = ee_position + attach_offset
        ee_reach_xy_threshold, ee_reach_z_threshold = phase_reach_thresholds(phase_name)
        ee_reached_target = is_position_close(
            ee_position,
            target_position,
            ee_reach_xy_threshold,
            ee_reach_z_threshold,
        )
        cube_ready_to_attach = is_position_close(
            cube_position,
            desired_cube_position,
            PICK_ATTACH_XY_THRESHOLD,
            PICK_ATTACH_Z_THRESHOLD,
        )
        cube_ready_to_release = is_position_close(
            desired_cube_position,
            PLACE_GOAL_POSITION,
            PLACE_RELEASE_XY_THRESHOLD,
            PLACE_RELEASE_Z_THRESHOLD,
        )

        if cube_attached and phase_name != "release":
            smoothed_cube_position = (
                (1.0 - ATTACHED_CUBE_BLEND) * cube_position + ATTACHED_CUBE_BLEND * desired_cube_position
            ).astype(np.float32)
            target_cube.set_world_pose(
                position=smoothed_cube_position,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
            target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        elif not cube_attached and phase_name == "grasp":
            if (
                ee_reached_target
                and cube_ready_to_attach
                and gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD
            ):
                cube_attached = True
                grasp_hold_frames = 0
                target_cube.set_world_pose(
                    position=desired_cube_position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        elif cube_attached and phase_name == "release":
            carried_cube_position = (
                (1.0 - ATTACHED_CUBE_BLEND) * cube_position + ATTACHED_CUBE_BLEND * desired_cube_position
            ).astype(np.float32)
            if ee_reached_target and cube_ready_to_release:
                if place_settle_step == 0:
                    release_start_position = carried_cube_position.copy()
                place_settle_step += 1
                settle_alpha = smoothstep(place_settle_step / float(PLACE_SETTLE_STEPS))
                settled_position = (
                    (1.0 - settle_alpha) * release_start_position + settle_alpha * PLACE_GOAL_POSITION
                ).astype(np.float32)
                target_cube.set_world_pose(
                    position=settled_position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
                if place_settle_step >= PLACE_SETTLE_STEPS:
                    cube_attached = False
            else:
                place_settle_step = 0
                target_cube.set_world_pose(
                    position=carried_cube_position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))

        front_images.append(capture_rgb(front_camera))
        wrist_images.append(capture_rgb(wrist_camera))
        states.append(get_robot_state(franka))
        actions.append(task_space_action)

        success = is_cube_inside_box(target_cube)
        rewards.append(1.0 if success else 0.0)
        dones.append(False)

        if phase_name == "grasp":
            if cube_attached:
                grasp_hold_frames += 1
                if grasp_hold_frames >= GRASP_HOLD_STEPS:
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                grasp_hold_frames = 0
        elif phase_name == "release":
            if not cube_attached and success:
                release_hold_frames += 1
                if release_hold_frames >= RELEASE_HOLD_STEPS:
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                release_hold_frames = 0
        else:
            if ee_reached_target:
                phase_reached_frames += 1
                if phase_reached_frames >= PHASE_STABLE_FRAMES:
                    phase_index += 1
                    phase_reached_frames = 0
                    if phase_name != "release":
                        place_settle_step = 0
            else:
                phase_reached_frames = 0

        if not simulation_app.is_running():
            break

    if dones:
        dones[-1] = True
        done = True

    final_cube_position, _ = target_cube.get_world_pose()
    final_cube_position = np.asarray(final_cube_position, dtype=np.float32)
    final_ee_position, _ = get_task_space_ee_pose(franka)
    final_phase_name = "finished" if phase_index >= len(phase_targets) else str(phase_targets[phase_index]["name"])
    final_phase_target = (
        final_ee_position.copy()
        if phase_index >= len(phase_targets)
        else np.asarray(phase_targets[phase_index]["target"], dtype=np.float32)
    )

    if not success:
        print(
            f"  red cube final position: {np.round(final_cube_position, 4).tolist()}",
            flush=True,
        )
        print(
            "  debug:"
            f" phase={final_phase_name}"
            f" ee={np.round(final_ee_position, 4).tolist()}"
            f" target={np.round(final_phase_target, 4).tolist()}"
            f" cube_attached={cube_attached}"
            f" gripper_width={round(gripper_width, 4)}",
            flush=True,
        )
        return False, None

    save_path = save_episode(
        output_dir=output_dir,
        episode_index=episode_index,
        front_images=front_images,
        wrist_images=wrist_images,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        success=success,
        seed=seed,
    )
    print(
        f"  red cube final position: {np.round(final_cube_position, 4).tolist()}",
        flush=True,
    )
    return success, save_path


def main() -> None:
    """脚本主入口。"""

    output_dir = ARGS.output_dir.resolve()

    world, franka, cubes = build_scene()
    franka.initialize()
    front_camera, wrist_camera = create_cameras()
    settle_scene(world, 10)

    controller = RMPFlowController(
        name="franka_scripted_rmpflow",
        robot_articulation=franka,
    )

    success_count = 0
    attempt_count = 0
    max_attempts = max(ARGS.episodes * 12, ARGS.episodes + 5)

    print(f"输出目录: {output_dir}", flush=True)
    print(f"目标成功 episode 数: {ARGS.episodes}", flush=True)
    print("开始采集，仅保存成功轨迹...", flush=True)

    while success_count < ARGS.episodes and attempt_count < max_attempts:
        episode_seed = 20260604 + attempt_count
        success, save_path = collect_episode(
            world=world,
            franka=franka,
            cubes=cubes,
            front_camera=front_camera,
            wrist_camera=wrist_camera,
            controller=controller,
            episode_index=success_count,
            output_dir=output_dir,
            seed=episode_seed,
        )
        if success:
            print(
                f"[attempt {attempt_count:03d}] success "
                f"saved -> {save_path}",
                flush=True,
            )
            success_count += 1
        else:
            print(f"[attempt {attempt_count:03d}] failed discarded", flush=True)
        attempt_count += 1

        if not simulation_app.is_running():
            print("检测到 Isaac Sim 已关闭，提前结束采集。", flush=True)
            break

    print(f"采集完成：成功保存 {success_count} / 目标 {ARGS.episodes}，总尝试 {attempt_count}。", flush=True)
    if success_count < ARGS.episodes:
        print("警告：未达到目标成功条数，请再次运行采集。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
