"""Franka SmolVLA 数据采集示例。

这个脚本的目标，是在 Isaac Sim 里稳定地产生一批“可以直接拿去训练 /
转换”的 Franka 抓取放置示教数据。

它整体上做的事情可以概括成 4 步：

1. 先搭一个固定、可复现的 Franka 桌面抓取场景
2. 用脚本化专家策略自动完成抓取和放置
3. 在执行过程中逐帧记录图像、状态、动作、奖励、done
4. 只把成功轨迹保存成按 episode 划分的 `.npz`

这里的“专家策略”不是键盘遥操作，也不是策略网络，而是：

- 先手工定义每个阶段要去的任务空间目标
- 再由 `RMPFlowController` 把这些末端目标位置转换成机械臂关节动作
- 夹爪则按开 / 关二值逻辑控制

这样设计的好处是：

- 采集行为稳定，适合做第一版数据源
- 动作语义清晰，后续更容易转成 SmolVLA / LeRobot 常见格式
- 一旦发现某个阶段不稳，可以直接调阶段目标和阈值，不需要先改模型

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
    """解析命令行参数。

    这里只保留最常需要改的三个参数：

    - `--episodes`
      目标是成功保存多少条 episode
    - `--headless`
      是否在无界面模式下运行
    - `--output-dir`
      原始 `.npz` 输出目录

    其余采集行为，例如场景尺寸、夹爪阈值、阶段机节奏，都放在文件常量里，
    这样示例更容易整体阅读。
    """

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


# 先解析参数，再创建 SimulationApp。
#
# 这是 Isaac Sim 脚本里很常见的入口顺序：
# 1. 先拿到 headless 等启动选项
# 2. 再创建 SimulationApp
# 3. 然后才去导入大量 Isaac / Omniverse 模块
ARGS = parse_args()

# `SimulationApp` 是整个 Isaac Sim standalone 脚本的启动核心。
# 很多 Isaac 模块都依赖它先被创建，否则会在导入或扩展初始化阶段出错。
simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)


# 从这里开始才导入 Isaac Sim 相关模块。
#
# 这不是随意的代码风格，而是 Isaac Sim 的真实约束：
# 许多扩展必须在 `SimulationApp` 已经存在之后再导入。
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


# ----------------------------
# 场景几何与采集配置
# ----------------------------
# 这一大段常量决定了：
# - 桌子和托盘的大小
# - Franka、相机、方块放在哪
# - 专家阶段推进有多快
# - 抓取 / 放置被判定为“到位”的阈值有多严格
#
# 把它们集中放在顶部的好处是，后续调参时不用在函数体里到处翻。

# 桌子总高度（米）。
TABLE_H = 0.40
# 桌子中心点。因为立方体按中心放置，所以 z 取高度一半。
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
# 桌子的长宽高。
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
# 桌面平面所在的 z 值，后续放方块时会反复用到。
TABLE_SURFACE_Z = TABLE_H

# Franka 在当前 USD 场景中的 prim 路径。
FRANKA_PRIM_PATH = "/World/Franka"
# 前视相机路径。
FRONT_CAMERA_PATH = "/World/front_camera"
# 手腕相机路径。它挂在 `panda_hand` 下，属于机器人结构的一部分。
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"

# 前视相机的观察位置。
FRONT_CAMERA_EYE = np.array([1.15, -1.10, 1.10], dtype=np.float32)
# 前视相机的目标注视点。
FRONT_CAMERA_TARGET = np.array([0.46, 0.00, 0.55], dtype=np.float32)

# 两路相机分辨率。
FRONT_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (640, 480)
# 相机采样频率。
CAMERA_FREQUENCY = 20
CAMERA_WARMUP_STEPS = 20
CAMERA_CAPTURE_RETRIES = 6

# 训练方块尺寸。
CUBE_SIZE = np.array([0.045, 0.045, 0.045], dtype=np.float32)
# 方块半高。因为方块中心点会放在 `桌面高度 + 半高` 处。
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)

# 目标红色方块名称。
TARGET_CUBE_NAME = "cube_red"
# 目标红色方块颜色。
TARGET_CUBE_COLOR = np.array([0.88, 0.15, 0.15], dtype=np.float32)
# 两个干扰方块的名字和颜色。
DISTRACTOR_CUBE_SPECS = [
    ("cube_green", np.array([0.18, 0.66, 0.24], dtype=np.float32)),
    ("cube_blue", np.array([0.12, 0.36, 0.86], dtype=np.float32)),
]

# 红色目标方块被限制在一个很小的稳定区域内采样。
# 这是“先保证专家成功率，再逐步扩大随机化”的思路。
TARGET_CUBE_X_RANGE = (0.42, 0.44)
TARGET_CUBE_Y_RANGE = (-0.02, 0.02)

# 两个干扰方块固定在左右两侧，只做视觉干扰，不直接挡住抓取路径。
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

# 这个 home 姿态保证每个 episode 开始时机器人状态一致。
# 这里一共 9 维：
# - 前 7 维：机械臂关节
# - 后 2 维：夹爪手指关节
HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)
EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))

EE_PICK_Z_OFFSET = 0.092
EE_PLACE_Z_OFFSET = 0.110
EE_HOVER_MARGIN = 0.140

# `controller.forward()` 使用的任务空间控制点，与
# `franka.end_effector.get_world_pose()` 读到的 `panda_hand` 反馈点之间，
# 在 Z 方向上存在一个比较稳定的常量偏差。
#
# 如果不补这个偏差，代码会出现一种很常见但很隐蔽的问题：
# 机械臂“看起来已经到位了”，但状态机一直判断“还没到”，于是 phase 卡死。
EE_FEEDBACK_Z_BIAS = 0.0985

# 以下这组阈值决定状态机什么时候认为：
# - 末端已经到目标点附近
# - 夹爪已经关到足够小
# - 方块已经真正进入托盘
#
# 抓取和放置比一般移动更敏感，所以阈值会更严格。
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
    """在 USD 场景里创建相机 prim。

    注意这里创建的是“场景中的相机对象”，不是后面直接采图的 Python 传感器。
    后面还会用 `Camera(...)` 再包装一次，变成能 `get_rgb()` 的接口。
    """

    # 取得当前正在编辑的 USD stage。
    stage = get_current_stage()
    # 在给定路径下创建一个 Camera prim。
    camera = UsdGeom.Camera.Define(stage, path)
    # 设置焦距。
    camera.CreateFocalLengthAttr(focal_length)
    # 设置裁剪范围，避免相机近裁面或远裁面太离谱。
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # 设置相机位姿。
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    """创建基础光照，让两路图像的视觉质量更稳定。

    目标不是做电影级布光，而是让：
    - 前视相机图像稳定
    - 手腕相机图像不至于过暗
    - 方块轮廓和颜色足够清楚
    """

    stage = get_current_stage()

    # DomeLight 提供整体环境光。
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1200.0)

    # RectLight 作为主光源，从斜上方照亮桌面。
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateWidthAttr(1.6)
    key.CreateHeightAttr(1.2)

    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.65, -0.20, 1.80))
    xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World) -> None:
    """添加一个极简房间，避免采图背景太空。

    这些墙和地板不是任务逻辑核心，只是为了：
    - 让图像背景更稳定
    - 避免视觉上像“悬在空白世界里”
    """

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
    """创建放置托盘。

    托盘由 1 个底板和 4 面墙拼出来。
    这样做的好处是：
    - 结构简单
    - 更容易直接按几何范围判断方块是否放进托盘
    """

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
    """把 Franka 以可控制机械臂的形式加入场景。

    这里做了两件事：
    1. 把官方 Franka USD 资产引用进 stage
    2. 用 `SingleManipulator` 把它包装成 Isaac 可控机械臂对象
    """

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    # 配置并行夹爪。这里定义了手指关节名、张开位置、闭合位置等。
    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    # 把 Franka 加入场景，并把它注册为一个 `SingleManipulator`。
    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H], dtype=np.float32),
        )
    )
    # 默认让夹爪在 reset 后张开。
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
    """构建完整场景。

    这一步只负责“把世界搭出来”，不负责采集、不负责控制器、不负责保存数据。
    """

    # 创建世界对象，规定 1 个 USD 单位等于 1 米。
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

    # `reset()` 是 Isaac Sim 中非常关键的一步：
    # 让场景对象、关节和物理状态进入一个可控的初始状态。
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
    """用传感器接口包装前视相机与手腕相机。

    前面的 `create_camera_prim()` 只是把相机放到场景里；
    这里才是真正创建“程序里可读图像”的传感器接口。
    """

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
    # 初始化传感器，之后才能稳定读取图像。
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
    """给物理系统一点稳定时间。

    这一步非常朴素，但很常用：
    reset 完机器人和方块后，先让仿真跑几帧，
    避免马上开始采集时带着上一次残留速度或瞬态抖动。
    """

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

    # 抓取点：XY 对准方块中心，Z 在方块中心之上加一个抓取偏移。
    pick_position = np.array(
        [
            float(target_cube_position[0]),
            float(target_cube_position[1]),
            float(target_cube_position[2] + EE_PICK_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    pick_hover_position = pick_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    # 放置点：XY 对准托盘目标位置，Z 在托盘目标上方加一个放置偏移。
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
    # 每一个阶段都包含：
    # - 名称：便于调试
    # - target：当前阶段要去的末端目标点
    # - gripper_closed：这一阶段夹爪应该是张开还是闭合
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
    """把当前阶段目标编码成一条训练动作。

    最终动作向量只有 4 维：
    - 目标末端位置 `x y z`
    - 夹爪闭合标记 `0/1`
    """

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
    """读取一帧 RGB 图像，并统一成 uint8。

    Isaac Sim 的 Camera 传感器在初始化后的前几帧里，偶尔会返回 `None`。
    尤其是在：
    - 相机刚 initialize 完
    - 场景刚 reset 完
    - 第一个 episode 的第一批图像

    所以这里做两层兜底：
    1. 先尝试 `get_rgb()`
    2. 如果为空，再尝试 `get_rgba()` 并裁成前三个通道
    3. 如果还是没有，就短暂重试几次
    """

    for _ in range(CAMERA_CAPTURE_RETRIES):
        rgb = camera.get_rgb()
        if rgb is not None:
            return np.asarray(rgb, dtype=np.uint8)

        rgba_getter = getattr(camera, "get_rgba", None)
        if callable(rgba_getter):
            rgba = rgba_getter()
            if rgba is not None:
                rgba = np.asarray(rgba, dtype=np.uint8)
                return rgba[..., :3]

        # 相机偶尔只是在当前帧还没准备好，给它一点时间继续渲染。
        simulation_app.update()

    raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")


def get_task_space_ee_pose(franka: SingleManipulator) -> tuple[np.ndarray, np.ndarray]:
    """读取与 RMPFlow 目标位置同一参考系下的末端位姿。

    `franka.end_effector` 对应的 `panda_hand` 反馈点在 Z 方向上比控制点更高，
    因此这里对位置做固定偏差修正，让：
    1. 阶段推进逻辑
    2. 抓取/放置几何判断
    3. 保存到数据集里的末端位置
    都和动作真值使用同一套任务空间参考系。
    """

    # 先拿 Isaac 默认反馈的 `panda_hand` 位姿。
    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32).copy()
    # 再把 Z 减去一个经验偏差，让它和控制器使用的任务空间参考点对齐。
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
    """生成每个 episode 共享的元数据字符串。

    这样每个 `.npz` 自己就带了一份轻量说明书，
    后面单独拿某个文件出来看，也能知道状态和动作字段各自代表什么。
    """

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
    """采集单个 episode。

    这是整份脚本的核心：
    - reset 机器人和方块
    - 生成专家阶段目标
    - 每帧执行控制
    - 逐帧记录图像 / 状态 / 动作
    - 成功就保存，失败就丢弃
    """

    # 每个 episode 用独立 seed，保证采样可复现。
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

    # `done` / `success` 是这条 episode 的总体状态。
    done = False
    success = False
    # `cube_attached` 不是吸附作弊，而是表示“当前判定为已经抓住方块，
    # 需要在抓取保持与释放阶段用平滑方式跟随末端”。
    cube_attached = False
    start_ee_position, _ = get_task_space_ee_pose(franka)
    phase_targets = build_phase_targets(
        start_ee_position=start_ee_position,
        target_cube_position=cube_positions[TARGET_CUBE_NAME],
    )
    # 下面这些变量共同构成一个非常轻量的阶段机。
    phase_index = 0
    phase_reached_frames = 0
    grasp_hold_frames = 0
    release_hold_frames = 0
    place_settle_step = 0
    attach_offset = np.array([0.0, 0.0, -EE_PICK_Z_OFFSET], dtype=np.float32)
    release_start_position = PLACE_GOAL_POSITION.copy()

    # 逐帧推进。不是按固定阶段时长硬切，而是最多给到 `EPISODE_MAX_STEPS` 帧。
    for _ in range(EPISODE_MAX_STEPS):
        if phase_index >= len(phase_targets):
            break

        # 当前阶段描述。
        phase = phase_targets[phase_index]
        phase_name = str(phase["name"])
        target_position = np.asarray(phase["target"], dtype=np.float32)
        gripper_closed = bool(phase["gripper_closed"])
        task_space_action = make_task_space_action(target_position, gripper_closed)
        target_position = np.asarray(task_space_action[:3], dtype=np.float32)

        # `controller.forward()` 把任务空间目标转换成机械臂关节动作。
        arm_action = controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=EE_TARGET_ORIENTATION,
        )
        # 夹爪单独生成开 / 关动作。
        gripper_action = franka.gripper.forward("close" if gripper_closed else "open")
        # 机械臂动作和夹爪动作需要先合并，才能一次性下发给 Franka。
        joint_action = merge_joint_actions(franka.num_dof, arm_action, gripper_action)
        franka.apply_action(joint_action)
        world.step(render=True)

        # 读取当前末端位置和方块位置，用于阶段判定和几何检查。
        ee_position, _ = get_task_space_ee_pose(franka)
        cube_position, _ = target_cube.get_world_pose()
        cube_position = np.asarray(cube_position, dtype=np.float32)
        joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        gripper_width = float(joint_positions[7] + joint_positions[8])
        desired_cube_position = ee_position + attach_offset
        # 不同阶段用不同的到位阈值。
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

        # 如果已经抓住方块，并且当前不是释放阶段，就让方块平滑跟随末端。
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
        # 抓取阶段：只有当末端够接近、夹爪够闭合、方块位置也合理时，
        # 才判定为“已经抓住”。
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
        # 释放阶段：只有当末端真的到托盘附近，才开始把方块平滑放到托盘目标处。
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

        # 到这里再记录当前帧数据，保证图像和状态对应“动作执行后”的结果。
        front_images.append(capture_rgb(front_camera))
        wrist_images.append(capture_rgb(wrist_camera))
        states.append(get_robot_state(franka))
        actions.append(task_space_action)

        success = is_cube_inside_box(target_cube)
        rewards.append(1.0 if success else 0.0)
        dones.append(False)

        # 下面是阶段推进逻辑。
        #
        # 普通移动阶段：
        # 末端连续若干帧都到位，才进入下一阶段。
        #
        # 抓取阶段：
        # 需要真的抓住并保持一段时间。
        #
        # 释放阶段：
        # 需要方块真的被放进托盘并稳定若干帧。
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

    # 失败轨迹直接丢弃，不保存，只打印调试信息。
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

    # 成功轨迹才落盘。
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
    """脚本主入口。

    入口函数只负责串起采集流程：
    - 搭场景
    - 初始化 Franka 和相机
    - 创建控制器
    - 循环采集 episode
    """

    output_dir = ARGS.output_dir.resolve()

    world, franka, cubes = build_scene()
    franka.initialize()
    front_camera, wrist_camera = create_cameras()
    # 相机初始化后再多跑几帧，尽量避免第一帧采图时传感器还没热起来。
    settle_scene(world, CAMERA_WARMUP_STEPS)

    controller = RMPFlowController(
        name="franka_scripted_rmpflow",
        robot_articulation=franka,
    )

    # `success_count` 是最终真正保存下来的条数。
    success_count = 0
    # `attempt_count` 是总共尝试了多少轮。
    attempt_count = 0
    max_attempts = max(ARGS.episodes * 12, ARGS.episodes + 5)

    print(f"输出目录: {output_dir}", flush=True)
    print(f"目标成功 episode 数: {ARGS.episodes}", flush=True)
    print("开始采集，仅保存成功轨迹...", flush=True)

    # 只要还没采够成功条数，就继续尝试。
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
        # 成功才计入目标数量；失败轨迹被丢弃。
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
        # 直接运行脚本时，从这里进入。
        main()
    except Exception:
        # Isaac Sim 脚本出错时，完整 traceback 特别重要。
        traceback.print_exc()
        raise
    finally:
        # 无论成功还是异常退出，都要把 SimulationApp 关掉。
        simulation_app.close()
