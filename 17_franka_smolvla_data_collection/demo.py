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
        "--capture-every",
        type=int,
        default=2,
        help="每隔多少个仿真 step 记录一帧。默认 2，可明显提升采集速度。",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="是否使用 np.savez_compressed 压缩保存。默认关闭以提升写盘速度。",
    )
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
# 俯视相机路径。它固定在世界坐标系下，从桌面上方观察整个操作区。
TOP_CAMERA_PATH = "/World/top_camera"

# 前视相机改成更贴近桌面的斜上方机位，减少整张图里“空背景”的占比，
# 让红色 cube、夹爪和托盘在训练图像中更大、更清楚。
FRONT_CAMERA_EYE = np.array([0.92, -0.62, 0.86], dtype=np.float32)
# 目标注视点略压低到桌面操作区中心，避免视野过多浪费在房间和桌子边缘。
FRONT_CAMERA_TARGET = np.array([0.50, 0.02, 0.43], dtype=np.float32)
FRONT_CAMERA_FOCAL_LENGTH = 14.0

# 俯视相机稍微后撤并抬高，保证整张桌面基本都在画面里。
TOP_CAMERA_EYE = np.array([0.50, -0.08, 1.34], dtype=np.float32)
TOP_CAMERA_TARGET = np.array([0.50, 0.00, 0.40], dtype=np.float32)
TOP_CAMERA_FOCAL_LENGTH = 15.0

# 两路相机分辨率。
FRONT_CAMERA_RESOLUTION = (640, 480)
TOP_CAMERA_RESOLUTION = (640, 480)
# 相机采样频率。
CAMERA_FREQUENCY = 20
CAMERA_WARMUP_STEPS = 20
CAMERA_CAPTURE_RETRIES = 6
DEBUG_PRINT_EVERY_STEPS = 8
CAPTURE_EVERY_STEPS = max(1, int(ARGS.capture_every))

# 训练方块尺寸。适当放大一些，让视觉上更醒目，也更容易稳定抓取。
CUBE_SIZE = np.array([0.055, 0.055, 0.055], dtype=np.float32)
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

# 红色目标方块不再只在一个小矩形里随机，而是改成“多个桌面分区随机抽样”。
#
# 这样做有两个目的：
# 1. 让抓取样本尽量覆盖桌面各个角落，提升泛化性
# 2. 仍把采样限制在 Franka 相对稳定、可达的工作区里
#
# 每个区域格式都是：
# `(区域名, (x_min, x_max), (y_min, y_max))`
TARGET_CUBE_SPAWN_REGIONS = [
    ("left_front", (0.28, 0.38), (0.12, 0.26)),
    ("left_mid", (0.28, 0.40), (-0.08, 0.10)),
    ("left_back", (0.28, 0.38), (-0.26, -0.12)),
    ("center_front", (0.42, 0.54), (0.02, 0.16)),
    ("center_back", (0.42, 0.58), (-0.24, -0.08)),
    ("right_back", (0.56, 0.66), (-0.26, -0.10)),
]
# 采样时要明确避开托盘内侧，并给一个额外安全边界，避免 cube 刚好刷在盒子边上。
TARGET_CUBE_BOX_EXCLUSION_MARGIN = 0.055
# 也避免太贴近两个固定干扰方块，否则一开始就可能发生重叠或碰撞。
TARGET_CUBE_DISTRACTOR_CLEARANCE_XY = 0.095
TARGET_CUBE_SAMPLE_MAX_TRIES = 80

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

# 抓取和放置各自拆成“靠近高度”和“真正执行高度”两层：
# - `*_APPROACH_*`：更安全，先把末端带到目标附近
# - `*_ACTION_*`  ：真正闭爪 / 松爪时所在的高度
#
# 这样可以避免机械臂还离 cube 中心较远时就闭爪，也能让放置阶段明显下探到托盘内。
# 这里继续把抓取高度压低：
# - `EE_PICK_APPROACH_Z_OFFSET` 先降到更贴近 cube 的预抓取位
# - `EE_PICK_ACTION_Z_OFFSET`   再降到接近 cube 中心附近的位置闭爪
EE_PICK_APPROACH_Z_OFFSET = 0.060
EE_PICK_ACTION_Z_OFFSET = 0.013
EE_PLACE_APPROACH_Z_OFFSET = 0.110
EE_PLACE_ACTION_Z_OFFSET = 0.085
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
PRE_GRASP_STABLE_FRAMES = 6
CLOSE_GRIPPER_STABLE_FRAMES = 4
PRE_RELEASE_STABLE_FRAMES = 4
OPEN_GRIPPER_STABLE_FRAMES = 4
# 为了避免某些 Isaac 版本里夹爪宽度反馈稍有偏差，抓取 / 松爪阶段除了宽度阈值，
# 再额外给一小段“命令生效时间”作为兜底。
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
# 下面这组阈值不再用于“吸附”，而是只用于判定：
# cube 是否真的被物理夹住并抬离桌面。
GRASP_VERIFY_XY_THRESHOLD = 0.020
GRASP_VERIFY_Z_THRESHOLD = 0.030
GRASP_LIFT_MIN_HEIGHT = 0.010
GRIPPER_OPEN_WIDTH_THRESHOLD = 0.085

EPISODE_MAX_STEPS = 680
# 放置尾段对训练很关键，不能再被隔帧采样漏掉。
ALWAYS_CAPTURE_PHASES = {
    "release_open",
    "post_release_settle",
    "retreat",
    "return_idle",
}
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
    "target_cube_pos_x",
    "target_cube_pos_y",
    "target_cube_pos_z",
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
        focal_length=FRONT_CAMERA_FOCAL_LENGTH,
    )
    create_camera_prim(
        path=TOP_CAMERA_PATH,
        position=tuple(TOP_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(0.0, 90.0, 0.0),
        focal_length=TOP_CAMERA_FOCAL_LENGTH,
    )

    # `reset()` 是 Isaac Sim 中非常关键的一步：
    # 让场景对象、关节和物理状态进入一个可控的初始状态。
    world.reset()

    set_camera_view(
        eye=FRONT_CAMERA_EYE,
        target=FRONT_CAMERA_TARGET,
        camera_prim_path=FRONT_CAMERA_PATH,
    )
    set_camera_view(
        eye=TOP_CAMERA_EYE,
        target=TOP_CAMERA_TARGET,
        camera_prim_path=TOP_CAMERA_PATH,
    )
    if not ARGS.headless:
        set_camera_view(
            eye=FRONT_CAMERA_EYE,
            target=FRONT_CAMERA_TARGET,
            camera_prim_path="/OmniverseKit_Persp",
        )

    return world, franka, cubes


def create_cameras() -> tuple[Camera, Camera]:
    """用传感器接口包装前视相机与俯视相机。

    前面的 `create_camera_prim()` 只是把相机放到场景里；
    这里才是真正创建“程序里可读图像”的传感器接口。
    """

    front_camera = Camera(
        prim_path=FRONT_CAMERA_PATH,
        name="front_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=FRONT_CAMERA_RESOLUTION,
    )
    top_camera = Camera(
        prim_path=TOP_CAMERA_PATH,
        name="top_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=TOP_CAMERA_RESOLUTION,
    )
    # 初始化传感器，之后才能稳定读取图像。
    front_camera.initialize()
    top_camera.initialize()
    return front_camera, top_camera


def sample_cube_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """采样三色方块的位置。

    红色方块是唯一抓取目标，因此这里专门做“多区域采样 + 排除约束”：
    1. 先随机选一个桌面分区
    2. 再在该分区内均匀采样
    3. 如果样本落进托盘区域，或者太靠近固定干扰块，就重采

    这样既能明显扩大分布范围，也不会一开始就把方块刷进盒子里。
    """

    for _ in range(TARGET_CUBE_SAMPLE_MAX_TRIES):
        _, x_range, y_range = TARGET_CUBE_SPAWN_REGIONS[rng.integers(len(TARGET_CUBE_SPAWN_REGIONS))]
        target_position = np.array(
            [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                TABLE_SURFACE_Z + CUBE_HALF_Z,
            ],
            dtype=np.float32,
        )
        # 托盘外边框加一点 margin 后，一律不允许作为初始采样点。
        inside_box_x = abs(float(target_position[0] - PLACE_BOX_CENTER[0])) <= (
            PLACE_BOX_OUTER_X / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        inside_box_y = abs(float(target_position[1] - PLACE_BOX_CENTER[1])) <= (
            PLACE_BOX_OUTER_Y / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        if inside_box_x and inside_box_y:
            continue

        # 同时避开固定干扰块，防止 reset 时直接和障碍块太近。
        too_close_to_distractor = False
        for distractor_position in DISTRACTOR_CUBE_LAYOUT.values():
            if planar_distance(target_position, distractor_position) < TARGET_CUBE_DISTRACTOR_CLEARANCE_XY:
                too_close_to_distractor = True
                break
        if too_close_to_distractor:
            continue

        return {TARGET_CUBE_NAME: target_position}

    # 理论上前面的多区域采样应该很容易成功。
    # 如果连续多次都撞上排除条件，就退回一个保底点，避免 episode 直接报错。
    return {
        TARGET_CUBE_NAME: np.array([0.46, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32)
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

    # 抓取分两层：
    # 1. `pick_approach_position` 先下降到较稳的预抓取高度
    # 2. `pick_grasp_position` 再进一步贴近 cube 中心，之后才闭爪
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

    # 放置也分两层：
    # 1. `place_approach_position` 先移动到托盘上方
    # 2. `place_release_position` 再下探到更低的位置执行松爪
    place_approach_position = np.array(
        [
            float(PLACE_GOAL_POSITION[0]),
            float(PLACE_GOAL_POSITION[1]),
            float(PLACE_GOAL_POSITION[2] + EE_PLACE_APPROACH_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    place_release_position = np.array(
        [
            float(PLACE_GOAL_POSITION[0]),
            float(PLACE_GOAL_POSITION[1]),
            float(PLACE_GOAL_POSITION[2] + EE_PLACE_ACTION_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    place_hover_position = place_approach_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)

    current_position = np.asarray(start_ee_position, dtype=np.float32)
    # 每一个阶段都包含：
    # - 名称：便于调试
    # - target：当前阶段要去的末端目标点
    # - gripper_closed：这一阶段夹爪应该是张开还是闭合
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


def rounded_list(array: np.ndarray, decimals: int = 4) -> list[float]:
    """把向量整理成更适合终端调试查看的短列表。"""

    return np.round(np.asarray(array, dtype=np.float32), decimals).tolist()


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

    if phase_name == "pre_grasp_settle":
        return EE_GRASP_ALIGN_XY_THRESHOLD, EE_GRASP_ALIGN_Z_THRESHOLD
    if phase_name in {"descend_pick", "grasp_close", "grasp_hold"}:
        return EE_PICK_REACH_XY_THRESHOLD, EE_PICK_REACH_Z_THRESHOLD
    if phase_name in {"descend_place", "pre_release_settle", "release_open", "post_release_settle"}:
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


def get_robot_state(franka: SingleManipulator, target_cube: DynamicCuboid) -> np.ndarray:
    """读取训练时常用的状态向量。"""

    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    ee_position, ee_orientation = get_task_space_ee_pose(franka)
    cube_position, _ = target_cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)
    gripper_width = float(joint_positions[7] + joint_positions[8])

    return np.concatenate(
        [
            joint_positions[:7],
            ee_position[:3],
            ee_orientation[:4],
            np.array([gripper_width], dtype=np.float32),
            cube_position[:3],
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
            "top_camera_resolution": TOP_CAMERA_RESOLUTION,
            "episode_max_steps": EPISODE_MAX_STEPS,
            "capture_every_steps": CAPTURE_EVERY_STEPS,
            "save_compressed": bool(ARGS.compress),
        },
        ensure_ascii=False,
        indent=2,
    )


def save_episode(
    output_dir: Path,
    episode_index: int,
    front_images: list[np.ndarray],
    top_images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    success: bool,
    seed: int,
) -> Path:
    """把单个 episode 保存成 npz。

    默认使用非压缩 `np.savez`，因为两路 RGB 图像做 zip 压缩会显著拖慢采集速度。
    如果更在意磁盘占用，可以用 `--compress` 切回压缩保存。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"episode_{episode_index:05d}.npz"

    payload = {
        "observation.images.front": np.asarray(front_images, dtype=np.uint8),
        "observation.images.top": np.asarray(top_images, dtype=np.uint8),
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
    }
    if ARGS.compress:
        np.savez_compressed(file_path, **payload)
    else:
        np.savez(file_path, **payload)
    return file_path


def collect_episode(
    world: World,
    franka: SingleManipulator,
    cubes: dict[str, DynamicCuboid],
    front_camera: Camera,
    top_camera: Camera,
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
    top_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    # `done` / `success` 是这条 episode 的总体状态。
    done = False
    success = False
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
    debug_step_index = 0
    previous_phase_name = ""
    phase_elapsed_frames = 0
    # 这里的参考偏移要和真正闭爪时的抓取高度一致，
    # 否则“cube 是否跟在末端下方”的物理判定会偏高。
    attach_offset = np.array([0.0, 0.0, -EE_PICK_ACTION_Z_OFFSET], dtype=np.float32)

    print(
        f"[episode {episode_index:03d}]"
        f" sampled_target_cube={rounded_list(cube_positions[TARGET_CUBE_NAME])}"
        f" place_goal={rounded_list(PLACE_GOAL_POSITION)}"
        f" start_ee={rounded_list(start_ee_position)}",
        flush=True,
    )

    # 逐帧推进。不是按固定阶段时长硬切，而是最多给到 `EPISODE_MAX_STEPS` 帧。
    for _ in range(EPISODE_MAX_STEPS):
        debug_step_index += 1
        if phase_index >= len(phase_targets):
            break

        # 当前阶段描述。
        phase = phase_targets[phase_index]
        phase_name = str(phase["name"])
        if phase_name != previous_phase_name:
            phase_elapsed_frames = 0
            print(
                f"[episode {episode_index:03d}] enter_phase={phase_name}"
                f" target={rounded_list(np.asarray(phase['target'], dtype=np.float32))}",
                flush=True,
            )
            previous_phase_name = phase_name
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
        phase_elapsed_frames += 1

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
        # “真正抓住”不再通过 set_world_pose 吸附来决定，
        # 而是要求 cube 满足两个物理迹象：
        # 1. 它仍然跟在末端下方
        # 2. 它已经明显抬离桌面
        cube_following_ee = is_position_close(
            cube_position,
            desired_cube_position,
            GRASP_VERIFY_XY_THRESHOLD,
            GRASP_VERIFY_Z_THRESHOLD,
        )
        cube_lifted_from_table = float(cube_position[2] - (TABLE_SURFACE_Z + CUBE_HALF_Z)) >= GRASP_LIFT_MIN_HEIGHT
        cube_grasped_by_physics = (
            gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD
            and cube_following_ee
            and cube_lifted_from_table
        )
        ee_to_target_xy = planar_distance(ee_position, target_position)
        ee_to_target_z = vertical_distance(ee_position, target_position)
        ee_to_cube_xy = planar_distance(ee_position, cube_position)
        ee_to_cube_z = vertical_distance(ee_position, cube_position)
        cube_to_desired_xy = planar_distance(cube_position, desired_cube_position)
        cube_to_desired_z = vertical_distance(cube_position, desired_cube_position)

        success = is_cube_inside_box(target_cube)
        should_capture_frame = (
            phase_name in ALWAYS_CAPTURE_PHASES
            or (debug_step_index % CAPTURE_EVERY_STEPS) == 0
        )
        # 到这里再记录当前帧数据，保证图像和状态对应“动作执行后”的结果。
        # 默认改成隔帧采样，主要是为了降低两路图像采集和写盘压力。
        if should_capture_frame:
            front_images.append(capture_rgb(front_camera))
            top_images.append(capture_rgb(top_camera))
            states.append(get_robot_state(franka, target_cube))
            actions.append(task_space_action)
            rewards.append(1.0 if success else 0.0)
            dones.append(False)

        if phase_name in {
            "pre_grasp_settle",
            "grasp_close",
            "grasp_hold",
            "pre_release_settle",
            "release_open",
            "post_release_settle",
        } and debug_step_index % DEBUG_PRINT_EVERY_STEPS == 0:
            print(
                f"[episode {episode_index:03d}]"
                f" phase={phase_name}"
                f" ee={rounded_list(ee_position)}"
                f" cube={rounded_list(cube_position)}"
                f" target={rounded_list(target_position)}"
                f" gripper_width={round(gripper_width, 4)}"
                f" reach_xy={round(ee_to_target_xy, 4)}/{round(ee_reach_xy_threshold, 4)}"
                f" reach_z={round(ee_to_target_z, 4)}/{round(ee_reach_z_threshold, 4)}"
                f" ee_cube_xy={round(ee_to_cube_xy, 4)}"
                f" ee_cube_z={round(ee_to_cube_z, 4)}"
                f" cube_follow_xy={round(cube_to_desired_xy, 4)}/{round(GRASP_VERIFY_XY_THRESHOLD, 4)}"
                f" cube_follow_z={round(cube_to_desired_z, 4)}/{round(GRASP_VERIFY_Z_THRESHOLD, 4)}"
                f" phase_elapsed={phase_elapsed_frames}"
                f" phase_frames={phase_reached_frames}"
                f" grasp_hold_frames={grasp_hold_frames}"
                f" release_hold_frames={release_hold_frames}"
                f" ee_reached={ee_reached_target}"
                f" cube_following={cube_following_ee}"
                f" cube_lifted={cube_lifted_from_table}"
                f" cube_grasped={cube_grasped_by_physics}"
                f" success={success}",
                flush=True,
            )

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
        if phase_name == "grasp_close":
            # 闭爪动作单独成一个阶段：
            # 末端已经停在 cube 中心附近后，再给夹爪几帧真正闭合时间。
            # 如果当前版本的夹爪宽度反馈略慢或略有偏差，也允许靠时间兜底推进，
            # 否则会出现“已经在抓取点，但永远不进下一阶段”的卡死现象。
            if ee_reached_target:
                if gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD:
                    phase_reached_frames += 1
                else:
                    phase_reached_frames = 0
                if (
                    phase_reached_frames >= CLOSE_GRIPPER_STABLE_FRAMES
                    or phase_elapsed_frames >= GRASP_CLOSE_COMMAND_STEPS
                ):
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason={'gripper_closed' if phase_reached_frames >= CLOSE_GRIPPER_STABLE_FRAMES else 'close_command_timeout'}"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" phase_frames={phase_reached_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                phase_reached_frames = 0
        elif phase_name == "grasp_hold":
            # 这里不再要求“已经抬离桌面”才能进入 lift。
            # 原因是当前阶段目标还停在抓取低位，如果把“已抬起”作为前置条件，
            # 状态机会卡在抓取点附近，后面的 lift / transfer / place 永远不会执行。
            #
            # 因此这一步只负责：
            # 1. 让夹爪在抓取点保持闭合一小段时间
            # 2. 给物理系统一点时间建立接触
            #
            # 真正是否抓稳，继续通过后续 lift 阶段的运动结果和最终 success 来判断。
            if ee_reached_target:
                grasp_hold_frames += 1
                if grasp_hold_frames >= GRASP_HOLD_STEPS:
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason=grasp_hold_complete"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" cube_grasped={cube_grasped_by_physics}"
                        f" grasp_hold_frames={grasp_hold_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                grasp_hold_frames = 0
        elif phase_name == "pre_grasp_settle":
            # 真正闭爪前，先要求末端在 cube 中心附近稳定停住几帧。
            # 这样不会一进入目标区域就立刻开始闭爪。
            #
            # 这里不强依赖“张开宽度反馈一定大于阈值”，因为不同版本里开爪反馈
            # 可能略小，但这不应该阻止进入闭爪阶段。
            if ee_reached_target:
                phase_reached_frames += 1
                if phase_reached_frames >= PRE_GRASP_STABLE_FRAMES:
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason=ee_stable_near_cube_center"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" phase_frames={phase_reached_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                phase_reached_frames = 0
        elif phase_name == "pre_release_settle":
            # 到托盘释放位后，先保持闭爪停稳几帧，再执行真正松爪。
            if ee_reached_target:
                phase_reached_frames += 1
                if phase_reached_frames >= PRE_RELEASE_STABLE_FRAMES:
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason=ee_stable_over_box"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" phase_frames={phase_reached_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                phase_reached_frames = 0
        elif phase_name == "release_open":
            # 松爪动作单独给几帧，让夹爪真正打开，而不是瞬间切到下一阶段。
            if ee_reached_target:
                if gripper_width >= GRIPPER_OPEN_WIDTH_THRESHOLD:
                    phase_reached_frames += 1
                else:
                    phase_reached_frames = 0
                if (
                    phase_reached_frames >= OPEN_GRIPPER_STABLE_FRAMES
                    or phase_elapsed_frames >= RELEASE_OPEN_COMMAND_STEPS
                ):
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason={'gripper_opened' if phase_reached_frames >= OPEN_GRIPPER_STABLE_FRAMES else 'open_command_timeout'}"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" phase_frames={phase_reached_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            else:
                phase_reached_frames = 0
        elif phase_name == "post_release_settle":
            # 松爪后停在托盘上方等待方块自然落稳，再认定放置完成。
            # 如果本次没有真正放成功，也不要在这里卡死，这样至少能把“松爪+撤离”
            # 这一整段动作跑完，便于从终端日志和画面一起调试。
            if gripper_width >= GRIPPER_OPEN_WIDTH_THRESHOLD and success:
                release_hold_frames += 1
                if release_hold_frames >= RELEASE_HOLD_STEPS:
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason=place_success_verified"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" gripper_width={round(gripper_width, 4)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" release_hold_frames={release_hold_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
            elif phase_elapsed_frames >= POST_RELEASE_WAIT_STEPS:
                print(
                    f"[episode {episode_index:03d}] advance_phase={phase_name}"
                    f" reason=post_release_timeout"
                    f" ee={rounded_list(ee_position)}"
                    f" cube={rounded_list(cube_position)}"
                    f" gripper_width={round(gripper_width, 4)}"
                    f" phase_elapsed={phase_elapsed_frames}"
                    f" success={success}",
                    flush=True,
                )
                phase_index += 1
                phase_reached_frames = 0
            else:
                release_hold_frames = 0
        else:
            if ee_reached_target:
                phase_reached_frames += 1
                if phase_reached_frames >= PHASE_STABLE_FRAMES:
                    print(
                        f"[episode {episode_index:03d}] advance_phase={phase_name}"
                        f" reason=ee_reached_target"
                        f" ee={rounded_list(ee_position)}"
                        f" cube={rounded_list(cube_position)}"
                        f" target={rounded_list(target_position)}"
                        f" phase_elapsed={phase_elapsed_frames}"
                        f" phase_frames={phase_reached_frames}",
                        flush=True,
                    )
                    phase_index += 1
                    phase_reached_frames = 0
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
            f" ee={rounded_list(final_ee_position)}"
            f" target={rounded_list(final_phase_target)}"
            f" cube={rounded_list(final_cube_position)}"
            f" cube_grasped={cube_grasped_by_physics}"
            f" gripper_width={round(gripper_width, 4)}",
            flush=True,
        )
        return False, None

    # 成功轨迹才落盘。
    save_path = save_episode(
        output_dir=output_dir,
        episode_index=episode_index,
        front_images=front_images,
        top_images=top_images,
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
    front_camera, top_camera = create_cameras()
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
            top_camera=top_camera,
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
