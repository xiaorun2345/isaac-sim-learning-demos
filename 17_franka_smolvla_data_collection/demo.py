"""Franka SmolVLA 数据采集示例。

这个脚本的目标很明确：

1. 在 Isaac Sim 里搭一个固定的 Franka 桌面抓取场景
2. 让一个“写死但稳定”的专家策略自动执行抓取放置
3. 逐帧记录图像、机器人状态、专家动作、奖励与结束标记
4. 把每条成功轨迹保存成一个独立 episode 的 `.npz` 文件

这里的数据组织方式刻意贴近后续 SmolVLA / LeRobot 这类流程常见格式，
这样后面无论是转 dataset，还是写 replay / train 脚本，都更顺手。

这个脚本记录的核心内容包括：

1. 前视相机图像
2. 手腕相机图像
3. 机器人状态
   - 7 维机械臂关节
   - 末端位置
   - 末端姿态四元数
   - 夹爪宽度
4. 专家动作
   - 目标末端位置 xyz
   - 夹爪是否闭合

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
    """解析少量必要参数，避免把脚本做成参数堆。

    这里故意只保留几个最常改的参数：

    - `episodes`：目标成功采集条数
    - `headless`：是否无界面运行
    - `output_dir`：保存原始数据的目录

    其余大多数采集超参数，例如场景尺寸、抓取步数、相机分辨率，都直接写成
    文件内常量，方便把这个脚本当成教学示例阅读。
    """

    # 用整个模块 docstring 作为命令行帮助描述，这样 `--help` 会更完整。
    parser = argparse.ArgumentParser(description=__doc__)
    # 成功保存多少个 episode 后停止。
    parser.add_argument("--episodes", type=int, default=20, help="采集多少个 episode。")
    # `store_true` 表示：只要命令行写了 `--headless`，这里就会变成 True。
    parser.add_argument("--headless", action="store_true", help="无界面运行。")
    # 默认输出目录放在当前 demo 目录下的 `outputs/raw`。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "raw",
        help="数据输出目录。",
    )
    return parser.parse_args()


# 先解析参数，后面创建 SimulationApp 时就能直接复用。
ARGS = parse_args()

# Isaac Sim 有个非常重要的约束：
# 大量 Isaac / Omniverse 模块必须在 `SimulationApp` 创建之后再导入，
# 否则很容易在扩展初始化阶段报错。
simulation_app = SimulationApp(
    {
        # 是否无界面运行。
        "headless": ARGS.headless,
        # 如果无界面，就顺便隐藏 UI。
        "hide_ui": ARGS.headless,
        # 选一个视觉质量更好的渲染器，方便采集相机图像。
        "renderer": "RaytracedLighting",
        # 主窗口/渲染分辨率。
        "width": 1280,
        "height": 720,
    }
)


# 从这里开始才导入 Isaac Sim 相关模块。
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
# 场景基础配置
# ----------------------------
# 下面这一组常量决定了桌子、机器人、相机、方块、托盘等对象的几何布局。
# 集中写在这里，目的是让场景结构一眼能扫清楚。

# 桌子高度，单位米。
TABLE_H = 0.40
# 桌子中心点。z 取高度一半，因为立方体类物体一般按中心点放置。
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
# 桌子的长宽高。
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
# 桌面平面的 z 坐标。
TABLE_SURFACE_Z = TABLE_H

# Franka 在当前 stage 里的 prim 路径。
FRANKA_PRIM_PATH = "/World/Franka"
# 前视相机路径。
FRONT_CAMERA_PATH = "/World/front_camera"
# 手腕相机挂在 `panda_hand` 下，是机器人局部结构的一部分。
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"

# 前视相机在世界坐标系中的位置。
FRONT_CAMERA_EYE = np.array([1.15, -1.10, 1.10], dtype=np.float32)
# 前视相机看的目标点。
FRONT_CAMERA_TARGET = np.array([0.46, 0.00, 0.55], dtype=np.float32)

# 两路图像的分辨率统一成 640x480。
FRONT_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (640, 480)
# 采图频率。
CAMERA_FREQUENCY = 20

# 目标方块的立方体尺寸。
CUBE_SIZE = np.array([0.045, 0.045, 0.045], dtype=np.float32)
# 半个 z 高度经常要用来算“放在桌面上时中心点应该多高”。
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)

# 目标方块名字。
TARGET_CUBE_NAME = "cube_red"
# 红色目标方块的颜色。
TARGET_CUBE_COLOR = np.array([0.88, 0.15, 0.15], dtype=np.float32)
# 干扰方块配置：名字 + 颜色。
DISTRACTOR_CUBE_SPECS = [
    ("cube_green", np.array([0.18, 0.66, 0.24], dtype=np.float32)),
    ("cube_blue", np.array([0.12, 0.36, 0.86], dtype=np.float32)),
]

# 红色目标方块压在最稳的抓取区，优先保证成功率。
# 这里没有把抓取目标撒得到处都是，而是先把专家数据做稳。
TARGET_CUBE_X_RANGE = (0.42, 0.44)
TARGET_CUBE_Y_RANGE = (-0.02, 0.02)

# 两个干扰方块固定在左右两侧，仅做视觉干扰，不进入主要抓取路径。
DISTRACTOR_CUBE_LAYOUT = {
    "cube_green": np.array([0.34, 0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
    "cube_blue": np.array([0.34, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
}
# 如果后续你想让干扰物更随机，可以用这两个抖动范围。
DISTRACTOR_CUBE_JITTER_X = 0.015
DISTRACTOR_CUBE_JITTER_Y = 0.020

# 放置托盘中心点。
PLACE_BOX_CENTER = np.array([0.64, 0.18], dtype=np.float32)
# 托盘外部长宽。
PLACE_BOX_OUTER_X = 0.18
PLACE_BOX_OUTER_Y = 0.18
# 托盘底板厚度、壁厚、壁高。
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.016
PLACE_BOX_WALL_H = 0.10
# 这个点表示“方块放进托盘后，希望它大概在哪个中心位置”。
PLACE_GOAL_POSITION = np.array(
    [PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], TABLE_SURFACE_Z + CUBE_HALF_Z],
    dtype=np.float32,
)

# 这个 home 姿态用来确保每个 episode 开始时机器人状态一致。
# 这里是 9 维：
# - 前 7 个值是机械臂关节
# - 后 2 个值是夹爪手指关节
HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)
# 末端抓取时统一朝下，用欧拉角转成四元数。
EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))

# 从方块中心到末端抓取参考点的 z 方向偏移。
EE_PICK_Z_OFFSET = 0.092
# 放置时末端相对托盘中目标点的 z 偏移。
EE_PLACE_Z_OFFSET = 0.110
# 抓取/放置前从上方悬停的额外抬高量。
EE_HOVER_MARGIN = 0.140

# 下面这组常量定义了每个阶段持续多少步。
# 相比原来的粗粒度轨迹，这里把“接近、下探、闭合、抬升、转移、释放”
# 拆得更细一些，让动作更连续，也给真实物理抓取留下接触和闭合时间。
APPROACH_PICK_HIGH_STEPS = 36
APPROACH_PICK_LOW_STEPS = 32
DESCEND_PICK_FAST_STEPS = 28
DESCEND_PICK_SLOW_STEPS = 24
PRE_GRASP_SETTLE_STEPS = 16
GRASP_CLOSE_STEPS = 28
GRASP_HOLD_STEPS = 44
LIFT_SLOW_STEPS = 26
LIFT_HIGH_STEPS = 34
TRANSFER_MID_STEPS = 44
TRANSFER_PLACE_STEPS = 42
DESCEND_PLACE_FAST_STEPS = 28
DESCEND_PLACE_SLOW_STEPS = 24
PRE_RELEASE_SETTLE_STEPS = 18
RELEASE_OPEN_STEPS = 26
POST_RELEASE_SETTLE_STEPS = 50
RETREAT_STEPS = 34

# 单个 episode 最多保留多少步。
EPISODE_MAX_STEPS = 600
# reset 后给物理系统多少步稳定时间。
EPISODE_SETTLE_STEPS = 20
# 成功后额外留一点缓冲时间，真实物理抓取释放后需要等 cube 落稳。
EPISODE_POST_SUCCESS_STEPS = 20

# 保存状态向量时的字段名字。
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

# 保存动作向量时的字段名字。
ACTION_NAMES = [
    "target_ee_pos_x",
    "target_ee_pos_y",
    "target_ee_pos_z",
    "target_gripper_closed",
]

# 用自然语言写一下任务描述，后续做 VLA / LeRobot 转换时通常有用。
TASK_DESCRIPTION = "Pick up the red cube with Franka and place it into the wooden tray."
EXPERT_TYPE = "scripted_task_space_rmpflow_physics_grasp"


def create_camera_prim(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """在 USD 场景里创建相机 prim。

    这里只负责“在 stage 中放一个相机对象”，真正读图像是在后面的 `Camera`
    传感器对象里完成的。
    """

    # 拿到当前正在编辑的 USD stage。
    stage = get_current_stage()
    # 在指定路径定义一个相机 prim。
    camera = UsdGeom.Camera.Define(stage, path)
    # 设置焦距。
    camera.CreateFocalLengthAttr(focal_length)
    # 设置裁剪范围，避免近处或远处渲染异常。
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # 用 USD 的通用变换 API 设置相机位姿。
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    """创建基础光照，让两路图像的视觉质量更稳定。

    这里用一个环境光 + 一个主光源，目的不是做特别真实的渲染，而是让视觉数据
    更稳定，不至于忽亮忽暗。
    """

    stage = get_current_stage()

    # 环境光，负责整体填亮。
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1200.0)

    # 主光源，负责给桌面和方块提供更清晰的阴影和形状感。
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateWidthAttr(1.6)
    key.CreateHeightAttr(1.2)

    # 主光源位于桌子斜上方。
    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.65, -0.20, 1.80))
    xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World) -> None:
    """添加简单房间，避免采图背景过空。

    这里的房间只是视觉背景，不参与任何复杂任务逻辑。
    """

    # 地板。
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
    # 后墙。
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
    # 左墙。
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
    # 右墙。
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

    # 桌子本身是一个固定长方体。
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

    托盘由一个底板和四面墙组成，方便后续直接按几何方式判断方块是否放进去。
    """

    # 底板中心高度。
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    # 墙体中心高度。
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    # 托盘统一颜色。
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)

    # 底板。
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
    # 左墙。
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
    # 右墙。
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
    # 前墙。
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
    # 后墙。
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

    这里不是简单加载模型，而是：

    1. 把 Franka USD 资源 reference 到 stage
    2. 用 `SingleManipulator` 封装成 Isaac 可控制的机械臂对象
    3. 配好夹爪张开/闭合参数
    """

    # 取 Isaac Sim 自带资产根路径。
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    # Franka 资产的标准位置。
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    # 把 Franka 挂到 `/World/Franka`。
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    # 配置夹爪。
    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    # 注册为 `SingleManipulator`，这样可以用高层控制接口。
    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H], dtype=np.float32),
        )
    )
    # 默认让夹爪张开。
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_training_cubes(world: World) -> dict[str, DynamicCuboid]:
    """添加一个可抓取的红色目标方块和两个固定干扰方块。

    注意：
    - 红色方块是动态物体，会参与物理模拟
    - 干扰方块是固定物体，只提供视觉干扰
    """

    # 红色目标方块是真正要抓取的对象，所以做成动态刚体。
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

    # 绿色和蓝色只是背景干扰。
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

    # 返回字典，后续 reset 和采集时按名字取更方便。
    return {TARGET_CUBE_NAME: target_cube}


def build_scene() -> tuple[World, SingleManipulator, dict[str, DynamicCuboid]]:
    """构建完整场景。

    返回：
    - `world`
    - `franka`
    - `cubes`
    """

    # 创建世界对象，设置 1 个单位等于 1 米。
    world = World(stage_units_in_meters=1.0)
    # 先搭视觉环境。
    create_lights()
    add_room(world)
    # 默认地面。
    world.scene.add_default_ground_plane()
    # 桌子和托盘。
    add_table(world)
    add_place_box(world)

    # 机械臂和目标物体。
    franka = add_franka(world)
    cubes = add_training_cubes(world)

    # 创建前视相机 prim。
    create_camera_prim(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-35.0, 0.0, 45.0),
        focal_length=10.0,
    )
    # 创建手腕相机 prim。
    create_camera_prim(
        path=WRIST_CAMERA_PATH,
        position=(0.06, 0.0, 0.03),
        rotation_xyz_deg=(-95.0, 0.0, -90.0),
        focal_length=4.0,
    )

    # `reset()` 是 Isaac Sim 初始化里非常关键的一步。
    world.reset()

    # 设置前视相机真正看向哪里。
    set_camera_view(
        eye=FRONT_CAMERA_EYE,
        target=FRONT_CAMERA_TARGET,
        camera_prim_path=FRONT_CAMERA_PATH,
    )
    # 如果有 UI，就顺手把默认透视视角也切过去。
    if not ARGS.headless:
        set_camera_view(
            eye=FRONT_CAMERA_EYE,
            target=FRONT_CAMERA_TARGET,
            camera_prim_path="/OmniverseKit_Persp",
        )

    return world, franka, cubes


def create_cameras() -> tuple[Camera, Camera]:
    """用传感器接口包装前视相机与手腕相机。

    前面的 `create_camera_prim()` 只是往 USD stage 放了相机 prim。
    这里的 `Camera` 才是后面真正拿来读 RGB 图像的 Isaac 传感器接口。
    """

    # 包装前视相机。
    front_camera = Camera(
        prim_path=FRONT_CAMERA_PATH,
        name="front_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=FRONT_CAMERA_RESOLUTION,
    )
    # 包装手腕相机。
    wrist_camera = Camera(
        prim_path=WRIST_CAMERA_PATH,
        name="wrist_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=WRIST_CAMERA_RESOLUTION,
    )
    # 初始化两个传感器。
    front_camera.initialize()
    wrist_camera.initialize()
    return front_camera, wrist_camera


def sample_cube_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """采样三色方块的位置。

    红色方块是唯一抓取目标，因此放在最稳的抓取区。
    绿色和蓝色方块是固定干扰物，不参与采样。
    """

    # 当前只对红色目标方块做随机采样。
    return {
        TARGET_CUBE_NAME: np.array(
            [
                # 在一个很小的 x 范围内采样。
                rng.uniform(*TARGET_CUBE_X_RANGE),
                # 在一个很小的 y 范围内采样。
                rng.uniform(*TARGET_CUBE_Y_RANGE),
                # z 高度固定在桌面上。
                TABLE_SURFACE_Z + CUBE_HALF_Z,
            ],
            dtype=np.float32,
        )
    }


def reset_robot(franka: SingleManipulator) -> None:
    """把机器人直接复位到一个统一 home 姿态。"""

    # 直接写 joint positions。
    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    # 同时清掉关节速度，避免上个 episode 的运动惯性残留。
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_cubes(cubes: dict[str, DynamicCuboid], cube_positions: dict[str, np.ndarray]) -> None:
    """把方块统一复位到新位置。"""

    # 遍历所有可控制的方块对象。
    for cube_name, cube in cubes.items():
        # 取该方块本次 episode 应该去的位置。
        cube_position = cube_positions[cube_name]
        # 重设位姿。
        cube.set_world_pose(
            position=cube_position,
            # 单位四元数，表示不旋转。
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        # 清掉速度，避免前一轮残留影响本轮开局。
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))


def settle_scene(world: World, steps: int) -> None:
    """给物理系统一点稳定时间。"""

    # 连续推进若干帧，让方块、机械臂、碰撞体进入稳定状态。
    for _ in range(steps):
        world.step(render=True)


def merge_joint_actions(num_dof: int, *actions: ArticulationAction) -> ArticulationAction:
    """把机械臂动作和夹爪动作合并成一条控制指令。

    典型场景是：
    - RMPFlowController 只给出机械臂关节动作
    - gripper.forward() 只给出夹爪动作

    但 Franka 最终要接收的是一整条 articulation action，所以这里做合并。
    """

    # 先准备三个长度为 `num_dof` 的缓冲区。
    merged_positions: list[float | None] = [None] * num_dof
    merged_velocities: list[float | None] = [None] * num_dof
    merged_efforts: list[float | None] = [None] * num_dof

    # 把每条子动作分别写进缓冲区。
    for action in actions:
        if action is None:
            continue
        _merge_single_field(merged_positions, action.joint_positions, action.joint_indices)
        _merge_single_field(merged_velocities, action.joint_velocities, action.joint_indices)
        _merge_single_field(merged_efforts, action.joint_efforts, action.joint_indices)

    # 最终重新打包成一条完整动作。
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

    # 这个字段根本没有值，就跳过。
    if values is None:
        return

    # 如果没有显式索引，表示 values 默认对应从 0 开始的一整段关节。
    if indices is None:
        for index, value in enumerate(values):
            if value is not None:
                target[index] = float(value)
        return

    # 如果有索引，则按指定 joint_indices 写入。
    for index, value in zip(indices, values):
        if value is not None:
            target[int(index)] = float(value)


def interpolate_positions(
    start_position: np.ndarray,
    end_position: np.ndarray,
    steps: int,
) -> list[np.ndarray]:
    """把一段末端轨迹离散成若干步。

    这里使用 smoothstep 插值，让每一段轨迹的起点和终点速度更柔和。
    它仍然不是完整的轨迹优化，但比线性插值更适合生成平滑示教动作。
    """

    # 不需要插值。
    if steps <= 0:
        return []

    # 统一转成 float32。
    start = np.asarray(start_position, dtype=np.float32)
    end = np.asarray(end_position, dtype=np.float32)
    # 只有一步时，直接返回终点。
    if steps == 1:
        return [end.copy()]

    # 生成一串插值点。
    positions: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        positions.append(((1.0 - smooth_alpha) * start + smooth_alpha * end).astype(np.float32))
    return positions


def build_scripted_actions(
    start_ee_position: np.ndarray,
    target_cube_position: np.ndarray,
) -> list[np.ndarray]:
    """生成一条显式的抓取放置任务空间专家轨迹。

    输出的每一项 action 语义是：

    `[target_ee_x, target_ee_y, target_ee_z, target_gripper_closed]`

    这里不是 joint action，而是“末端目标位置 + 夹爪开关”。
    后续执行时会再通过 RMPFlowController 把它转成机械臂关节动作。
    """

    # 真正抓取点位于目标方块中心之上一个固定偏移。
    pick_position = np.array(
        [
            float(target_cube_position[0]),
            float(target_cube_position[1]),
            float(target_cube_position[2] + EE_PICK_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    # 抓取前，先到两个高度不同的悬停点，让接近动作更细。
    pick_high_hover_position = pick_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)
    pick_low_hover_position = pick_position + np.array([0.0, 0.0, EE_HOVER_MARGIN * 0.45], dtype=np.float32)

    # 放置点同理，位于托盘中心上方一个固定偏移。
    place_position = np.array(
        [
            float(PLACE_GOAL_POSITION[0]),
            float(PLACE_GOAL_POSITION[1]),
            float(PLACE_GOAL_POSITION[2] + EE_PLACE_Z_OFFSET),
        ],
        dtype=np.float32,
    )
    # 放置前也先经过两个高度不同的悬停点。
    place_high_hover_position = place_position + np.array([0.0, 0.0, EE_HOVER_MARGIN], dtype=np.float32)
    place_low_hover_position = place_position + np.array([0.0, 0.0, EE_HOVER_MARGIN * 0.45], dtype=np.float32)
    # 中间转移点抬高一点，降低搬运时撞到托盘或方块的概率。
    transfer_mid_position = np.array(
        [
            (float(pick_high_hover_position[0]) + float(place_high_hover_position[0])) * 0.5,
            (float(pick_high_hover_position[1]) + float(place_high_hover_position[1])) * 0.5,
            max(float(pick_high_hover_position[2]), float(place_high_hover_position[2])) + 0.04,
        ],
        dtype=np.float32,
    )

    # 这里累积整条轨迹。
    actions: list[np.ndarray] = []

    def append_segment(start: np.ndarray, end: np.ndarray, closed: float, steps: int) -> np.ndarray:
        """追加一小段“末端直线插值 + 固定夹爪状态”的动作序列。"""

        last_position = np.asarray(start, dtype=np.float32)
        for position in interpolate_positions(start, end, steps):
            # 每一步 action 由 3D 位置和 1D 夹爪闭合标记拼起来。
            actions.append(
                np.concatenate(
                    [position, np.array([closed], dtype=np.float32)],
                ).astype(np.float32)
            )
            last_position = position
        return last_position

    # 从当前末端起点开始，逐段拼一条完整轨迹。
    current_position = np.asarray(start_ee_position, dtype=np.float32)
    # 1. 移动到抓取高悬停点，夹爪张开。
    current_position = append_segment(current_position, pick_high_hover_position, 0.0, APPROACH_PICK_HIGH_STEPS)
    # 2. 移动到抓取低悬停点，仍保持张开。
    current_position = append_segment(current_position, pick_low_hover_position, 0.0, APPROACH_PICK_LOW_STEPS)
    # 3. 较快下降到接近方块的位置。
    pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.025], dtype=np.float32)
    current_position = append_segment(current_position, pre_grasp_position, 0.0, DESCEND_PICK_FAST_STEPS)
    # 4. 慢速下探到真正抓取高度，给末端和方块对齐更多时间。
    current_position = append_segment(current_position, pick_position, 0.0, DESCEND_PICK_SLOW_STEPS)
    # 5. 抓取前短暂停顿，避免还在移动时立刻关夹爪。
    current_position = append_segment(current_position, pick_position, 0.0, PRE_GRASP_SETTLE_STEPS)
    # 6. 夹爪闭合阶段。
    current_position = append_segment(current_position, pick_position, 1.0, GRASP_CLOSE_STEPS)
    # 7. 闭合后继续保持，让物理接触真正建立。
    current_position = append_segment(current_position, pick_position, 1.0, GRASP_HOLD_STEPS)
    # 8. 慢速抬起一小段，避免刚夹住就大幅移动导致滑落。
    current_position = append_segment(current_position, pick_low_hover_position, 1.0, LIFT_SLOW_STEPS)
    # 9. 抬到高悬停点。
    current_position = append_segment(current_position, pick_high_hover_position, 1.0, LIFT_HIGH_STEPS)
    # 10. 经过中间转移点。
    current_position = append_segment(current_position, transfer_mid_position, 1.0, TRANSFER_MID_STEPS)
    # 11. 到托盘高悬停点。
    current_position = append_segment(current_position, place_high_hover_position, 1.0, TRANSFER_PLACE_STEPS)
    # 12. 下降到托盘低悬停点。
    current_position = append_segment(current_position, place_low_hover_position, 1.0, DESCEND_PLACE_FAST_STEPS)
    # 13. 慢速下降到放置高度。
    current_position = append_segment(current_position, place_position, 1.0, DESCEND_PLACE_SLOW_STEPS)
    # 14. 释放前短暂停顿。
    current_position = append_segment(current_position, place_position, 1.0, PRE_RELEASE_SETTLE_STEPS)
    # 15. 张开夹爪释放。
    current_position = append_segment(current_position, place_position, 0.0, RELEASE_OPEN_STEPS)
    # 16. 保持在放置点，让 cube 自然落稳。
    current_position = append_segment(current_position, place_position, 0.0, POST_RELEASE_SETTLE_STEPS)
    # 17. 再抬起离开。
    append_segment(current_position, place_high_hover_position, 0.0, RETREAT_STEPS)
    return actions


def capture_rgb(camera: Camera) -> np.ndarray:
    """读取一帧 RGB 图像，并统一成 uint8。"""

    # Isaac 相机接口返回 RGB 数据。
    rgb = camera.get_rgb()
    # 如果为空，说明相机没正常出图。
    if rgb is None:
        raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")
    # 强制转成 uint8，后续保存更标准。
    return np.asarray(rgb, dtype=np.uint8)


def get_robot_state(franka: SingleManipulator) -> np.ndarray:
    """读取训练时常用的状态向量。

    输出顺序与 `STATE_NAMES` 一一对应。
    """

    # 取当前所有关节位置。
    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    # 取末端执行器世界位姿。
    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32)
    ee_orientation = np.asarray(ee_orientation, dtype=np.float32)
    # 用两个夹爪手指的关节值之和近似夹爪开口宽度。
    gripper_width = float(joint_positions[7] + joint_positions[8])

    # 拼成最终状态向量。
    return np.concatenate(
        [
            joint_positions[:7],
            ee_position[:3],
            ee_orientation[:4],
            np.array([gripper_width], dtype=np.float32),
        ]
    ).astype(np.float32)


def is_cube_inside_box(cube: DynamicCuboid) -> bool:
    """判断指定方块是否已经被放进托盘。

    这里不用复杂碰撞检测，而是用一个简单几何条件判断：
    - x 在托盘内侧
    - y 在托盘内侧
    - z 没有高得离谱
    """

    # 读取当前方块中心位置。
    cube_position, _ = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)

    # 扣掉墙体厚度后的托盘内部半宽。
    inner_half_x = PLACE_BOX_OUTER_X / 2.0 - PLACE_BOX_WALL_T
    inner_half_y = PLACE_BOX_OUTER_Y / 2.0 - PLACE_BOX_WALL_T
    # 判断 x 是否落在内部区域。
    within_x = abs(float(cube_position[0] - PLACE_BOX_CENTER[0])) < inner_half_x
    # 判断 y 是否落在内部区域。
    within_y = abs(float(cube_position[1] - PLACE_BOX_CENTER[1])) < inner_half_y
    # 判断 z 是否处在合理范围内。
    within_z = float(cube_position[2]) < TABLE_SURFACE_Z + 0.12
    return bool(within_x and within_y and within_z)


def episode_metadata() -> str:
    """生成每个 episode 共享的元数据字符串。

    之所以额外保存为 JSON 字符串，是为了让单个 npz 文件自带结构说明。
    """

    return json.dumps(
        {
            # 数据结构版本。
            "schema_version": 1,
            # 自然语言任务描述。
            "task": TASK_DESCRIPTION,
            # 专家类型，明确这批数据不再使用吸附辅助。
            "expert_type": EXPERT_TYPE,
            # 状态字段名。
            "state_names": STATE_NAMES,
            # 动作字段名。
            "action_names": ACTION_NAMES,
            # 两路图像分辨率。
            "front_camera_resolution": FRONT_CAMERA_RESOLUTION,
            "wrist_camera_resolution": WRIST_CAMERA_RESOLUTION,
            # 每条 episode 最多记录多少步。
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

    # 确保输出目录存在。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 文件名按固定 5 位编号命名。
    file_path = output_dir / f"episode_{episode_index:05d}.npz"

    # 使用压缩格式保存，减少磁盘占用。
    np.savez_compressed(
        file_path,
        **{
            # 两路图像观测。
            "observation.images.front": np.asarray(front_images, dtype=np.uint8),
            "observation.images.wrist": np.asarray(wrist_images, dtype=np.uint8),
            # 机器人状态。
            "observation.state": np.asarray(states, dtype=np.float32),
            # 专家动作。
            "action": np.asarray(actions, dtype=np.float32),
            # reward / done 用 next.* 命名，贴近常见 RL / IL 数据格式。
            "next.reward": np.asarray(rewards, dtype=np.float32),
            "next.done": np.asarray(dones, dtype=np.bool_),
            # 附加说明字段。
            "state_names": np.asarray(STATE_NAMES),
            "action_names": np.asarray(ACTION_NAMES),
            "task": np.asarray(TASK_DESCRIPTION),
            "expert_type": np.asarray(EXPERT_TYPE),
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

    返回：
    - `success`：这一轮是否成功把红色方块放进托盘
    - `save_path`：如果成功且已保存，则返回文件路径；否则返回 `None`
    """

    # 用每个 episode 自己的 seed 生成随机数。
    rng = np.random.default_rng(seed)
    # 采样本轮目标方块位置。
    cube_positions = sample_cube_positions(rng)
    # 取红色目标方块对象。
    target_cube = cubes[TARGET_CUBE_NAME]
    # 机器人归位。
    reset_robot(franka)
    # 方块归位。
    reset_cubes(cubes, cube_positions)
    # 控制器状态清空。
    controller.reset()
    # 让场景先稳定几步。
    settle_scene(world, EPISODE_SETTLE_STEPS)

    # 下面这些列表逐帧积累，最后整条保存。
    front_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    # 是否终止。
    done = False
    # 是否完成任务。
    success = False
    # 读取当前末端起始位置，用来生成专家轨迹。
    start_ee_position, _ = franka.end_effector.get_world_pose()
    start_ee_position = np.asarray(start_ee_position, dtype=np.float32)
    # 根据当前方块位置生成一整条专家任务空间动作序列。
    scripted_actions = build_scripted_actions(
        start_ee_position=start_ee_position,
        target_cube_position=cube_positions[TARGET_CUBE_NAME],
    )

    # 逐帧执行专家动作。
    for task_space_action in scripted_actions[:EPISODE_MAX_STEPS]:
        # 前 3 维是末端目标位置。
        target_position = np.asarray(task_space_action[:3], dtype=np.float32)
        # 第 4 维大于等于 0.5 就视为夹爪闭合。
        gripper_closed = bool(float(task_space_action[3]) >= 0.5)

        # 用 RMPFlowController 把任务空间目标转成机械臂关节动作。
        arm_action = controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=EE_TARGET_ORIENTATION,
        )
        # 生成夹爪动作。
        gripper_action = franka.gripper.forward("close" if gripper_closed else "open")
        # 把机械臂和夹爪动作合并。
        joint_action = merge_joint_actions(franka.num_dof, arm_action, gripper_action)
        # 应用到机器人。
        franka.apply_action(joint_action)
        # 推进仿真一帧。
        world.step(render=True)

        # 记录前视图。
        front_images.append(capture_rgb(front_camera))
        # 记录手腕图。
        wrist_images.append(capture_rgb(wrist_camera))
        # 记录当前状态。
        states.append(get_robot_state(franka))
        # 记录当前专家动作。
        actions.append(task_space_action)

        # 计算当前是否已经完成“方块在托盘里”。
        success = is_cube_inside_box(target_cube)
        # 这里 reward 做得很简单：成功就 1，否则 0。
        rewards.append(1.0 if success else 0.0)
        # 先全记成 False，最后一帧再补 True。
        dones.append(False)

        # 如果用户手动关闭 Isaac Sim，就提前退出。
        if not simulation_app.is_running():
            break

    # 如果至少记录了一帧，就把最后一帧设为 done。
    if dones:
        dones[-1] = True
        done = True

    # 读取最终方块位置，便于打印调试。
    final_cube_position, _ = target_cube.get_world_pose()
    final_cube_position = np.asarray(final_cube_position, dtype=np.float32)

    # 失败轨迹直接丢弃，不保存。
    if not success:
        print(
            f"  red cube final position: {np.round(final_cube_position, 4).tolist()}",
            flush=True,
        )
        return False, None

    # 成功轨迹才写盘保存。
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
    # 打印最终位置，帮助快速确认数据分布。
    print(
        f"  red cube final position: {np.round(final_cube_position, 4).tolist()}",
        flush=True,
    )
    return success, save_path


def main() -> None:
    """脚本主入口。"""

    # 统一把输出目录转成绝对路径。
    output_dir = ARGS.output_dir.resolve()

    # 搭场景。
    world, franka, cubes = build_scene()
    # 初始化机械臂 articulation。
    franka.initialize()
    # 初始化两路相机传感器。
    front_camera, wrist_camera = create_cameras()
    # 再推进几步，让相机和场景更稳定。
    settle_scene(world, 10)

    # 构造 RMPFlowController，用来把末端目标位置转成机械臂动作。
    controller = RMPFlowController(
        name="franka_scripted_rmpflow",
        robot_articulation=franka,
    )

    # 统计已保存成功条数。
    success_count = 0
    # 统计总尝试次数。
    attempt_count = 0
    # 给总尝试次数设一个上限，避免一直失败导致无限循环。
    max_attempts = max(ARGS.episodes * 12, ARGS.episodes + 5)

    # 打印启动信息。
    print(f"输出目录: {output_dir}", flush=True)
    print(f"目标成功 episode 数: {ARGS.episodes}", flush=True)
    print("开始采集，仅保存成功轨迹...", flush=True)

    # 不断尝试，直到成功条数达到目标，或尝试次数超过上限。
    while success_count < ARGS.episodes and attempt_count < max_attempts:
        # 给每次尝试一个可复现 seed。
        episode_seed = 20260604 + attempt_count
        # 采集单条 episode。
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
        # 成功就打印保存路径并累加成功计数。
        if success:
            print(
                f"[attempt {attempt_count:03d}] success "
                f"saved -> {save_path}",
                flush=True,
            )
            success_count += 1
        # 失败就丢弃，只做日志提示。
        else:
            print(f"[attempt {attempt_count:03d}] failed discarded", flush=True)
        # 总尝试次数每轮都增加。
        attempt_count += 1

        # 如果 Isaac Sim 被关闭，就提前退出 while。
        if not simulation_app.is_running():
            print("检测到 Isaac Sim 已关闭，提前结束采集。", flush=True)
            break

    # 最终统计信息。
    print(f"采集完成：成功保存 {success_count} / 目标 {ARGS.episodes}，总尝试 {attempt_count}。", flush=True)
    # 如果没采满目标数量，就给出警告。
    if success_count < ARGS.episodes:
        print("警告：未达到目标成功条数，请再次运行采集。", flush=True)


if __name__ == "__main__":
    try:
        # 直接运行文件时进入主流程。
        main()
    except Exception:
        # 出错时打印完整 traceback，方便定位 Isaac 扩展或数据采集逻辑问题。
        traceback.print_exc()
        raise
    finally:
        # 无论成功还是失败，都要关闭 SimulationApp。
        simulation_app.close()
