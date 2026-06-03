"""Franka SmolVLA 数据采集示例。

这个脚本会在 Isaac Sim 中创建一个 Franka 桌面抓取场景，并用内置的
PickPlaceController 作为“专家策略”自动采集示教数据。

采集内容包括：
1. 前视相机图像
2. 手腕相机图像
3. 机器人状态（7 关节、末端位置、夹爪开合宽度）
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
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
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

CUBE_NAME = "training_cube"
CUBE_SIZE = np.array([0.055, 0.055, 0.055], dtype=np.float32)
CUBE_COLOR = np.array([0.10, 0.45, 0.90], dtype=np.float32)
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)

# 这个范围专门压在 Franka 正前方偏中间区域，减少专家动作失败率。
CUBE_X_RANGE = (0.36, 0.50)
CUBE_Y_RANGE = (-0.10, 0.10)

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

EPISODE_MAX_STEPS = 360
EPISODE_SETTLE_STEPS = 20
EPISODE_POST_SUCCESS_STEPS = 20

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
    "gripper_width",
]

ACTION_NAMES = [
    "target_ee_pos_x",
    "target_ee_pos_y",
    "target_ee_pos_z",
    "target_gripper_closed",
]

TASK_DESCRIPTION = "Pick up the blue cube with Franka and place it into the wooden tray."


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


def add_training_cube(world: World) -> DynamicCuboid:
    """添加被抓取的训练方块。"""

    return world.scene.add(
        DynamicCuboid(
            name=CUBE_NAME,
            prim_path=f"/World/{CUBE_NAME}",
            position=np.array([0.42, 0.00, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
            scale=CUBE_SIZE,
            size=1.0,
            color=CUBE_COLOR,
        )
    )


def build_scene() -> tuple[World, SingleManipulator, DynamicCuboid]:
    """构建完整场景。"""

    world = World(stage_units_in_meters=1.0)
    create_lights()
    add_room(world)
    world.scene.add_default_ground_plane()
    add_table(world)
    add_place_box(world)

    franka = add_franka(world)
    cube = add_training_cube(world)

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

    return world, franka, cube


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


def sample_cube_position(rng: np.random.Generator) -> np.ndarray:
    """在 Franka 可稳定抓取的范围里随机采样方块位置。"""

    return np.array(
        [
            rng.uniform(*CUBE_X_RANGE),
            rng.uniform(*CUBE_Y_RANGE),
            TABLE_SURFACE_Z + CUBE_HALF_Z,
        ],
        dtype=np.float32,
    )


def reset_robot(franka: SingleManipulator) -> None:
    """把机器人直接复位到一个统一 home 姿态。"""

    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_cube(cube: DynamicCuboid, cube_position: np.ndarray) -> None:
    """把训练方块复位到新的随机位置。"""

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


def smoothstep(alpha: float) -> float:
    """和 PickPlaceController 内部保持一致的平滑插值。"""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(alpha * np.pi))


def infer_task_space_action(
    controller: PickPlaceController,
    picking_position: np.ndarray,
    placing_position: np.ndarray,
    previous_target_position: np.ndarray,
) -> np.ndarray:
    """把 Isaac 专家控制器的内部阶段，转成任务空间训练真值。

    这里不再输出关节目标，而是输出更适合 VLA 学习的：
    1. 末端目标 xyz
    2. 夹爪开合状态

    之所以单独做这层转换，是因为 `PickPlaceController.forward()` 返回的是
    关节空间动作，但你后续真正想学的是“末端应该去哪里、夹爪该开还是关”。
    """

    if controller.is_done():
        return np.concatenate(
            [np.asarray(previous_target_position, dtype=np.float32), np.array([0.0], dtype=np.float32)]
        ).astype(np.float32)

    event = controller.get_current_event()
    phase_t = float(getattr(controller, "_t", 0.0))
    pick_position = np.asarray(picking_position, dtype=np.float32)
    place_position = np.asarray(placing_position, dtype=np.float32)
    target_position = np.asarray(previous_target_position, dtype=np.float32).copy()

    if event in (0, 1):
        start_xy = pick_position[:2]
        pick_height = float(pick_position[2])
    else:
        start_xy = np.array(
            [
                float(getattr(controller, "_current_target_x", previous_target_position[0])),
                float(getattr(controller, "_current_target_y", previous_target_position[1])),
            ],
            dtype=np.float32,
        )
        pick_height = float(getattr(controller, "_h0", previous_target_position[2]))

    hover_height = float(getattr(controller, "_h1", previous_target_position[2]))
    place_height = float(place_position[2])

    if event < 5:
        xy_alpha = 0.0
    elif event == 5:
        xy_alpha = smoothstep(phase_t)
    else:
        xy_alpha = 1.0

    if event == 0:
        target_height = hover_height
    elif event == 1:
        target_height = (1.0 - smoothstep(phase_t)) * hover_height + smoothstep(phase_t) * pick_height
    elif event in (2, 3):
        target_height = pick_height
    elif event == 4:
        target_height = (1.0 - smoothstep(phase_t)) * pick_height + smoothstep(phase_t) * hover_height
    elif event == 5:
        target_height = hover_height
    elif event == 6:
        target_height = (1.0 - smoothstep(phase_t)) * hover_height + smoothstep(phase_t) * place_height
    elif event == 7:
        target_height = place_height
    elif event == 8:
        target_height = (1.0 - smoothstep(phase_t)) * place_height + smoothstep(phase_t) * hover_height
    else:
        target_height = hover_height

    if event not in (2, 3, 7):
        target_xy = (1.0 - xy_alpha) * start_xy + xy_alpha * place_position[:2]
        target_position = np.array(
            [float(target_xy[0]), float(target_xy[1]), float(target_height)],
            dtype=np.float32,
        )

    gripper_closed = 1.0 if event in (3, 4, 5, 6) else 0.0
    return np.concatenate([target_position, np.array([gripper_closed], dtype=np.float32)]).astype(np.float32)


def capture_rgb(camera: Camera) -> np.ndarray:
    """读取一帧 RGB 图像，并统一成 uint8。"""

    rgb = camera.get_rgb()
    if rgb is None:
        raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")
    return np.asarray(rgb, dtype=np.uint8)


def get_robot_state(franka: SingleManipulator) -> np.ndarray:
    """读取训练时常用的状态向量。"""

    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    ee_position, _ = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32)
    gripper_width = float(joint_positions[7] + joint_positions[8])

    return np.concatenate(
        [
            joint_positions[:7],
            ee_position[:3],
            np.array([gripper_width], dtype=np.float32),
        ]
    ).astype(np.float32)


def is_cube_inside_box(cube: DynamicCuboid) -> bool:
    """判断方块是否已经被放进托盘。"""

    cube_position, _ = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)

    inner_half_x = PLACE_BOX_OUTER_X / 2.0 - PLACE_BOX_WALL_T
    inner_half_y = PLACE_BOX_OUTER_Y / 2.0 - PLACE_BOX_WALL_T
    within_x = abs(float(cube_position[0] - PLACE_BOX_CENTER[0])) < inner_half_x
    within_y = abs(float(cube_position[1] - PLACE_BOX_CENTER[1])) < inner_half_y
    within_z = float(cube_position[2]) < TABLE_SURFACE_Z + 0.12
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
    cube: DynamicCuboid,
    front_camera: Camera,
    wrist_camera: Camera,
    controller: PickPlaceController,
    episode_index: int,
    output_dir: Path,
    seed: int,
) -> tuple[bool, Path]:
    """采集单个 episode。"""

    rng = np.random.default_rng(seed)
    cube_position = sample_cube_position(rng)
    reset_robot(franka)
    reset_cube(cube, cube_position)
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
    extra_steps_after_done = EPISODE_POST_SUCCESS_STEPS
    previous_target_position = np.asarray(franka.end_effector.get_world_pose()[0], dtype=np.float32)

    for _step in range(EPISODE_MAX_STEPS):
        current_joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        current_cube_position, _ = cube.get_world_pose()
        current_cube_position = np.asarray(current_cube_position, dtype=np.float32)

        task_space_action = infer_task_space_action(
            controller=controller,
            picking_position=current_cube_position,
            placing_position=PLACE_GOAL_POSITION,
            previous_target_position=previous_target_position,
        )
        control_action = controller.forward(
            picking_position=current_cube_position,
            placing_position=PLACE_GOAL_POSITION,
            current_joint_positions=current_joint_positions,
        )
        previous_target_position = task_space_action[:3].copy()
        franka.apply_action(control_action)
        world.step(render=True)

        front_images.append(capture_rgb(front_camera))
        wrist_images.append(capture_rgb(wrist_camera))
        states.append(get_robot_state(franka))
        actions.append(task_space_action)

        success = is_cube_inside_box(cube)
        rewards.append(1.0 if success else 0.0)

        controller_finished = controller.is_done()
        last_frame = controller_finished and extra_steps_after_done <= 0
        dones.append(last_frame)

        if controller_finished:
            extra_steps_after_done -= 1

        if last_frame:
            done = True
            break

        if not simulation_app.is_running():
            break

    if not done and dones:
        dones[-1] = True

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
    return success, save_path


def main() -> None:
    """脚本主入口。"""

    output_dir = ARGS.output_dir.resolve()

    world, franka, cube = build_scene()
    franka.initialize()
    front_camera, wrist_camera = create_cameras()
    settle_scene(world, 10)

    controller = PickPlaceController(
        name="franka_pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
        end_effector_initial_height=0.22,
    )

    success_count = 0
    collected_count = 0

    print(f"输出目录: {output_dir}")
    print(f"计划采集 episode 数: {ARGS.episodes}")
    print("开始采集...")

    for episode_index in range(ARGS.episodes):
        episode_seed = 20260603 + episode_index
        success, save_path = collect_episode(
            world=world,
            franka=franka,
            cube=cube,
            front_camera=front_camera,
            wrist_camera=wrist_camera,
            controller=controller,
            episode_index=episode_index,
            output_dir=output_dir,
            seed=episode_seed,
        )
        collected_count += 1
        success_count += int(success)
        print(
            f"[episode {episode_index:03d}] "
            f"{'success' if success else 'failed '} "
            f"saved -> {save_path}"
        )

        if not simulation_app.is_running():
            print("检测到 Isaac Sim 已关闭，提前结束采集。")
            break

    print(f"采集完成：成功 {success_count} / 实际采集 {collected_count}。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
