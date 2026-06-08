"""在 Isaac Sim 中回放 17_demo 采集得到的原始 NPZ 轨迹。

这个脚本不是播放保存下来的图像，而是：

1. 重新搭建与采集时一致的 Franka 场景
2. 按 episode 里的动作序列逐帧驱动 Franka
3. 在 Isaac Sim 中重新执行整条轨迹

当前支持回放的动作语义是：

    [target_ee_pos_x, target_ee_pos_y, target_ee_pos_z, target_gripper_closed]

也就是：
1. 末端目标位置
2. 夹爪闭合标记

运行示例：

    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py
    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py --episode 1
    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py --headless
"""

from __future__ import annotations

import argparse
import re
import time
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析回放参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "raw",
        help="原始 npz episode 所在目录。",
    )
    parser.add_argument("--episode", type=int, default=0, help="回放哪一个 episode。默认回放 episode_00000.npz")
    parser.add_argument("--fps", type=float, default=20.0, help="回放节奏。")
    parser.add_argument("--headless", action="store_true", help="无界面运行。")
    parser.add_argument("--loop", action="store_true", help="循环回放。")
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


# 这部分场景配置与 demo.py 保持一致，保证“采集时”和“回放时”环境尽量一致。
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

CUBE_SIZE = np.array([0.045, 0.045, 0.045], dtype=np.float32)
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)

TARGET_CUBE_NAME = "cube_red"
TARGET_CUBE_COLOR = np.array([0.88, 0.15, 0.15], dtype=np.float32)
DISTRACTOR_CUBE_SPECS = [
    ("cube_green", np.array([0.18, 0.66, 0.24], dtype=np.float32)),
    ("cube_blue", np.array([0.12, 0.36, 0.86], dtype=np.float32)),
]

TARGET_CUBE_X_RANGE = (0.42, 0.44)
TARGET_CUBE_Y_RANGE = (-0.02, 0.02)

DISTRACTOR_CUBE_LAYOUT = {
    "cube_green": np.array([0.34, 0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
    "cube_blue": np.array([0.34, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
}

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

HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)

EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))
EE_PICK_Z_OFFSET = 0.092
EE_FEEDBACK_Z_BIAS = 0.0985
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
ATTACH_POSITION_BLEND = 0.35
PLACE_SETTLE_STEPS = 20


def create_camera_prim(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """在 stage 中创建相机 prim。"""

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    """创建光照。"""

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
    """添加房间包围物。"""

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
    """添加托盘。"""

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
    """把 Franka 加入场景。"""

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
    """添加与录制端一致的目标方块和干扰方块。"""

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
    """构建与录制端一致的回放场景。"""

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
    """初始化相机。

    这里保留相机初始化，主要是为了让场景结构与采集时一致。
    回放本身不依赖读取图像，但初始化相机后更利于从界面观察。
    """

    front_camera = Camera(prim_path=FRONT_CAMERA_PATH, name="front_camera", resolution=FRONT_CAMERA_RESOLUTION)
    wrist_camera = Camera(prim_path=WRIST_CAMERA_PATH, name="wrist_camera", resolution=WRIST_CAMERA_RESOLUTION)
    front_camera.initialize()
    wrist_camera.initialize()
    return front_camera, wrist_camera


def sample_cube_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """按与录制时相同的规则重建红色目标方块初始位置。"""

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
    """把机器人恢复到采集起点姿态。"""

    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_cubes(cubes: dict[str, DynamicCuboid], cube_positions: dict[str, np.ndarray]) -> None:
    """把需要回放的动态方块放回 episode 起始位置。"""

    for cube_name, cube in cubes.items():
        cube_position = cube_positions[cube_name]
        cube.set_world_pose(
            position=cube_position,
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))


def settle_scene(world: World, steps: int = 20) -> None:
    """给物理系统少量稳定时间。"""

    for _ in range(steps):
        world.step(render=True)


def merge_joint_actions(num_dof: int, *actions: ArticulationAction) -> ArticulationAction:
    """把多个关节动作合并成一个动作。

    这样做的目的是：
    1. RMPFlow 输出的是机械臂动作
    2. gripper.forward 输出的是夹爪动作
    3. 回放时我们希望同一帧里同时施加这两部分控制
    """

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
    """把一个动作字段填入合并缓冲区。"""

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


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算 XY 平面距离。"""

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
    """按 XY 和 Z 分开判断是否真正到位。"""

    return bool(
        planar_distance(current_position, target_position) <= xy_threshold
        and vertical_distance(current_position, target_position) <= z_threshold
    )


def get_task_space_ee_pose(franka: SingleManipulator) -> tuple[np.ndarray, np.ndarray]:
    """读取与录制动作同一参考系下的末端位姿。"""

    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32).copy()
    ee_position[2] -= EE_FEEDBACK_Z_BIAS
    ee_orientation = np.asarray(ee_orientation, dtype=np.float32)
    return ee_position, ee_orientation


def episode_index_from_path(path: Path) -> int:
    """从 episode 文件名里解析数字编号。"""

    match = re.search(r"episode_(\d+)\.npz$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def resolve_episode_path(raw_dir: Path, episode: int) -> Path:
    """解析要回放的 episode 文件路径。"""

    path = raw_dir / f"episode_{episode:05d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")
    return path


def reset_episode_from_npz(
    world: World,
    franka: SingleManipulator,
    cubes: dict[str, DynamicCuboid],
    episode_data: np.lib.npyio.NpzFile,
    episode_path: Path,
) -> int:
    """根据 npz 里的 seed 重建 episode 起始场景。"""

    if "episode_seed" in episode_data.files:
        seed = int(episode_data["episode_seed"])
    else:
        seed = 20260603 + episode_index_from_path(episode_path)

    rng = np.random.default_rng(seed)
    cube_positions = sample_cube_positions(rng)
    reset_robot(franka)
    reset_cubes(cubes, cube_positions)
    settle_scene(world, steps=20)
    return seed


def replay_episode(
    world: World,
    franka: SingleManipulator,
    cubes: dict[str, DynamicCuboid],
    controller: RMPFlowController,
    episode_path: Path,
    fps: float,
) -> None:
    """在 Isaac Sim 中回放单条 episode。

    注意：
    当前 raw npz 保存的是“动作真值”，不是每一帧完整物理世界状态。
    所以这里是“动作重放”，不是像视频那样的严格逐帧状态复刻。
    """

    data = np.load(episode_path, allow_pickle=True)
    actions = np.asarray(data["action"], dtype=np.float32)
    done_flags = np.asarray(data["next.done"], dtype=np.bool_)
    task = str(np.asarray(data["task"]).item())
    seed = reset_episode_from_npz(world, franka, cubes, data, episode_path)
    controller.reset()
    target_cube = cubes[TARGET_CUBE_NAME]
    cube_attached = False
    attach_offset = np.array([0.0, 0.0, -EE_PICK_Z_OFFSET], dtype=np.float32)
    placement_step_index = 0
    placement_start_position = PLACE_GOAL_POSITION.copy()

    print(f"replay: {episode_path.name}", flush=True)
    print(f"  task: {task}", flush=True)
    print(f"  seed: {seed}", flush=True)
    print(f"  frames: {len(actions)}", flush=True)

    timestep = 1.0 / max(fps, 1e-6)
    start = time.perf_counter()

    for frame_idx, action in enumerate(actions):
        if not simulation_app.is_running():
            break

        target_position = np.asarray(action[:3], dtype=np.float32)
        gripper_closed = bool(float(action[3]) >= 0.5)

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
        if gripper_closed:
            ee_reached_target = is_position_close(
                ee_position,
                target_position,
                EE_PICK_REACH_XY_THRESHOLD,
                EE_PICK_REACH_Z_THRESHOLD,
            )
        else:
            ee_reached_target = is_position_close(
                ee_position,
                target_position,
                EE_PLACE_REACH_XY_THRESHOLD,
                EE_PLACE_REACH_Z_THRESHOLD,
            )

        if gripper_closed and not cube_attached:
            cube_ready_to_attach = is_position_close(
                cube_position,
                desired_cube_position,
                PICK_ATTACH_XY_THRESHOLD,
                PICK_ATTACH_Z_THRESHOLD,
            )
            if ee_reached_target and cube_ready_to_attach and gripper_width <= GRIPPER_CLOSE_WIDTH_THRESHOLD:
                attach_offset = np.array([0.0, 0.0, -EE_PICK_Z_OFFSET], dtype=np.float32)
                cube_attached = True
                placement_step_index = 0

        if cube_attached and gripper_closed:
            smoothed_cube_position = (
                (1.0 - ATTACH_POSITION_BLEND) * cube_position + ATTACH_POSITION_BLEND * desired_cube_position
            ).astype(np.float32)
            target_cube.set_world_pose(
                position=smoothed_cube_position,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
            target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        elif cube_attached and not gripper_closed:
            cube_ready_to_release = is_position_close(
                desired_cube_position,
                PLACE_GOAL_POSITION,
                PLACE_RELEASE_XY_THRESHOLD,
                PLACE_RELEASE_Z_THRESHOLD,
            )
            carried_cube_position = (
                (1.0 - ATTACH_POSITION_BLEND) * cube_position + ATTACH_POSITION_BLEND * desired_cube_position
            ).astype(np.float32)
            if ee_reached_target and cube_ready_to_release:
                if placement_step_index == 0:
                    placement_start_position = carried_cube_position.copy()
                    placement_step_index = 1
                alpha = min(placement_step_index / float(PLACE_SETTLE_STEPS), 1.0)
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                settled_position = (
                    (1.0 - smooth_alpha) * placement_start_position + smooth_alpha * PLACE_GOAL_POSITION
                ).astype(np.float32)
                target_cube.set_world_pose(
                    position=settled_position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
                if placement_step_index < PLACE_SETTLE_STEPS:
                    placement_step_index += 1
                else:
                    placement_step_index = 0
                    cube_attached = False
            else:
                placement_step_index = 0
                target_cube.set_world_pose(
                    position=carried_cube_position,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
                target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
                target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        elif placement_step_index > 0:
            alpha = min(placement_step_index / float(PLACE_SETTLE_STEPS), 1.0)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            settled_position = (
                (1.0 - smooth_alpha) * placement_start_position + smooth_alpha * PLACE_GOAL_POSITION
            ).astype(np.float32)
            target_cube.set_world_pose(
                position=settled_position,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            target_cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
            target_cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
            if placement_step_index < PLACE_SETTLE_STEPS:
                placement_step_index += 1
            else:
                placement_step_index = 0

        target_time = start + (frame_idx + 1) * timestep
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0 and not ARGS.headless:
            time.sleep(sleep_time)

        if frame_idx < len(done_flags) and bool(done_flags[frame_idx]):
            break

    cube_position, _ = target_cube.get_world_pose()
    ee_position, _ = franka.end_effector.get_world_pose()
    print(
        "  final:"
        f" cube={np.round(np.asarray(cube_position, dtype=np.float32), 4).tolist()}"
        f" ee={np.round(np.asarray(ee_position, dtype=np.float32), 4).tolist()}",
        flush=True,
    )


def main() -> None:
    """脚本主入口。"""

    episode_path = resolve_episode_path(ARGS.raw_dir.resolve(), ARGS.episode)
    world, franka, cubes = build_scene()
    franka.initialize()
    create_cameras()
    settle_scene(world, steps=10)

    controller = RMPFlowController(name="franka_replay_rmpflow", robot_articulation=franka)

    try:
        while simulation_app.is_running():
            print("scene ready, start replay", flush=True)
            replay_episode(world, franka, cubes, controller, episode_path, ARGS.fps)
            if not ARGS.loop:
                break
    finally:
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
