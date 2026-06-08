"""Build a multi-station Isaac Sim scene with 4-6 replicated Franka workcells.

Run with Isaac Sim's Python environment:

    python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py

这个示例是在 `build_franka_scene_only.py` 的基础上扩展出来的：

1. 每个工位都复制同样的 Franka + 桌子 + 方块 + 托盘 + 双相机
2. 工位数量支持 4 / 5 / 6
3. 通过统一的网格布局把多个工位排开，减少机械臂之间的碰撞风险
4. 每个工位保留独立 prim 路径，后续接数据采集脚本时更方便逐个索引
"""

from __future__ import annotations

import argparse
import math
import traceback

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析必要参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无界面运行，只验证场景是否能成功构建。")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        choices=(4, 5, 6),
        help="创建多少个复制工位，推荐先从 4 个开始。",
    )
    return parser.parse_args()


ARGS = parse_args()

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
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux


TABLE_H = 0.40
TABLE_CENTER_LOCAL = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H

FRONT_CAMERA_EYE_LOCAL = np.array([1.15, -1.10, 1.10], dtype=np.float32)
FRONT_CAMERA_TARGET_LOCAL = np.array([0.40, 0.0, 0.65], dtype=np.float32)
FRONT_CAMERA_ROTATION = (-35.0, 0.0, 45.0)
WRIST_CAMERA_LOCAL_POS = (0.06, 0.0, 0.03)
WRIST_CAMERA_LOCAL_ROT = (-95.0, 0.0, -90.0)

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


def translated(origin: np.ndarray, local_xyz: np.ndarray) -> np.ndarray:
    """把工位局部坐标转换成世界坐标。"""

    return np.array(
        [origin[0] + local_xyz[0], origin[1] + local_xyz[1], local_xyz[2]],
        dtype=np.float32,
    )


def env_layout(num_envs: int) -> list[np.ndarray]:
    """生成居中网格布局。

    - 4 个工位：2 x 2
    - 5 / 6 个工位：3 x 2
    """

    columns = 2 if num_envs <= 4 else 3
    rows = int(math.ceil(num_envs / columns))

    y_offsets = ((rows - 1) / 2.0 - np.arange(rows, dtype=np.float32)) * ENV_SPACING_Y

    origins: list[np.ndarray] = []
    for row_index in range(rows):
        remaining = num_envs - len(origins)
        row_count = min(columns, remaining)
        x_offsets = (np.arange(row_count, dtype=np.float32) - (row_count - 1) / 2.0) * ENV_SPACING_X
        for column_index in range(row_count):
            if len(origins) >= num_envs:
                return origins
            origins.append(np.array([x_offsets[column_index], y_offsets[row_index]], dtype=np.float32))
    return origins


def env_root(env_index: int) -> str:
    """返回某个工位的 prim 根路径。"""

    return f"/World/Envs/env_{env_index:02d}"


def create_camera(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """在当前 USD stage 中创建相机 prim。"""

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights(origins: list[np.ndarray]) -> None:
    """创建环境光和每个工位各自的顶灯。"""

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
    """根据工位布局自适应生成一个更大的房间外壳。"""

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
    """为某个工位添加桌子。"""

    root = env_root(env_index)
    world.scene.add(
        FixedCuboid(
            name=f"table_{env_index:02d}",
            prim_path=f"{root}/Table",
            position=translated(origin, TABLE_CENTER_LOCAL),
            scale=TABLE_SIZE,
            size=1.0,
            color=np.array([0.55, 0.35, 0.15], dtype=np.float32),
        )
    )


def add_place_box(world: World, env_index: int, origin: np.ndarray) -> None:
    """为某个工位添加放置盒。"""

    root = env_root(env_index)
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)

    world.scene.add(
        FixedCuboid(
            name=f"place_box_bottom_{env_index:02d}",
            prim_path=f"{root}/PlaceBox/Bottom",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1], bottom_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_left_{env_index:02d}",
            prim_path=f"{root}/PlaceBox/WallLeft",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] - PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_right_{env_index:02d}",
            prim_path=f"{root}/PlaceBox/WallRight",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] + PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_front_{env_index:02d}",
            prim_path=f"{root}/PlaceBox/WallFront",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name=f"place_box_wall_back_{env_index:02d}",
            prim_path=f"{root}/PlaceBox/WallBack",
            position=np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )


def add_franka(world: World, env_index: int, origin: np.ndarray) -> SingleManipulator:
    """在指定工位加入一台 Franka。"""

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    root = env_root(env_index)
    franka_prim_path = f"{root}/Franka"
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=franka_prim_path)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{franka_prim_path}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path=franka_prim_path,
            name=f"franka_{env_index:02d}",
            end_effector_prim_path=f"{franka_prim_path}/panda_hand",
            gripper=gripper,
            position=np.array([origin[0], origin[1], TABLE_H], dtype=np.float32),
        )
    )
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_cubes(world: World, env_index: int, origin: np.ndarray) -> list[DynamicCuboid]:
    """为某个工位复制一组动态方块。"""

    root = env_root(env_index)
    cubes: list[DynamicCuboid] = []
    for cube_name, local_position, scale, color in CUBES:
        cubes.append(
            world.scene.add(
                DynamicCuboid(
                    name=f"{cube_name}_{env_index:02d}",
                    prim_path=f"{root}/{cube_name}",
                    position=translated(origin, local_position),
                    scale=scale,
                    size=1.0,
                    color=color,
                )
            )
        )
    return cubes


def add_station_cameras(env_index: int, origin: np.ndarray) -> tuple[str, str]:
    """为某个工位创建前视相机和手腕相机 prim。"""

    root = env_root(env_index)
    franka_prim_path = f"{root}/Franka"
    front_camera_path = f"{root}/front_camera"
    wrist_camera_path = f"{franka_prim_path}/panda_hand/wrist_camera"

    create_camera(
        path=front_camera_path,
        position=tuple(translated(origin, FRONT_CAMERA_EYE_LOCAL).tolist()),
        rotation_xyz_deg=FRONT_CAMERA_ROTATION,
        focal_length=10.0,
    )
    create_camera(
        path=wrist_camera_path,
        position=WRIST_CAMERA_LOCAL_POS,
        rotation_xyz_deg=WRIST_CAMERA_LOCAL_ROT,
        focal_length=4.0,
    )
    return front_camera_path, wrist_camera_path


def overview_camera_pose(origins: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """根据工位范围生成一个房间内的总览视角。

    之前把相机放在房间外侧，打开 UI 后第一眼会看到墙外。
    这里改成放在房间内部前上方，俯视四个操作台。
    """

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

    # 把相机放在房间内部靠前的位置，并稍微偏右上方俯视整个工作区。
    # 这样 Isaac Sim 打开后会直接看到完整工位，而不是先看到房间外墙。
    eye = np.array(
        [
            center_x + min(1.20, room_size_x * 0.16),
            min_y + min(1.10, room_size_y * 0.20),
            2.55,
        ],
        dtype=np.float32,
    )
    return eye, target


def build_scene() -> tuple[World, list[SingleManipulator], list[list[DynamicCuboid]], list[tuple[str, str]]]:
    """构建多工位场景。"""

    origins = env_layout(ARGS.num_envs)
    world = World(stage_units_in_meters=1.0)

    create_lights(origins)
    add_room(world, origins)
    world.scene.add_default_ground_plane()

    robots: list[SingleManipulator] = []
    cubes_per_env: list[list[DynamicCuboid]] = []
    camera_paths: list[tuple[str, str]] = []

    for env_index, origin in enumerate(origins):
        add_table(world, env_index, origin)
        add_place_box(world, env_index, origin)
        robots.append(add_franka(world, env_index, origin))
        cubes_per_env.append(add_cubes(world, env_index, origin))
        camera_paths.append(add_station_cameras(env_index, origin))

    world.reset()

    for env_index, origin in enumerate(origins):
        front_camera_path, _ = camera_paths[env_index]
        set_camera_view(
            eye=translated(origin, FRONT_CAMERA_EYE_LOCAL),
            target=translated(origin, FRONT_CAMERA_TARGET_LOCAL),
            camera_prim_path=front_camera_path,
        )

    if not ARGS.headless:
        overview_eye, overview_target = overview_camera_pose(origins)
        set_camera_view(
            eye=overview_eye,
            target=overview_target,
            camera_prim_path="/OmniverseKit_Persp",
        )

    print(f"Built {len(origins)} Franka workcells.")
    for env_index, (front_camera_path, wrist_camera_path) in enumerate(camera_paths):
        print(f"[env_{env_index:02d}] robot={env_root(env_index)}/Franka")
        print(f"[env_{env_index:02d}] front_camera={front_camera_path}")
        print(f"[env_{env_index:02d}] wrist_camera={wrist_camera_path}")

    return world, robots, cubes_per_env, camera_paths


def main() -> None:
    """脚本主入口。"""

    try:
        world, _, _, _ = build_scene()
        if ARGS.headless:
            return

        world.play()
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
