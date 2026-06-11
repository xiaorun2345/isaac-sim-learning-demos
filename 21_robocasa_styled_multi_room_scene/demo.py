"""Build a multi-room Franka showroom with RoboCasa-inspired decorations.

Run with Isaac Sim's Python environment:

    python isaac-sim-learning-demos/21_robocasa_styled_multi_room_scene/demo.py

这个 demo 建立在 `19_franka_multi_station_scene` 之上，但重点从“复制工位”
切到“多场景装修展示”：

1. 每个工位仍然保留 Franka、桌子、方块、托盘和双相机
2. 4 / 5 / 6 个工位会被放进不同装修主题的小房间里
3. 房间风格参考 RoboCasa 常见的厨房 / 家居配色语言
4. 如果本机提供了 RoboCasa 的本地 USD 资产目录，还会尝试把一部分装饰件挂进去
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无界面运行，只验证场景是否能成功构建。")
    parser.add_argument("--num-envs", type=int, default=4, choices=(4, 5, 6), help="创建多少个装修工位。")
    parser.add_argument(
        "--robocasa-asset-root",
        type=Path,
        default=None,
        help="可选的本地 RoboCasa USD 资产目录。存在时会尝试挂一部分装饰件。",
    )
    return parser.parse_args()


ARGS = parse_args()

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1800,
        "height": 960,
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

from common.robocasa_scene_style import (
    add_optional_robocasa_assets,
    add_styled_room_shell,
    add_styled_workcell_decor,
    add_table_finish,
    get_theme,
    theme_names,
)


TABLE_H = 0.40
TABLE_CENTER_LOCAL = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H
ROOM_CENTER_LOCAL = np.array([0.55, 0.0], dtype=np.float32)
ROOM_SIZE_LOCAL = np.array([3.15, 2.85], dtype=np.float32)

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

ENV_SPACING_X = 4.25
ENV_SPACING_Y = 3.55
WORKCELL_HALF_EXTENT_X = 1.70
WORKCELL_HALF_EXTENT_Y = 1.55
SHOWROOM_THEME_SEQUENCE = theme_names()


def translated(origin: np.ndarray, local_xyz: np.ndarray) -> np.ndarray:
    return np.array([origin[0] + local_xyz[0], origin[1] + local_xyz[1], local_xyz[2]], dtype=np.float32)


def env_layout(num_envs: int) -> list[np.ndarray]:
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


def env_root(env_index: int) -> str:
    return f"/World/Envs/env_{env_index:02d}"


def create_camera(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights(origins: list[np.ndarray]) -> None:
    stage = get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(920.0)

    ambient = UsdLux.RectLight.Define(stage, "/World/Lights/CentralFill")
    ambient.CreateIntensityAttr(2600.0)
    ambient.CreateWidthAttr(12.0)
    ambient.CreateHeightAttr(8.0)
    xform = UsdGeom.XformCommonAPI(ambient.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.4, 0.0, 4.0))
    xform.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    for env_index, origin in enumerate(origins):
        key = UsdLux.RectLight.Define(stage, f"/World/Lights/EnvKey_{env_index:02d}")
        key.CreateIntensityAttr(3200.0)
        key.CreateWidthAttr(1.8)
        key.CreateHeightAttr(1.2)
        key_xform = UsdGeom.XformCommonAPI(key.GetPrim())
        key_xform.SetTranslate(Gf.Vec3d(float(origin[0] + 0.72), float(origin[1] - 0.28), 2.20))
        key_xform.SetRotate(Gf.Vec3f(-66.0, 0.0, 58.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_showroom_shell(world: World, origins: list[np.ndarray]) -> None:
    x_values = np.array([origin[0] for origin in origins], dtype=np.float32)
    y_values = np.array([origin[1] for origin in origins], dtype=np.float32)
    min_x = float(np.min(x_values) - WORKCELL_HALF_EXTENT_X - 0.90)
    max_x = float(np.max(x_values) + WORKCELL_HALF_EXTENT_X + 0.90)
    min_y = float(np.min(y_values) - WORKCELL_HALF_EXTENT_Y - 0.90)
    max_y = float(np.max(y_values) + WORKCELL_HALF_EXTENT_Y + 0.90)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    room_size_x = max_x - min_x
    room_size_y = max_y - min_y

    world.scene.add(
        FixedCuboid(
            name="showroom_floor",
            prim_path="/World/Showroom/Floor",
            position=np.array([center_x, center_y, -0.05], dtype=np.float32),
            scale=np.array([room_size_x, room_size_y, 0.10], dtype=np.float32),
            size=1.0,
            color=np.array([0.18, 0.18, 0.19], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="showroom_back_wall",
            prim_path="/World/Showroom/BackWall",
            position=np.array([center_x, max_y, 1.55], dtype=np.float32),
            scale=np.array([room_size_x, 0.06, 3.1], dtype=np.float32),
            size=1.0,
            color=np.array([0.23, 0.24, 0.25], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="showroom_left_wall",
            prim_path="/World/Showroom/LeftWall",
            position=np.array([min_x, center_y, 1.55], dtype=np.float32),
            scale=np.array([0.06, room_size_y, 3.1], dtype=np.float32),
            size=1.0,
            color=np.array([0.23, 0.24, 0.25], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="showroom_right_wall",
            prim_path="/World/Showroom/RightWall",
            position=np.array([max_x, center_y, 1.55], dtype=np.float32),
            scale=np.array([0.06, room_size_y, 3.1], dtype=np.float32),
            size=1.0,
            color=np.array([0.23, 0.24, 0.25], dtype=np.float32),
        )
    )


def add_table(world: World, env_index: int, origin: np.ndarray, theme_name: str) -> None:
    table_center = translated(origin, TABLE_CENTER_LOCAL)
    world.scene.add(
        FixedCuboid(
            name=f"table_{env_index:02d}",
            prim_path=f"{env_root(env_index)}/Table",
            position=table_center,
            scale=TABLE_SIZE,
            size=1.0,
            color=np.array([0.52, 0.34, 0.19], dtype=np.float32),
        )
    )
    add_table_finish(
        world,
        table_root=f"{env_root(env_index)}/TableDecor",
        table_center=table_center,
        table_size=TABLE_SIZE,
        theme=get_theme(theme_name),
    )


def add_place_box(world: World, env_index: int, origin: np.ndarray) -> None:
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)
    root = f"{env_root(env_index)}/PlaceBox"

    for name, position, scale in (
        (
            "Bottom",
            np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1], bottom_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
        ),
        (
            "WallLeft",
            np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] - PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallRight",
            np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0] + PLACE_BOX_OUTER_X / 2.0, origin[1] + PLACE_BOX_CENTER_LOCAL[1], wall_z], dtype=np.float32),
            np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallFront",
            np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallBack",
            np.array([origin[0] + PLACE_BOX_CENTER_LOCAL[0], origin[1] + PLACE_BOX_CENTER_LOCAL[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
    ):
        world.scene.add(
            FixedCuboid(
                name=f"place_box_{name.lower()}_{env_index:02d}",
                prim_path=f"{root}/{name}",
                position=position,
                scale=scale,
                size=1.0,
                color=box_color,
            )
        )


def add_franka(world: World, env_index: int, origin: np.ndarray) -> SingleManipulator:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    robot_path = f"{env_root(env_index)}/Franka"
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


def add_cubes(world: World, env_index: int, origin: np.ndarray) -> list[DynamicCuboid]:
    cubes: list[DynamicCuboid] = []
    for name, position, scale, color in CUBES:
        cubes.append(
            world.scene.add(
                DynamicCuboid(
                    name=f"{name}_{env_index:02d}",
                    prim_path=f"{env_root(env_index)}/{name}",
                    position=translated(origin, position),
                    scale=scale,
                    size=1.0,
                    color=color,
                )
            )
        )
    return cubes


def add_station_cameras(env_index: int, origin: np.ndarray) -> tuple[str, str]:
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
    x_values = np.array([origin[0] for origin in origins], dtype=np.float32)
    y_values = np.array([origin[1] for origin in origins], dtype=np.float32)
    min_x = float(np.min(x_values) - WORKCELL_HALF_EXTENT_X - 0.90)
    max_x = float(np.max(x_values) + WORKCELL_HALF_EXTENT_X + 0.90)
    min_y = float(np.min(y_values) - WORKCELL_HALF_EXTENT_Y - 0.90)
    max_y = float(np.max(y_values) + WORKCELL_HALF_EXTENT_Y + 0.90)
    center_x = float((min_x + max_x) / 2.0)
    center_y = float((min_y + max_y) / 2.0)
    room_size_x = max_x - min_x
    room_size_y = max_y - min_y

    target = np.array([center_x + 0.45, center_y, 0.72], dtype=np.float32)
    eye = np.array(
        [
            center_x + min(1.60, room_size_x * 0.18),
            min_y + min(1.35, room_size_y * 0.18),
            4.20,
        ],
        dtype=np.float32,
    )
    return eye, target


def build_scene() -> tuple[World, list[SingleManipulator], list[list[DynamicCuboid]], list[tuple[str, str]]]:
    origins = env_layout(ARGS.num_envs)
    world = World(stage_units_in_meters=1.0)

    create_lights(origins)
    add_showroom_shell(world, origins)

    robots: list[SingleManipulator] = []
    cubes_per_env: list[list[DynamicCuboid]] = []
    camera_paths: list[tuple[str, str]] = []

    for env_index, origin in enumerate(origins):
        theme_name = SHOWROOM_THEME_SEQUENCE[env_index % len(SHOWROOM_THEME_SEQUENCE)]
        add_styled_room_shell(
            world,
            room_root=f"{env_root(env_index)}/RoomShell",
            center_xy=origin + ROOM_CENTER_LOCAL,
            size_xy=ROOM_SIZE_LOCAL,
            theme=get_theme(theme_name),
        )
        add_table(world, env_index, origin, theme_name)
        add_place_box(world, env_index, origin)
        add_styled_workcell_decor(
            world,
            decor_root=f"{env_root(env_index)}/Decor",
            station_center=translated(origin, TABLE_CENTER_LOCAL),
            theme=get_theme(theme_name),
            table_size=TABLE_SIZE,
        )
        add_optional_robocasa_assets(
            asset_root=str(ARGS.robocasa_asset_root) if ARGS.robocasa_asset_root is not None else None,
            world_root=f"{env_root(env_index)}/Decor",
            placements=[
                (
                    "coffee",
                    ("coffee", "kettle", "mug"),
                    translated(origin, np.array([0.78, 0.52, 0.69], dtype=np.float32)),
                    (0.0, 0.0, 0.0),
                ),
                (
                    "small_appliance",
                    ("toaster", "microwave", "blender"),
                    translated(origin, np.array([0.30, 0.50, 0.69], dtype=np.float32)),
                    (0.0, 24.0, 0.0),
                ),
                (
                    "plant",
                    ("plant", "potted", "vase"),
                    translated(origin, np.array([0.12, -0.55, 0.83], dtype=np.float32)),
                    (0.0, 0.0, 0.0),
                ),
                (
                    "stool",
                    ("chair", "stool"),
                    translated(origin, np.array([1.15, -0.32, 0.0], dtype=np.float32)),
                    (0.0, 0.0, 18.0),
                ),
            ],
        )
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
        set_camera_view(eye=overview_eye, target=overview_target, camera_prim_path="/OmniverseKit_Persp")

    print(f"Built {len(origins)} styled RoboCasa-inspired workcells.")
    for env_index, _origin in enumerate(origins):
        theme_name = SHOWROOM_THEME_SEQUENCE[env_index % len(SHOWROOM_THEME_SEQUENCE)]
        print(f"[env_{env_index:02d}] theme={theme_name}")
        print(f"[env_{env_index:02d}] robot={env_root(env_index)}/Franka")
        print(f"[env_{env_index:02d}] room={env_root(env_index)}/RoomShell")
    return world, robots, cubes_per_env, camera_paths


def main() -> None:
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
