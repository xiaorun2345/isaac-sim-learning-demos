"""Build a clean Franka task scene inside an Isaac Sim room asset.

Run with Isaac Sim's Python environment:

    python isaac-sim-learning-demos/22_robocasa_styled_single_scene/demo.py
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Build the scene without opening the UI.")
    return parser.parse_args()


ARGS = parse_args()

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1400,
        "height": 840,
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
TABLE_CENTER = np.array([0.28, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H
ROOM_ENV_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"

FRANKA_PRIM_PATH = "/World/Franka"
FRONT_CAMERA_PATH = "/World/front_camera"
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"
FRONT_CAMERA_EYE = np.array([2.90, -2.35, 1.75], dtype=np.float32)
FRONT_CAMERA_TARGET = np.array([0.55, 0.0, 0.72], dtype=np.float32)
WRIST_CAMERA_LOCAL_POS = (0.12, 0.0, 0.10)
WRIST_CAMERA_LOCAL_ROT = (70.0, 0.0, -90.0)

CUBES = (
    ("cube_red", np.array([0.35, -0.18, TABLE_SURFACE_Z + 0.0275], dtype=np.float32), np.array([0.055, 0.055, 0.055], dtype=np.float32), np.array([0.90, 0.15, 0.10], dtype=np.float32)),
    ("cube_blue", np.array([0.50, -0.05, TABLE_SURFACE_Z + 0.0450], dtype=np.float32), np.array([0.040, 0.040, 0.090], dtype=np.float32), np.array([0.20, 0.40, 0.90], dtype=np.float32)),
    ("cube_green", np.array([0.62, 0.10, TABLE_SURFACE_Z + 0.0250], dtype=np.float32), np.array([0.070, 0.070, 0.050], dtype=np.float32), np.array([0.15, 0.80, 0.25], dtype=np.float32)),
    ("cube_yellow", np.array([0.75, -0.20, TABLE_SURFACE_Z + 0.0375], dtype=np.float32), np.array([0.055, 0.055, 0.075], dtype=np.float32), np.array([0.95, 0.80, 0.10], dtype=np.float32)),
)

PLACE_BOX_CENTER = np.array([0.82, 0.16], dtype=np.float32)
PLACE_BOX_OUTER_X = 0.24
PLACE_BOX_OUTER_Y = 0.20
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.018
PLACE_BOX_WALL_H = 0.13


def create_camera(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    if path == WRIST_CAMERA_PATH:
        position = (0.06, 0.0, 0.03)
        rotation_xyz_deg = (-95.0, 0.0, -90.0)

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    stage = get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(650.0)

    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(5200.0)
    key.CreateWidthAttr(2.4)
    key.CreateHeightAttr(1.6)
    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(1.10, -1.45, 2.25))
    xform.SetRotate(Gf.Vec3f(-58.0, 0.0, 42.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    fill = UsdLux.RectLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(1600.0)
    fill.CreateWidthAttr(1.8)
    fill.CreateHeightAttr(1.0)
    fill_xform = UsdGeom.XformCommonAPI(fill.GetPrim())
    fill_xform.SetTranslate(Gf.Vec3d(-0.95, 1.15, 1.90))
    fill_xform.SetRotate(Gf.Vec3f(-42.0, 0.0, -128.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_environment() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    add_reference_to_stage(usd_path=assets_root + ROOM_ENV_PATH, prim_path="/World/Environment")


def add_task_table(world: World) -> None:
    worktop_size = np.array([1.34, 0.92, 0.045], dtype=np.float32)
    worktop_center = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_SURFACE_Z - worktop_size[2] / 2.0], dtype=np.float32)
    world.scene.add(
        FixedCuboid(
            name="task_table_top",
            prim_path="/World/TaskTable/Top",
            position=worktop_center,
            scale=worktop_size,
            size=1.0,
            color=np.array([0.72, 0.73, 0.75], dtype=np.float32),
        )
    )

    leg_size = np.array([0.055, 0.055, TABLE_SURFACE_Z - worktop_size[2]], dtype=np.float32)
    leg_center_z = leg_size[2] / 2.0
    x_offset = worktop_size[0] / 2.0 - 0.09
    y_offset = worktop_size[1] / 2.0 - 0.09
    for idx, (x_sign, y_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        world.scene.add(
            FixedCuboid(
                name=f"task_table_leg_{idx:02d}",
                prim_path=f"/World/TaskTable/Leg_{idx:02d}",
                position=np.array(
                    [
                        TABLE_CENTER[0] + x_sign * x_offset,
                        TABLE_CENTER[1] + y_sign * y_offset,
                        leg_center_z,
                    ],
                    dtype=np.float32,
                ),
                scale=leg_size,
                size=1.0,
                color=np.array([0.14, 0.15, 0.16], dtype=np.float32),
            )
        )


def add_place_box(world: World) -> None:
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)
    for name, position, scale in (
        (
            "Bottom",
            np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], bottom_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
        ),
        (
            "WallLeft",
            np.array([PLACE_BOX_CENTER[0] - PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallRight",
            np.array([PLACE_BOX_CENTER[0] + PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallFront",
            np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
        (
            "WallBack",
            np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
        ),
    ):
        world.scene.add(
            FixedCuboid(
                name=f"place_box_{name.lower()}",
                prim_path=f"/World/PlaceBox/{name}",
                position=position,
                scale=scale,
                size=1.0,
                color=box_color,
            )
        )


def add_franka(world: World) -> SingleManipulator:
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


def add_cubes(world: World) -> list[DynamicCuboid]:
    cubes: list[DynamicCuboid] = []
    for name, position, scale, color in CUBES:
        cubes.append(
            world.scene.add(
                DynamicCuboid(
                    name=name,
                    prim_path=f"/World/{name}",
                    position=position,
                    scale=scale,
                    size=1.0,
                    color=color,
                )
            )
        )
    return cubes


def build_scene() -> tuple[World, SingleManipulator, list[DynamicCuboid]]:
    world = World(stage_units_in_meters=1.0)
    add_environment()
    create_lights()
    add_task_table(world)
    add_place_box(world)

    franka = add_franka(world)
    cubes = add_cubes(world)

    create_camera(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-24.0, 0.0, 130.0),
        focal_length=10.0,
    )
    create_camera(
        path=WRIST_CAMERA_PATH,
        position=WRIST_CAMERA_LOCAL_POS,
        rotation_xyz_deg=WRIST_CAMERA_LOCAL_ROT,
        focal_length=4.0,
    )

    world.reset()
    set_camera_view(eye=FRONT_CAMERA_EYE, target=FRONT_CAMERA_TARGET, camera_prim_path=FRONT_CAMERA_PATH)
    if not ARGS.headless:
        set_camera_view(eye=FRONT_CAMERA_EYE, target=FRONT_CAMERA_TARGET, camera_prim_path="/OmniverseKit_Persp")

    print(f"Background environment: {ROOM_ENV_PATH}")
    print(f"Franka prim path: {FRANKA_PRIM_PATH}")
    print(f"Front camera prim: {FRONT_CAMERA_PATH}")
    print(f"Wrist camera prim: {WRIST_CAMERA_PATH}")
    return world, franka, cubes


def main() -> None:
    try:
        world, _, _ = build_scene()
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
