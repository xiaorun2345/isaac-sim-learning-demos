"""Franka ROS 2 camera demo.

这个文件以 `build_franka_scene_only.py` 的场景代码为基准，只额外增加一件事：
把前视相机图像通过 ROS 2 发布到 `/front_camera/rgb`。

运行方式：

    python demo.py
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析启动参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Build the scene without opening the UI.")
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
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux


TABLE_H = 0.40
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0])
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H])
TABLE_SURFACE_Z = TABLE_H

FRANKA_PRIM_PATH = "/World/Franka"
FRONT_CAMERA_PATH = "/World/front_camera"
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"
ROS2_CAMERA_GRAPH_PATH = "/World/ROS2CameraGraph"
ROS2_CAMERA_TOPIC = "/front_camera/rgb"

FRONT_CAMERA_EYE = np.array([1.15, -1.10, 1.10])
FRONT_CAMERA_TARGET = np.array([0.40, 0.0, 0.65])
WRIST_CAMERA_LOCAL_POS = (0.12, 0.0, 0.10)
WRIST_CAMERA_LOCAL_ROT = (70.0, 0.0, -90.0)

CUBES = (
    ("cube_red", np.array([0.35, -0.18, TABLE_SURFACE_Z + 0.0275]), np.array([0.055, 0.055, 0.055]), np.array([0.90, 0.15, 0.10])),
    ("cube_blue", np.array([0.50, -0.05, TABLE_SURFACE_Z + 0.0450]), np.array([0.040, 0.040, 0.090]), np.array([0.20, 0.40, 0.90])),
    ("cube_green", np.array([0.62, 0.10, TABLE_SURFACE_Z + 0.0250]), np.array([0.070, 0.070, 0.050]), np.array([0.15, 0.80, 0.25])),
    ("cube_yellow", np.array([0.75, -0.20, TABLE_SURFACE_Z + 0.0375]), np.array([0.055, 0.055, 0.075]), np.array([0.95, 0.80, 0.10])),
)

PLACE_BOX_CENTER = np.array([0.82, 0.16])
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
    """在当前 USD stage 中创建一个相机 prim。"""

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


def create_ros2_camera_graph(width: int = 1280, height: int = 720) -> str:
    """创建 ROS 2 图像发布 OmniGraph。"""

    import omni.graph.core as og
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    enabled_names = {ext["name"] for ext in manager.get_extensions() if ext.get("enabled")}
    if "isaacsim.ros2.bridge" not in enabled_names:
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        omni.kit.app.get_app().update()

    stage = get_current_stage()
    if stage.GetPrimAtPath(ROS2_CAMERA_GRAPH_PATH).IsValid():
        return ROS2_CAMERA_GRAPH_PATH

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": ROS2_CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
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
                ("CreateRenderProduct.inputs:cameraPrim", [Sdf.Path(FRONT_CAMERA_PATH)]),
                ("CreateRenderProduct.inputs:width", max(1, width)),
                ("CreateRenderProduct.inputs:height", max(1, height)),
                ("ROS2Context.inputs:useDomainIDEnvVar", False),
                ("PublishRgb.inputs:frameId", "front_camera"),
                ("PublishRgb.inputs:nodeNamespace", "front_camera"),
                ("PublishRgb.inputs:queueSize", 1),
                ("PublishRgb.inputs:topicName", "rgb"),
                ("PublishRgb.inputs:type", "rgb"),
            ],
        },
    )
    return ROS2_CAMERA_GRAPH_PATH


def create_lights() -> None:
    """创建场景灯光。"""

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
    """向世界中添加一个简化的房间外壳。"""

    world.scene.add(
        FixedCuboid(
            name="room_floor",
            prim_path="/World/Room/Floor",
            position=np.array([0.55, 0.0, -0.025]),
            scale=np.array([3.4, 3.0, 0.05]),
            size=1.0,
            color=np.array([0.34, 0.35, 0.36]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_back_wall",
            prim_path="/World/Room/BackWall",
            position=np.array([0.55, 1.50, 1.20]),
            scale=np.array([3.4, 0.04, 2.4]),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_left_wall",
            prim_path="/World/Room/LeftWall",
            position=np.array([-1.15, 0.0, 1.20]),
            scale=np.array([0.04, 3.0, 2.4]),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_right_wall",
            prim_path="/World/Room/RightWall",
            position=np.array([2.25, 0.0, 1.20]),
            scale=np.array([0.04, 3.0, 2.4]),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48]),
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
            color=np.array([0.55, 0.35, 0.15]),
        )
    )


def add_place_box(world: World) -> None:
    """创建一个开口盒子，作为放置目标区域。"""

    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0

    world.scene.add(
        FixedCuboid(
            name="place_box_bottom",
            prim_path="/World/PlaceBox/Bottom",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], bottom_z]),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H]),
            size=1.0,
            color=np.array([0.54, 0.32, 0.14]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_left",
            prim_path="/World/PlaceBox/WallLeft",
            position=np.array([PLACE_BOX_CENTER[0] - PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z]),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H]),
            size=1.0,
            color=np.array([0.54, 0.32, 0.14]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_right",
            prim_path="/World/PlaceBox/WallRight",
            position=np.array([PLACE_BOX_CENTER[0] + PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z]),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H]),
            size=1.0,
            color=np.array([0.54, 0.32, 0.14]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_front",
            prim_path="/World/PlaceBox/WallFront",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z]),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H]),
            size=1.0,
            color=np.array([0.54, 0.32, 0.14]),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_back",
            prim_path="/World/PlaceBox/WallBack",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z]),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H]),
            size=1.0,
            color=np.array([0.54, 0.32, 0.14]),
        )
    )


def add_franka(world: World) -> SingleManipulator:
    """把 Franka 机器人加入到场景并返回控制对象。"""

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.01, 0.01]),
        action_deltas=np.array([0.01, 0.01]),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H]),
        )
    )
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_cubes(world: World) -> list[DynamicCuboid]:
    """添加动态方块。"""

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
    """构建完整场景。"""

    world = World(stage_units_in_meters=1.0)
    create_lights()
    add_room(world)
    world.scene.add_default_ground_plane()
    add_table(world)
    add_place_box(world)
    franka = add_franka(world)
    cubes = add_cubes(world)

    create_camera(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-35.0, 0.0, 45.0),
        focal_length=10.0,
    )
    create_camera(
        path=WRIST_CAMERA_PATH,
        position=WRIST_CAMERA_LOCAL_POS,
        rotation_xyz_deg=WRIST_CAMERA_LOCAL_ROT,
        focal_length=4.0,
    )
    create_ros2_camera_graph()

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

    print(f"Franka prim path: {FRANKA_PRIM_PATH}")
    print(f"Front camera prim: {FRONT_CAMERA_PATH}")
    print(f"Wrist camera prim: {WRIST_CAMERA_PATH}")
    print(f"ROS2 camera topic: {ROS2_CAMERA_TOPIC}")
    print("Place box prim: /World/PlaceBox")
    print(f"Cubes: {[cube.name for cube in cubes]}")
    return world, franka, cubes


def main() -> None:
    """脚本主入口。"""

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
