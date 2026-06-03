"""Build a standalone Isaac Sim scene with a Franka arm, cubes, and cameras.

Run with Isaac Sim's Python environment:

    python example/build_franka_scene_only.py

这个示例只负责“搭场景并让场景保持运行”，不包含 ROS2 通信，也不包含
自动抓取状态机。因此它适合作为入门文件，帮助理解一个 Isaac Sim 脚本
最基础的结构：

1. 先解析命令行参数
2. 尽早创建 `SimulationApp`
3. 再导入 Isaac/Omniverse 相关模块
4. 创建 `World`
5. 往 `World` 里添加灯光、房间、桌子、机器人、相机和物体
6. `world.reset()` 完成初始化
7. 进入循环，不断 `world.step(render=True)`
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    """解析这个示例支持的命令行参数。

    返回值是 `argparse.Namespace`，里面会包含诸如 `headless` 这样的属性。
    例如：

        python example/build_franka_scene_only.py --headless

    时，`ARGS.headless` 就会是 `True`。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Build the scene without opening the UI.")
    return parser.parse_args()


# 这里先解析命令行参数，因为稍后创建 `SimulationApp` 时就要用到 `headless`
# 这个配置。
ARGS = parse_args()

# Isaac Sim/Omniverse 的一个核心约束：
# 很多 Isaac 模块必须在 `SimulationApp` 创建之后再导入，否则容易在扩展
# 初始化阶段报错。
#
# 所以这个文件刻意采用“先创建 app，后导入 Isaac 其他模块”的结构。
simulation_app = SimulationApp(
    {
        # `headless=True` 表示不打开图形界面，只在后台创建模拟环境。
        "headless": ARGS.headless,
        # 如果 headless，就顺便隐藏 UI；否则保留完整界面。
        "hide_ui": ARGS.headless,
        # 使用较好看的光照渲染器，便于观察场景。
        "renderer": "RaytracedLighting",
        # 下面两个参数是窗口/渲染分辨率。
        "width": 1280,
        "height": 720,
    }
)


# 从这里开始才导入 Isaac/Omniverse 相关模块。
# 这些导入放在文件中部不是随意写法，而是为了满足 Isaac Sim 的启动顺序要求。
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux


# ----------------------------
# 场景基础尺寸配置
# ----------------------------
# 这部分常量决定桌子、相机、盒子、机器人等对象的默认位置和大小。
# 把它们集中放在这里，后面想改布局时会更方便。

# 桌子高度（米）。
TABLE_H = 0.40

# 桌子的中心点坐标。
# 注意这里的 z 用的是“高度的一半”，因为立方体/长方体通常按中心点放置。
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0])

# 桌子的长宽高。
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H])

# 桌面上表面的 z 坐标，后面摆放物块时会用这个值做基准。
TABLE_SURFACE_Z = TABLE_H

# 机器人在 USD 场景树中的 prim 路径。
FRANKA_PRIM_PATH = "/World/Franka"

# 世界坐标系下的固定前视相机路径。
FRONT_CAMERA_PATH = "/World/front_camera"

# 安装在 Franka 手腕末端的相机路径。
# 它挂在 `panda_hand` 下面，所以是一个相对机器人结构的子 prim。
WRIST_CAMERA_PATH = f"{FRANKA_PRIM_PATH}/panda_hand/wrist_camera"

# 前视相机的“眼睛位置”，也就是相机放在哪。
FRONT_CAMERA_EYE = np.array([1.15, -1.10, 1.10])

# 前视相机观察的目标点。
FRONT_CAMERA_TARGET = np.array([0.40, 0.0, 0.65])

# 手腕相机的局部平移和局部旋转。
# 这是相对于 `panda_hand` 的局部坐标，不是世界坐标。
WRIST_CAMERA_LOCAL_POS = (0.12, 0.0, 0.10)
WRIST_CAMERA_LOCAL_ROT = (70.0, 0.0, -90.0)

# 待抓取物体配置：
# 每个元素依次表示
# (名字, 世界坐标位置, XYZ尺寸, RGB颜色)
CUBES = (
    ("cube_red", np.array([0.35, -0.18, TABLE_SURFACE_Z + 0.0275]), np.array([0.055, 0.055, 0.055]), np.array([0.90, 0.15, 0.10])),
    ("cube_blue", np.array([0.50, -0.05, TABLE_SURFACE_Z + 0.0450]), np.array([0.040, 0.040, 0.090]), np.array([0.20, 0.40, 0.90])),
    ("cube_green", np.array([0.62, 0.10, TABLE_SURFACE_Z + 0.0250]), np.array([0.070, 0.070, 0.050]), np.array([0.15, 0.80, 0.25])),
    ("cube_yellow", np.array([0.75, -0.20, TABLE_SURFACE_Z + 0.0375]), np.array([0.055, 0.055, 0.075]), np.array([0.95, 0.80, 0.10])),
)

# 放置盒子的中心点（只给 x/y，z 由盒子厚度和桌面高度推出来）。
PLACE_BOX_CENTER = np.array([0.82, 0.16])

# 盒子的外部尺寸。
PLACE_BOX_OUTER_X = 0.24
PLACE_BOX_OUTER_Y = 0.20

# 盒子底板厚度、壁厚、壁高。
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.018
PLACE_BOX_WALL_H = 0.13


def create_camera(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    """在当前 USD stage 中创建一个相机 prim。

    参数说明：
    - `path`: 相机 prim 的完整路径
    - `position`: 相机位置
    - `rotation_xyz_deg`: 欧拉角旋转，单位是度
    - `focal_length`: 焦距

    这里既可以创建世界坐标系下的固定相机，也可以创建挂在机器人手腕上的局部相机。
    """

    if path == WRIST_CAMERA_PATH:
        # Mount the wrist camera slightly forward and above the hand, tilted
        # down toward the tabletop workspace instead of up toward the ceiling.
        #
        # 虽然函数参数里传进来了 wrist camera 的局部位姿，但这里仍然对它做一次
        # 特殊处理，目的是强制给出一个更稳定、更适合看桌面的默认视角。
        position = (0.06, 0.0, 0.03)
        rotation_xyz_deg = (-95.0, 0.0, -90.0)

    # 获取 Isaac 当前正在编辑的 USD stage。
    stage = get_current_stage()

    # 在该路径上定义一个 Camera prim；如果路径已存在，则会复用/覆盖该 prim 定义。
    camera = UsdGeom.Camera.Define(stage, path)

    # 设置相机光学参数。
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # 使用 USD 的通用变换 API 设置平移和旋转。
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    """创建场景灯光。

    这里使用两种灯：
    - DomeLight：环境光，整体把场景照亮
    - RectLight：矩形主光源，提供更明显的方向感和阴影
    """

    stage = get_current_stage()

    # 半球/环境光，让整个空间不会太黑。
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1200.0)

    # 主光源，从斜上方照向桌面和机器人。
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateWidthAttr(1.6)
    key.CreateHeightAttr(1.2)

    # 给主光源设置位置和角度。
    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.65, -0.20, 1.80))
    xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World) -> None:
    """向世界中添加一个简化的房间外壳。

    这里的地板和三面墙主要是为了让场景更有空间感，视觉上不至于漂浮在空白里。
    因为这些物体不需要参与运动控制，所以使用 `FixedCuboid`。
    """

    # 地板。z = -0.025 说明它的中心略低于 0，这样顶面大约刚好贴近世界地面。
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

    # 后墙，位于 y 正方向远处。
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

    # 左墙，位于 x 负方向。
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

    # 右墙，位于 x 正方向。
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
    """添加工作台。

    桌子是一个固定长方体，后续方块会被摆在桌面上，机器人底座也会相对于桌高来放置。
    """

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
    """创建一个开口盒子，作为“放置目标区域”。

    这个盒子不是通过一个复杂 mesh 导入的，而是用 1 个底板 + 4 面墙拼出来。
    这样更容易看懂坐标，也方便后续调尺寸。
    """

    # 盒子底板中心高度。
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0

    # 盒子侧壁中心高度。它是在底板上方继续抬高半个墙高得到的。
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0

    # 底板。
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

    # 左墙。
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

    # 右墙。
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

    # 前墙。
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

    # 后墙。
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
    """把 Franka 机器人加入到场景并返回其控制对象。

    这一步分成两层：
    1. 先把 Isaac 自带的 Franka USD 资源 reference 到当前 stage
    2. 再用 `SingleManipulator` 把它包装成 Isaac 可控制的机器人对象
    """

    # 获取 Isaac Sim 自带资产库的根路径。
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    # Franka 机器人模型在 Isaac 资产库里的标准位置。
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    # 把机器人 USD 资源挂到 `/World/Franka` 这个 prim 路径下。
    # 这里是 reference，不是手动复制所有 mesh 数据。
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    # 配置夹爪。
    # `end_effector_prim_path` 指向夹爪末端挂载位置。
    # `joint_prim_names` 指定左右手指对应的关节。
    # opened/closed positions 给出张开和闭合时的关节目标值。
    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.01, 0.01]),
        action_deltas=np.array([0.01, 0.01]),
    )

    # `SingleManipulator` 是 Isaac 对单机械臂机器人的一个高层封装。
    # 它把底层 USD 机器人和运动控制接口关联起来。
    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H]),
        )
    )

    # 默认让夹爪张开，这样启动后更像“待命状态”。
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_cubes(world: World) -> list[DynamicCuboid]:
    """添加所有动态方块并返回列表。

    这里用 `DynamicCuboid` 而不是 `FixedCuboid`，因为这些方块应该参与物理模拟：
    它们可以受重力影响、与桌子碰撞，也可以被机器人推动或抓取。
    """

    cubes: list[DynamicCuboid] = []
    for name, position, scale, color in CUBES:
        # 逐个读取配置，并在场景里创建对应方块。
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
    """构建完整场景，并返回世界、机器人对象、方块列表。

    这个函数是本文件最核心的“搭建流程”。
    顺序上基本遵循：

    1. 创建 `World`
    2. 添加光照和环境
    3. 添加地面、桌子、盒子
    4. 添加机器人和动态物体
    5. 添加相机
    6. `world.reset()` 完成初始化
    7. 设置观察视角
    """

    # 创建 Isaac 世界对象。
    # `stage_units_in_meters=1.0` 表示 USD 中 1 个单位就是 1 米。
    world = World(stage_units_in_meters=1.0)

    # 先搭“环境”。
    create_lights()
    add_room(world)

    # 添加 Isaac 默认地面。它和我们自定义的 room floor 可以共存。
    # 默认地面更偏向物理/示例基础设施，自定义地板更偏视觉包裹感。
    world.scene.add_default_ground_plane()

    # 再搭“工作区域”。
    add_table(world)
    add_place_box(world)

    # 然后加入机器人和可交互物体。
    franka = add_franka(world)
    cubes = add_cubes(world)

    # 添加外部观察相机。
    create_camera(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-35.0, 0.0, 45.0),
        focal_length=10.0,
    )

    # 添加机器人手腕相机。
    create_camera(
        path=WRIST_CAMERA_PATH,
        position=WRIST_CAMERA_LOCAL_POS,
        rotation_xyz_deg=WRIST_CAMERA_LOCAL_ROT,
        focal_length=4.0,
    )

    # 这是 Isaac 中很关键的一步。
    # `reset()` 会让所有对象完成一次初始化，确保物理、关节、传感器等状态就位。
    # 很多时候如果不 reset，后面直接控制或读状态会出现异常。
    world.reset()

    # 给前视相机设置一个“看向桌面中心区域”的视角。
    set_camera_view(
        eye=FRONT_CAMERA_EYE,
        target=FRONT_CAMERA_TARGET,
        camera_prim_path=FRONT_CAMERA_PATH,
    )

    # 如果当前不是无界面模式，还顺便把 Isaac 默认透视视口也切到同样的好视角，
    # 这样脚本启动后用户第一眼看到的就是我们关心的工作区。
    if not ARGS.headless:
        set_camera_view(
            eye=FRONT_CAMERA_EYE,
            target=FRONT_CAMERA_TARGET,
            camera_prim_path="/OmniverseKit_Persp",
        )

    # 打印一些关键信息，方便在终端里确认场景对象确实建出来了。
    print(f"Franka prim path: {FRANKA_PRIM_PATH}")
    print(f"Front camera prim: {FRONT_CAMERA_PATH}")
    print(f"Wrist camera prim: {WRIST_CAMERA_PATH}")
    print("Place box prim: /World/PlaceBox")
    print(f"Cubes: {[cube.name for cube in cubes]}")
    return world, franka, cubes


def main() -> None:
    """脚本主入口。

    行为分两种：
    - `headless=True`：只搭场景，验证成功后直接退出
    - `headless=False`：搭场景后进入 Isaac Sim 主循环，持续渲染并响应窗口关闭
    """

    try:
        # 构建场景。这里虽然返回了机器人和方块，但这个示例当前只需要 `world`。
        world, _, _ = build_scene()

        # 无界面模式通常只用来验证“场景能不能成功创建”，不需要继续跑渲染循环。
        if ARGS.headless:
            return

        # 开始世界时间线/物理推进。
        world.play()

        # 这是最常见的 Isaac Sim 脚本运行循环：
        # 只要 app 还没被关闭，就不断推进一帧模拟和渲染。
        while simulation_app.is_running():
            # `render=True` 表示这一步同时更新画面。
            world.step(render=True)
    except KeyboardInterrupt:
        # 允许用户用 Ctrl+C 优雅中断。
        pass
    except Exception:
        # 打印完整 traceback，方便排查 Isaac 扩展、资源路径、初始化顺序等问题。
        traceback.print_exc()
        raise
    finally:
        # 无论正常退出还是异常退出，都要关闭 `SimulationApp`。
        # 这是释放 Isaac/Omniverse 资源的标准收尾动作。
        simulation_app.close()


if __name__ == "__main__":
    # 只有直接运行这个文件时才进入主函数。
    # 如果这个文件被其他模块 import，则不会自动执行。
    main()
