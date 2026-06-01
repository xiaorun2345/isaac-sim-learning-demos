from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from pxr import UsdLux

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path


def add_basic_light(stage):
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(500.0)


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)

    assets_root = get_assets_root_path()
    robot_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot_prim_path = "/World/Franka"

    add_reference_to_stage(robot_usd, robot_prim_path)
    robot = SingleArticulation(prim_path=robot_prim_path, name="franka")

    world.reset()
    robot.initialize()

    # 9 joints total: 7 arm joints + 2 gripper joints
    home_action = ArticulationAction(
        joint_positions=np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)
    )
    close_gripper_action = ArticulationAction(
        joint_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_indices=np.array([7, 8], dtype=np.int32),
    )
    lift_joint_2_action = ArticulationAction(
        joint_positions=np.array([-0.3], dtype=np.float32),
        joint_indices=np.array([1], dtype=np.int32),
    )

    for step in range(600):
        if step == 60:
            robot.apply_action(home_action)
        elif step == 180:
            robot.apply_action(close_gripper_action)
        elif step == 300:
            robot.apply_action(lift_joint_2_action)

        world.step(render=True)

        if not simulation_app.is_running():
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
