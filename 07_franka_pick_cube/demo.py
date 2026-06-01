from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World

from common.core_utils import add_basic_light
from common.franka_utils import arm_pose_action, gripper_close_action, gripper_open_action, load_franka


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            position=[0.45, 0.0, 0.30],
            size=0.05,
            color=[0.9, 0.1, 0.1],
        )
    )
    robot = load_franka()

    world.reset()
    robot.initialize()

    scripted_actions = {
        50: gripper_open_action(),
        100: arm_pose_action([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.8, 0.04, 0.04]),
        180: arm_pose_action([0.05, -0.35, 0.0, -1.55, 0.0, 1.85, 0.8, 0.04, 0.04]),
        240: gripper_close_action(),
        320: arm_pose_action([0.0, -0.9, 0.0, -1.9, 0.0, 2.15, 0.8, 0.0, 0.0]),
    }

    for step in range(420):
        if step in scripted_actions:
            robot.apply_action(scripted_actions[step])
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
