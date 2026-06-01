from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World

from common.core_utils import add_basic_light
from common.franka_utils import arm_pose_action, gripper_close_action, gripper_open_action, load_franka


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    robot = load_franka()

    world.reset()
    robot.initialize()

    waypoints = {
        60: arm_pose_action([0.0, -0.6, 0.0, -1.8, 0.0, 2.1, 0.8, 0.04, 0.04]),
        180: gripper_close_action(),
        260: gripper_open_action(),
    }

    for step in range(360):
        if step in waypoints:
            robot.apply_action(waypoints[step])
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
