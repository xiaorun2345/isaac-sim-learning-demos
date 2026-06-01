from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World

from common.core_utils import add_basic_light
from common.franka_utils import home_action, load_franka


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    robot = load_franka()

    world.reset()
    robot.initialize()

    for step in range(360):
        if step == 60:
            robot.apply_action(home_action())
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
