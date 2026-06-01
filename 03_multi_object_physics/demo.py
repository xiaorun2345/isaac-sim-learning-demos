from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.world import World

from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            position=[0.6, 0.0, 0.2],
            scale=[0.6, 1.0, 0.4],
            color=[0.7, 0.7, 0.7],
        )
    )

    for index in range(3):
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Cube_{index}",
                position=[0.15 * index, 0.0, 0.7 + 0.15 * index],
                size=0.08,
                color=[0.2 + 0.2 * index, 0.6, 0.3],
            )
        )

    world.reset()

    for _ in range(360):
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
