from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.sensors.camera import Camera

from common.camera_utils import format_rgb_shape
from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            position=[0.4, 0.0, 0.3],
            size=0.15,
            color=[0.1, 0.8, 0.2],
        )
    )

    camera = Camera(
        prim_path="/World/Camera",
        position=[1.5, 1.2, 1.2],
        frequency=20,
        resolution=(640, 480),
        orientation=[0.339, 0.176, 0.425, 0.820],
    )

    world.reset()
    camera.initialize()

    for step in range(240):
        world.step(render=True)
        if step % 60 == 0:
            rgba = camera.get_rgba()
            print(format_rgb_shape(rgba))
            try:
                print("Camera intrinsics:", camera.get_intrinsics())
            except AttributeError:
                print("Camera intrinsics API is unavailable in this Isaac Sim version.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
