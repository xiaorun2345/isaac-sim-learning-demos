from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
#导入所需的模块和类，包括动态立方体、视觉立方体、世界对象以及一个用于添加基本光源的实用函数。
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.world import World

#从common.core_utils模块导入add_basic_light函数，用于在场景中添加基本的光源，以便更好地照亮对象和环境。
from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(VisualCuboid(prim_path="/World/BoxA", position=[0.0, 0.0, 0.5], size=0.30))
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/BoxB",
            position=[0.5, 0.0, 0.5],
            size=0.20,
            color=[1.0, 0.2, 0.2],
        )
    )
    world.reset()
#运行仿真240步，每步都渲染场景以显示窗口中的更新。当运行完成后，关闭仿真应用程序以释放资源。
    for _ in range(240):
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
