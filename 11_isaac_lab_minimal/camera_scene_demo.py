import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Franka camera scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG


@configclass
class FrankaCameraSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)),
    )
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.05)),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.05,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 10.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(1.4, 0.0, 1.0),
            rot=(0.683, 0.183, 0.183, 0.683),
            convention="world",
        ),
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    scene = InteractiveScene(FrankaCameraSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("Isaac Lab Franka camera scene started.")

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        rgb = scene["camera"].data.output["rgb"]
        print(f"batched_rgb_shape={tuple(rgb.shape)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
