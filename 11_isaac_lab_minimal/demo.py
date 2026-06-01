from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    try:
        import argparse

        import isaaclab.sim as sim_utils
        from isaaclab.app import AppLauncher
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
    except ImportError:
        print("Isaac Lab is not available in this Python environment.")
        print("Run this demo from an Isaac Lab environment.")
        return

    parser = argparse.ArgumentParser(description="Isaac Lab minimal scene demo.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args([])

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    class DemoSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)),
        )

    try:
        sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
        scene = InteractiveScene(DemoSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        print("Demo 11: Isaac Lab minimal interactive scene")
        for step in range(180):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            if step % 60 == 0:
                print(f"step={step}, physics_dt={sim.get_physics_dt():.4f}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
