import argparse
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab plus GR00T closed-loop Franka control.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
from gr00t.policy.server_client import PolicyClient

from common.groot_utils import build_groot_policy_observation, extract_first_action


ACTION_KEY = os.environ.get("GROOT_ACTION_KEY", "joints")


@configclass
class FrankaVlaSceneCfg(InteractiveSceneCfg):
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


def read_policy_observation(scene, instruction):
    rgb = scene["camera"].data.output["rgb"][0].detach().cpu().numpy()
    joints = scene["robot"].data.joint_pos[0].detach().cpu().numpy()
    return build_groot_policy_observation(rgb, instruction, joints)


def main():
    policy = PolicyClient(
        host=os.environ.get("GROOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("GROOT_PORT", "5555")),
        timeout_ms=15000,
        strict=False,
    )
    if not policy.ping():
        raise RuntimeError("Cannot reach GR00T policy server.")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    scene = InteractiveScene(FrankaVlaSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    instruction = "pick the red cube and place it in the tray"

    step = 0
    while simulation_app.is_running():
        if step % 8 == 0:
            observation = read_policy_observation(scene, instruction)
            action_dict, info_dict = policy.get_action(observation)
            joint_targets = extract_first_action(action_dict, ACTION_KEY)
            if joint_targets.shape[0] != scene["robot"].num_joints:
                raise ValueError(
                    f"Policy returned {joint_targets.shape[0]} joints, "
                    f"but Franka expects {scene['robot'].num_joints}."
                )
            targets = torch.as_tensor(joint_targets, device=scene["robot"].device).unsqueeze(0)
            scene["robot"].set_joint_position_target(targets)
            print(f"step={step}, action_key={ACTION_KEY}, info={info_dict}")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        step += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
