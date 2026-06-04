"""Launch and inspect the custom Demo 18 Isaac Lab Mimic environment."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the Franka red-cube-to-tray Mimic environment.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import franka_tray_mimic
from franka_tray_mimic.scene_contract import TASK_ID
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=env_cfg)
    observation, _ = env.reset()

    print(f"Task ID: {TASK_ID}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Observation groups: {list(observation.keys())}")
    print("Subtasks: grasp -> placed_in_tray -> release/retreat")

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        observation, _, terminated, truncated, _ = env.step(zero_action)
        if torch.any(terminated | truncated):
            observation, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
