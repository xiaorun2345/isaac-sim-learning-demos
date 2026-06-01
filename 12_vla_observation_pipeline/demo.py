from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.groot_utils import build_vla_observation


def capture_fake_camera_frame():
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[40:120, 40:120, 0] = 255
    return rgb


def read_fake_robot_state():
    joint_positions = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)
    joint_velocities = np.zeros(9, dtype=np.float32)
    return joint_positions, joint_velocities


def main():
    instruction = "pick the red cube"
    rgb = capture_fake_camera_frame()
    joint_positions, joint_velocities = read_fake_robot_state()
    observation = build_vla_observation(
        rgb=rgb,
        instruction=instruction,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
    )

    print("Demo 12: detailed VLA observation pipeline")
    print(f"instruction: {observation['instruction']}")
    print(f"rgb shape: {observation['rgb'].shape}")
    print(f"rgb min/max: {observation['rgb'].min():.3f}/{observation['rgb'].max():.3f}")
    print(f"joint_positions: {observation['joint_positions']}")
    print(f"joint_velocities: {observation['joint_velocities']}")


if __name__ == "__main__":
    main()
