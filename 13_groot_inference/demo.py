from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.groot_utils import DummyGrootPolicy, build_vla_observation, ensure_groot_checkpoint


def main():
    try:
        checkpoint = ensure_groot_checkpoint()
    except RuntimeError as exc:
        print(f"GR00T setup error: {exc}")
        return

    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[40:120, 40:120, 0] = 255
    observation = build_vla_observation(
        rgb=rgb,
        instruction="pick the red cube",
        joint_positions=np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32),
        joint_velocities=np.zeros(9, dtype=np.float32),
    )

    policy = DummyGrootPolicy(checkpoint)
    action = policy.infer_action(observation)

    print("Demo 13: detailed GR00T-style inference")
    print(f"checkpoint: {checkpoint}")
    print(f"instruction: {observation['instruction']}")
    print(f"normalized_rgb_shape: {observation['rgb'].shape}")
    print(f"joint_positions_shape: {observation['joint_positions'].shape}")
    print(f"predicted_action: {action}")
    print("Action meaning: [dx, dy, dz, wrist_delta, gripper_target]")


if __name__ == "__main__":
    main()
