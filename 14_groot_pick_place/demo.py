from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.groot_utils import (
    DummyGrootPolicy,
    build_vla_observation,
    ensure_groot_checkpoint,
    map_action_to_joint_targets,
)


def main():
    try:
        checkpoint = ensure_groot_checkpoint()
    except RuntimeError as exc:
        print(f"GR00T setup error: {exc}")
        return

    current_joint_positions = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)
    policy = DummyGrootPolicy(checkpoint)

    print("Demo 14: closed-loop GR00T control sketch")
    for step in range(3):
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb[40 + 10 * step : 120 + 10 * step, 40:120, 0] = 255
        observation = build_vla_observation(
            rgb=rgb,
            instruction="pick the red cube and lift it",
            joint_positions=current_joint_positions,
            joint_velocities=np.zeros(9, dtype=np.float32),
        )
        action = policy.infer_action(observation)
        joint_delta = np.pad(action[:5], (0, 4), mode="constant")
        next_joint_targets = map_action_to_joint_targets(current_joint_positions, joint_delta, scale=0.25)
        print(f"step={step}")
        print(f"  action={action}")
        print(f"  next_joint_targets={next_joint_targets}")
        current_joint_positions = next_joint_targets


if __name__ == "__main__":
    main()
