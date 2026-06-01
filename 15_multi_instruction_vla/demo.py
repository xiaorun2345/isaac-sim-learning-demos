from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np

from common.groot_utils import DummyGrootPolicy, build_vla_observation, ensure_groot_checkpoint


def run_instruction(policy, instruction):
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    if "red" in instruction:
        rgb[40:120, 40:120, 0] = 255
    if "blue" in instruction:
        rgb[40:120, 120:200, 2] = 255
    observation = build_vla_observation(
        rgb=rgb,
        instruction=instruction,
        joint_positions=np.zeros(9, dtype=np.float32),
        joint_velocities=np.zeros(9, dtype=np.float32),
    )
    action = policy.infer_action(observation)
    return {
        "instruction": instruction,
        "action": action.tolist(),
    }


def main():
    try:
        checkpoint = ensure_groot_checkpoint()
    except RuntimeError as exc:
        print(f"GR00T setup error: {exc}")
        return

    policy = DummyGrootPolicy(checkpoint)
    instructions = [
        "pick the red cube",
        "pick the blue cube",
        "lift the object and move to the tray",
    ]

    print("Demo 15: multi-instruction VLA runner")
    logs = []
    for instruction in instructions:
        result = run_instruction(policy, instruction)
        logs.append(result)
        print(json.dumps(result, ensure_ascii=True))

    print("Episode summary:")
    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
