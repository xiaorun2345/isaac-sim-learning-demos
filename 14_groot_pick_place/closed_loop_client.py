import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.groot_utils import build_groot_policy_observation, extract_first_action


ACTION_KEY = os.environ.get("GROOT_ACTION_KEY", "joints")


class FrankaControlAdapter:
    """Replace these methods with Isaac Lab scene reads and robot writes."""

    def __init__(self):
        self.joint_positions = np.zeros(9, dtype=np.float32)

    def capture_rgb(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def read_joint_positions(self):
        return self.joint_positions.copy()

    def apply_joint_targets(self, targets):
        self.joint_positions = np.asarray(targets, dtype=np.float32)
        print(f"applied_joint_targets={self.joint_positions.tolist()}")


def main():
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError:
        print("Install the Isaac-GR00T client package before running this script.")
        return

    policy = PolicyClient(
        host=os.environ.get("GROOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("GROOT_PORT", "5555")),
        timeout_ms=15000,
        strict=False,
    )
    if not policy.ping():
        raise RuntimeError("Cannot reach the GR00T policy server.")

    robot = FrankaControlAdapter()
    for step in range(10):
        observation = build_groot_policy_observation(
            rgb=robot.capture_rgb(),
            instruction="pick the red cube and place it in the tray",
            joint_positions=robot.read_joint_positions(),
        )
        action_dict, info_dict = policy.get_action(observation)
        joint_targets = extract_first_action(action_dict, ACTION_KEY)
        robot.apply_joint_targets(joint_targets)
        print(f"step={step}, info={info_dict}")


if __name__ == "__main__":
    main()
