import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.groot_utils import build_groot_policy_observation


def main():
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError:
        print("Install the Isaac-GR00T client package before running this script.")
        print("Official repo: https://github.com/NVIDIA/Isaac-GR00T")
        return

    host = os.environ.get("GROOT_HOST", "127.0.0.1")
    port = int(os.environ.get("GROOT_PORT", "5555"))
    policy = PolicyClient(host=host, port=port, timeout_ms=15000, strict=False)
    if not policy.ping():
        raise RuntimeError(f"Cannot reach GR00T policy server at {host}:{port}.")

    observation = build_groot_policy_observation(
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        instruction="pick the red cube",
        joint_positions=np.zeros(9, dtype=np.float32),
    )
    action_dict, info_dict = policy.get_action(observation)
    print("GR00T policy server replied.")
    print(f"action_keys={list(action_dict.keys())}")
    print(f"info={info_dict}")


if __name__ == "__main__":
    main()
