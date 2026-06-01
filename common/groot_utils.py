"""Helpers for GR00T-oriented VLA demos."""

import os

import numpy as np


def get_groot_checkpoint():
    return os.environ.get("GROOT_CHECKPOINT", "")


def ensure_groot_checkpoint():
    checkpoint = get_groot_checkpoint()
    if not checkpoint:
        raise RuntimeError("Set GROOT_CHECKPOINT before running GR00T demos.")
    return checkpoint


def normalize_rgb_frame(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb / 255.0


def build_vla_observation(rgb, instruction, joint_positions, joint_velocities=None):
    if joint_velocities is None:
        joint_velocities = np.zeros_like(joint_positions, dtype=np.float32)
    return {
        "rgb": normalize_rgb_frame(rgb),
        "instruction": str(instruction),
        "joint_positions": np.asarray(joint_positions, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
    }


def build_groot_policy_observation(rgb, instruction, joint_positions):
    """Build the nested batched format used by the GR00T Policy API examples."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    return {
        "video": {"wrist_cam": rgb[np.newaxis, np.newaxis, ...]},
        "state": {"joints": joint_positions[np.newaxis, np.newaxis, ...]},
        "language": {"task": [[str(instruction)]]},
    }


def extract_first_action(action_dict, action_key):
    """Return the first action from a batched GR00T action chunk."""
    action_chunk = np.asarray(action_dict[action_key], dtype=np.float32)
    if action_chunk.ndim != 3:
        raise ValueError(f"Expected action chunk shape (batch, horizon, dim), got {action_chunk.shape}.")
    return action_chunk[0, 0]


def map_action_to_joint_targets(current_joint_positions, delta_action, scale=0.1):
    current_joint_positions = np.asarray(current_joint_positions, dtype=np.float32)
    delta_action = np.asarray(delta_action, dtype=np.float32)
    return current_joint_positions + delta_action * scale


class DummyGrootPolicy:
    """A tiny stand-in that keeps demo structure readable."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def infer_action(self, observation):
        instruction = observation.get("instruction", "").lower()
        if "red" in instruction:
            return np.array([0.15, 0.0, 0.12, 0.02, 0.00], dtype=np.float32)
        if "blue" in instruction:
            return np.array([-0.10, 0.05, 0.10, -0.02, 0.00], dtype=np.float32)
        return np.array([0.0, 0.0, 0.08, 0.0, 0.04], dtype=np.float32)
