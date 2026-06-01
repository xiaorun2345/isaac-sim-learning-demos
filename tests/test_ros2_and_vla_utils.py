import unittest

import numpy as np

from common.groot_utils import (
    build_vla_observation,
    build_groot_policy_observation,
    extract_first_action,
    map_action_to_joint_targets,
    normalize_rgb_frame,
)
from common.ros2_utils import decode_joint_command, summarize_image_message


class DummyImageMessage:
    def __init__(self, height=480, width=640, encoding="rgb8", frame_id="camera_frame"):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.header = type("Header", (), {"frame_id": frame_id})()


class Ros2UtilsTests(unittest.TestCase):
    def test_summarize_image_message_returns_expected_metadata(self):
        summary = summarize_image_message(DummyImageMessage())
        self.assertEqual(summary["height"], 480)
        self.assertEqual(summary["width"], 640)
        self.assertEqual(summary["encoding"], "rgb8")
        self.assertEqual(summary["frame_id"], "camera_frame")

    def test_decode_joint_command_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            decode_joint_command([0.0, 1.0], expected_len=9)


class VlaUtilsTests(unittest.TestCase):
    def test_normalize_rgb_frame_scales_to_zero_one(self):
        rgb = np.array([[[0, 127, 255]]], dtype=np.uint8)
        normalized = normalize_rgb_frame(rgb)
        self.assertAlmostEqual(float(normalized[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[0, 0, 1]), 127.0 / 255.0, places=5)
        self.assertAlmostEqual(float(normalized[0, 0, 2]), 1.0)

    def test_build_vla_observation_contains_expected_keys(self):
        observation = build_vla_observation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            instruction="pick the red cube",
            joint_positions=np.zeros(9, dtype=np.float32),
            joint_velocities=np.ones(9, dtype=np.float32),
        )
        self.assertEqual(set(observation.keys()), {"rgb", "instruction", "joint_positions", "joint_velocities"})

    def test_map_action_to_joint_targets_adds_scaled_delta(self):
        current = np.zeros(3, dtype=np.float32)
        delta = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        target = map_action_to_joint_targets(current, delta, scale=0.2)
        np.testing.assert_allclose(target, np.array([0.2, -0.2, 0.1], dtype=np.float32))

    def test_build_groot_policy_observation_adds_batch_and_time_axes(self):
        observation = build_groot_policy_observation(
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            instruction="pick the red cube",
            joint_positions=np.zeros(9, dtype=np.float32),
        )
        self.assertEqual(observation["video"]["wrist_cam"].shape, (1, 1, 224, 224, 3))
        self.assertEqual(observation["state"]["joints"].shape, (1, 1, 9))
        self.assertEqual(observation["language"]["task"], [["pick the red cube"]])

    def test_extract_first_action_returns_first_step_from_chunk(self):
        action = {"joints": np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)}
        first = extract_first_action(action, action_key="joints")
        np.testing.assert_allclose(first, np.array([0.1, 0.2], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
