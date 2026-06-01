"""Helpers for ROS 2 teaching demos."""

from typing import Iterable

import numpy as np


def ros2_not_ready_message():
    return "ROS 2 bridge setup is environment-dependent. Check this demo README before running."


def print_expected_topics(command_topic, state_topic):
    print(f"Expected command topic: {command_topic}")
    print(f"Expected state topic: {state_topic}")


def summarize_image_message(msg):
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "encoding": str(msg.encoding),
        "frame_id": str(getattr(msg.header, "frame_id", "")),
    }


def decode_joint_command(values: Iterable[float], expected_len=9):
    values = np.asarray(list(values), dtype=np.float32)
    if values.shape[0] != expected_len:
        raise ValueError(f"Expected {expected_len} values, but received {values.shape[0]}.")
    return values
