"""Pure-Python scene constants shared by the Isaac Lab configuration and tests."""

from __future__ import annotations

from collections.abc import Sequence


TASK_ID = "Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0"

# Demo 17 uses world Z=0.40 as the tabletop. Isaac Lab places the Franka base
# at tabletop Z=0, so all Z values below are expressed relative to that surface.
TABLE_SIZE = (1.0, 0.8, 0.40)
TABLE_CENTER = (0.45, 0.0, -0.20)

CUBE_SIZE = (0.045, 0.045, 0.045)
CUBE_HALF_Z = CUBE_SIZE[2] / 2.0
TARGET_CUBE_X_RANGE = (0.42, 0.44)
TARGET_CUBE_Y_RANGE = (-0.02, 0.02)

BLUE_CUBE_POSITION = (0.34, -0.18, CUBE_HALF_Z)
GREEN_CUBE_POSITION = (0.34, 0.18, CUBE_HALF_Z)

TRAY_CENTER_XY = (0.64, 0.18)
TRAY_OUTER_X = 0.18
TRAY_OUTER_Y = 0.18
TRAY_BOTTOM_HEIGHT = 0.024
TRAY_WALL_THICKNESS = 0.016
TRAY_WALL_HEIGHT = 0.10
TRAY_REFERENCE_Z = TRAY_BOTTOM_HEIGHT / 2.0
TRAY_SUCCESS_MAX_Z = 0.12

FRONT_CAMERA_POSITION = (1.15, -1.10, 0.70)
FRONT_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (640, 480)

# Each tuple is (reference object, termination signal). The final subtask does
# not require a termination signal because the episode success term ends it.
MIMIC_SUBTASKS = (
    ("cube_2", "grasp"),
    ("tray", "placed_in_tray"),
    ("tray", None),
)


def is_position_inside_tray(position: Sequence[float]) -> bool:
    """Return whether an XYZ position satisfies Demo 17's tray success rule."""

    x, y, z = (float(value) for value in position[:3])
    inner_half_x = TRAY_OUTER_X / 2.0 - TRAY_WALL_THICKNESS
    inner_half_y = TRAY_OUTER_Y / 2.0 - TRAY_WALL_THICKNESS
    return (
        abs(x - TRAY_CENTER_XY[0]) < inner_half_x
        and abs(y - TRAY_CENTER_XY[1]) < inner_half_y
        and z < TRAY_SUCCESS_MAX_Z
    )
