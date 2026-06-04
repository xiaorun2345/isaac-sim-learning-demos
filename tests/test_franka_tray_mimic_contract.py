import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "18_franka_isaac_lab_mimic"
    / "franka_tray_mimic"
    / "scene_contract.py"
)


def load_scene_contract():
    spec = importlib.util.spec_from_file_location("franka_tray_scene_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FrankaTrayMimicContractTests(unittest.TestCase):
    def test_scene_contract_matches_demo_17_layout(self):
        contract = load_scene_contract()

        np.testing.assert_allclose(contract.TARGET_CUBE_X_RANGE, (0.42, 0.44))
        np.testing.assert_allclose(contract.TARGET_CUBE_Y_RANGE, (-0.02, 0.02))
        np.testing.assert_allclose(contract.TRAY_CENTER_XY, (0.64, 0.18))
        np.testing.assert_allclose(contract.CUBE_SIZE, (0.045, 0.045, 0.045))

    def test_cube_inside_tray_accepts_center_and_rejects_outside_position(self):
        contract = load_scene_contract()

        self.assertTrue(contract.is_position_inside_tray((0.64, 0.18, 0.04)))
        self.assertFalse(contract.is_position_inside_tray((0.80, 0.18, 0.04)))
        self.assertFalse(contract.is_position_inside_tray((0.64, 0.18, 0.20)))

    def test_mimic_subtasks_use_red_cube_then_tray_reference(self):
        contract = load_scene_contract()

        self.assertEqual(
            contract.MIMIC_SUBTASKS,
            (
                ("cube_2", "grasp"),
                ("tray", "placed_in_tray"),
                ("tray", None),
            ),
        )


if __name__ == "__main__":
    unittest.main()
