# Isaac Sim 15 Demos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a progressive `01-15` Isaac Sim learning repository covering Core API, ROS 2, Isaac Lab, and GR00T-oriented VLA demos.

**Architecture:** Use one top-level README, one lightweight `common/` helper package, and fifteen standalone demo folders. Keep each `demo.py` readable and self-contained while extracting repeated setup code into small shared helpers once duplication stops being educational.

**Tech Stack:** Python, NVIDIA Isaac Sim Core API, ROS 2 bridge concepts, Isaac Lab, GR00T-oriented policy wrappers, NumPy

---

### Task 1: Create Repository Skeleton

**Files:**
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\__init__.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\core_utils.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\franka_utils.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\camera_utils.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\ros2_utils.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\common\groot_utils.py`

- [ ] **Step 1: Create the top-level README**

```markdown
# Isaac Sim Learning Demos

This repository contains 15 progressive demos:

- `01-10`: Isaac Sim Core API and ROS 2 foundations
- `11-15`: Isaac Lab plus GR00T-style VLA demos

Each demo is self-contained and includes:

- `demo.py`
- `README.md`

Run demos from the Isaac Sim or Isaac Lab Python environment that matches the demo.
```

- [ ] **Step 2: Create the shared package marker**

```python
"""Shared helpers for the Isaac Sim learning demos."""
```

- [ ] **Step 3: Create `common/core_utils.py`**

```python
"""Small helpers for standalone Isaac Sim demos."""

from pxr import UsdLux


def add_basic_light(stage, intensity=500.0, path="/World/DistantLight"):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    return light
```

- [ ] **Step 4: Create `common/franka_utils.py`**

```python
"""Helpers for loading and commanding Franka."""

import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path


FRANKA_PRIM_PATH = "/World/Franka"


def get_franka_usd_path():
    return get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def load_franka(prim_path=FRANKA_PRIM_PATH, name="franka"):
    add_reference_to_stage(get_franka_usd_path(), prim_path)
    return SingleArticulation(prim_path=prim_path, name=name)


def home_action():
    return ArticulationAction(
        joint_positions=np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)
    )
```

- [ ] **Step 5: Create `common/camera_utils.py`**

```python
"""Helpers for simple camera configuration notes."""


def format_rgb_shape(rgb):
    if rgb is None:
        return "no frame"
    return f"frame shape={getattr(rgb, 'shape', 'unknown')}"
```

- [ ] **Step 6: Create `common/ros2_utils.py`**

```python
"""Helpers and placeholders for ROS 2 teaching demos."""


def ros2_not_ready_message():
    return "ROS 2 bridge setup is environment-dependent. Check this demo README before running."
```

- [ ] **Step 7: Create `common/groot_utils.py`**

```python
"""Helpers for GR00T-oriented VLA demos."""

import os


def get_groot_checkpoint():
    return os.environ.get("GROOT_CHECKPOINT", "")


def ensure_groot_checkpoint():
    checkpoint = get_groot_checkpoint()
    if not checkpoint:
        raise RuntimeError("Set GROOT_CHECKPOINT before running GR00T demos.")
    return checkpoint
```

- [ ] **Step 8: Commit**

```bash
git add README.md common
git commit -m "feat: add shared skeleton for isaac sim demos"
```

### Task 2: Create Core Isaac Sim Demos 01-04

**Files:**
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\01_minimal_scene\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\01_minimal_scene\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\02_lights_and_prims\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\02_lights_and_prims\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\03_multi_object_physics\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\03_multi_object_physics\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\04_camera_capture\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\04_camera_capture\README.md`

- [ ] **Step 1: Write `01_minimal_scene/demo.py`**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()
    for _ in range(240):
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

- [ ] **Step 2: Write `01_minimal_scene/README.md`**

```markdown
# 01 Minimal Scene

This demo shows the smallest useful standalone Isaac Sim program.

Run:

```powershell
python.bat demo.py
```
```

- [ ] **Step 3: Write `02_lights_and_prims/demo.py`**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.world import World

from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(VisualCuboid(prim_path="/World/BoxA", position=[0.0, 0.0, 0.5], size=0.3))
    world.scene.add(DynamicCuboid(prim_path="/World/BoxB", position=[0.5, 0.0, 0.5], size=0.2, color=[1.0, 0.2, 0.2]))
    world.reset()
    for _ in range(240):
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

- [ ] **Step 4: Write `02_lights_and_prims/README.md`**

```markdown
# 02 Lights And Primitives

This demo adds a light and a pair of simple cuboids so the scene becomes visually informative.
```

- [ ] **Step 5: Write `03_multi_object_physics/demo.py`**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.world import World

from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(FixedCuboid(prim_path="/World/Table", position=[0.6, 0.0, 0.2], scale=[0.6, 1.0, 0.4], color=[0.7, 0.7, 0.7]))
    for index in range(3):
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Cube_{index}",
                position=[0.1 * index, 0.0, 0.6 + 0.15 * index],
                size=0.08,
                color=[0.2 + 0.2 * index, 0.6, 0.2],
            )
        )
    world.reset()
    for _ in range(360):
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

- [ ] **Step 6: Write `03_multi_object_physics/README.md`**

```markdown
# 03 Multi-Object Physics

This demo introduces static geometry plus multiple dynamic cubes so you can observe gravity and collisions.
```

- [ ] **Step 7: Write `04_camera_capture/demo.py`**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.sensors.camera import Camera

from common.camera_utils import format_rgb_shape
from common.core_utils import add_basic_light


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    world.scene.add(DynamicCuboid(prim_path="/World/Cube", position=[0.4, 0.0, 0.3], size=0.15, color=[0.1, 0.8, 0.2]))
    camera = Camera(
        prim_path="/World/Camera",
        position=[1.5, 1.2, 1.2],
        frequency=20,
        resolution=(640, 480),
        orientation=[0.339, 0.176, 0.425, 0.820],
    )
    camera.initialize()
    world.reset()
    for step in range(240):
        world.step(render=True)
        if step % 60 == 0:
            rgba = camera.get_rgba()
            print(format_rgb_shape(rgba))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
```

- [ ] **Step 8: Write `04_camera_capture/README.md`**

```markdown
# 04 Camera Capture

This demo adds a camera sensor and prints frame-shape information while the simulation runs.
```

- [ ] **Step 9: Commit**

```bash
git add 01_minimal_scene 02_lights_and_prims 03_multi_object_physics 04_camera_capture
git commit -m "feat: add introductory isaac sim scene demos"
```

### Task 3: Create Franka Manipulation Demos 05-08

**Files:**
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\05_franka_joint_control\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\05_franka_joint_control\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\06_franka_gripper_sequence\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\06_franka_gripper_sequence\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\07_franka_pick_cube\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\07_franka_pick_cube\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\08_pick_place_state_machine\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\08_pick_place_state_machine\README.md`

- [ ] **Step 1: Write `05_franka_joint_control/demo.py` using `load_franka()` and `home_action()`**
- [ ] **Step 2: Write `05_franka_joint_control/README.md` explaining articulation initialization and a home pose**
- [ ] **Step 3: Write `06_franka_gripper_sequence/demo.py` with timed arm and gripper commands**
- [ ] **Step 4: Write `06_franka_gripper_sequence/README.md` describing joint subsets for gripper control**
- [ ] **Step 5: Write `07_franka_pick_cube/demo.py` with a cube, staged reach, grasp, and lift motions**
- [ ] **Step 6: Write `07_franka_pick_cube/README.md` describing pre-grasp and lift phases**
- [ ] **Step 7: Write `08_pick_place_state_machine/demo.py` with named states like `MOVE_ABOVE`, `LOWER`, `CLOSE`, `LIFT`, `MOVE_TO_DROP`, `OPEN`**
- [ ] **Step 8: Write `08_pick_place_state_machine/README.md` explaining why state machines are clearer than raw step counters**
- [ ] **Step 9: Commit**

```bash
git add 05_franka_joint_control 06_franka_gripper_sequence 07_franka_pick_cube 08_pick_place_state_machine
git commit -m "feat: add franka manipulation demos"
```

### Task 4: Create ROS 2 Demos 09-10

**Files:**
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\09_ros2_basics\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\09_ros2_basics\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\10_ros2_franka_control\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\10_ros2_franka_control\README.md`

- [ ] **Step 1: Write `09_ros2_basics/demo.py` as a guarded ROS 2 teaching example**

```python
from common.ros2_utils import ros2_not_ready_message


def main():
    print("Demo 09: ROS 2 basics")
    print(ros2_not_ready_message())
    print("Add your ROS 2 bridge startup here and publish simulation state topics.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `09_ros2_basics/README.md` with environment notes, expected bridge setup, and suggested topics**
- [ ] **Step 3: Write `10_ros2_franka_control/demo.py` as a guarded example showing where Franka command topics would be wired in**
- [ ] **Step 4: Write `10_ros2_franka_control/README.md` with expected command and state message flow**
- [ ] **Step 5: Commit**

```bash
git add 09_ros2_basics 10_ros2_franka_control
git commit -m "feat: add ros2 integration demos"
```

### Task 5: Create Isaac Lab and GR00T Demos 11-15

**Files:**
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\11_isaac_lab_minimal\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\11_isaac_lab_minimal\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\12_vla_observation_pipeline\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\12_vla_observation_pipeline\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\13_groot_inference\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\13_groot_inference\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\14_groot_pick_place\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\14_groot_pick_place\README.md`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\15_multi_instruction_vla\demo.py`
- Create: `C:\Users\Administrator\Documents\isaac_sim学习\15_multi_instruction_vla\README.md`

- [ ] **Step 1: Write `11_isaac_lab_minimal/demo.py` with an Isaac Lab app-launcher style skeleton and a short loop**
- [ ] **Step 2: Write `11_isaac_lab_minimal/README.md` describing how Isaac Lab entrypoints differ from standalone Isaac Sim**
- [ ] **Step 3: Write `12_vla_observation_pipeline/demo.py` showing image plus instruction plus robot-state packaging into a dict**
- [ ] **Step 4: Write `12_vla_observation_pipeline/README.md` describing VLA observation design**
- [ ] **Step 5: Write `13_groot_inference/demo.py` using `ensure_groot_checkpoint()` and a thin policy-wrapper demo**
- [ ] **Step 6: Write `13_groot_inference/README.md` documenting `GROOT_CHECKPOINT` and expected local setup**
- [ ] **Step 7: Write `14_groot_pick_place/demo.py` showing how policy output can be mapped into robot actions**
- [ ] **Step 8: Write `14_groot_pick_place/README.md` describing closed-loop inference and control**
- [ ] **Step 9: Write `15_multi_instruction_vla/demo.py` with several language instructions and a dispatch loop**
- [ ] **Step 10: Write `15_multi_instruction_vla/README.md` describing reusable task scaffolding for language-conditioned tasks**
- [ ] **Step 11: Commit**

```bash
git add 11_isaac_lab_minimal 12_vla_observation_pipeline 13_groot_inference 14_groot_pick_place 15_multi_instruction_vla
git commit -m "feat: add isaac lab and groot demos"
```

### Task 6: Verify and Polish

**Files:**
- Modify: `C:\Users\Administrator\Documents\isaac_sim学习\README.md`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\01_minimal_scene\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\02_lights_and_prims\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\03_multi_object_physics\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\04_camera_capture\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\05_franka_joint_control\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\06_franka_gripper_sequence\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\07_franka_pick_cube\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\08_pick_place_state_machine\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\09_ros2_basics\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\10_ros2_franka_control\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\11_isaac_lab_minimal\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\12_vla_observation_pipeline\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\13_groot_inference\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\14_groot_pick_place\demo.py`
- Verify: `C:\Users\Administrator\Documents\isaac_sim学习\15_multi_instruction_vla\demo.py`

- [ ] **Step 1: Expand the top-level README with a numbered table of demos and run guidance**
- [ ] **Step 2: Run syntax verification on all created Python files**

```bash
python -m py_compile common\*.py 01_minimal_scene\demo.py 02_lights_and_prims\demo.py 03_multi_object_physics\demo.py 04_camera_capture\demo.py 05_franka_joint_control\demo.py 06_franka_gripper_sequence\demo.py 07_franka_pick_cube\demo.py 08_pick_place_state_machine\demo.py 09_ros2_basics\demo.py 10_ros2_franka_control\demo.py 11_isaac_lab_minimal\demo.py 12_vla_observation_pipeline\demo.py 13_groot_inference\demo.py 14_groot_pick_place\demo.py 15_multi_instruction_vla\demo.py
```

Expected: command exits cleanly with no syntax errors

- [ ] **Step 3: Manually inspect README files for numbering, naming consistency, and dependency notes**
- [ ] **Step 4: Commit**

```bash
git add README.md common 01_minimal_scene 02_lights_and_prims 03_multi_object_physics 04_camera_capture 05_franka_joint_control 06_franka_gripper_sequence 07_franka_pick_cube 08_pick_place_state_machine 09_ros2_basics 10_ros2_franka_control 11_isaac_lab_minimal 12_vla_observation_pipeline 13_groot_inference 14_groot_pick_place 15_multi_instruction_vla
git commit -m "feat: add complete isaac sim learning demo series"
```
