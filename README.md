# Isaac Sim Learning Demos

This repository contains 20 progressive Python demos for learning:

- Isaac Sim standalone Python scripting
- Scene building and sensors
- Franka articulation control and grasping
- ROS 2 integration concepts
- Isaac Lab structure
- GR00T-style VLA inference workflows

## Learning Path

### Core Isaac Sim

1. `01_minimal_scene`
2. `02_lights_and_prims`
3. `03_multi_object_physics`
4. `04_camera_capture`
5. `05_franka_joint_control`
6. `06_franka_gripper_sequence`
7. `07_franka_pick_cube`
8. `08_pick_place_state_machine`
9. `09_ros2_basics`
10. `10_ros2_franka_control`

### Isaac Lab + VLA

11. `11_isaac_lab_minimal`
12. `12_vla_observation_pipeline`
13. `13_groot_inference`
14. `14_groot_pick_place`
15. `15_multi_instruction_vla`
16. `16_franka_ros2_camera_demo`
17. `17_franka_smolvla_data_collection`
18. `18_franka_isaac_lab_mimic`
19. `19_franka_multi_station_scene`
20. `20_franka_multi_station_ros2_collection`

## Repository Layout

- `common/`: shared helpers
- `01_*` to `20_*`: one demo folder per lesson
- `docs/superpowers/`: design and plan documents
- [`docs/franka-panda-introduction.md`](docs/franka-panda-introduction.md): illustrated Franka Panda beginner guide
- [`docs/franka-panda-hand-frame-guide.md`](docs/franka-panda-hand-frame-guide.md): realistic gripper frame, TCP, and wrist-camera guide
- [`docs/franka-smolvla-scene-tree-guide.md`](docs/franka-smolvla-scene-tree-guide.md): illustrated USD Stage tree for Demo 17

Each demo folder contains:

- `demo.py`
- `README.md`

## How To Run

### Isaac Sim demos

Run `01-10` from an Isaac Sim Python environment. A typical Windows pattern is:

```powershell
python.bat .\01_minimal_scene\demo.py
```

### Isaac Lab demos

Run `11-15` from an Isaac Lab environment that can import both Isaac Lab and the required NVIDIA robotics stack.

## Notes

- Some imports and asset paths can vary slightly between Isaac Sim versions.
- The ROS 2 demos are intentionally teaching-oriented and may need local bridge setup.
- Demo `09` includes both the Isaac Sim camera publisher and the external ROS 2 image subscriber.
- Demo `10` includes both the Isaac Sim Franka bridge and an external ROS 2 command publisher.
- Demo `16` includes a full Franka scene plus a ROS 2 front-camera image subscriber example.
- Demo `17` generates dual-camera scripted-expert data for SmolVLA-style training.
- Demo `18` migrates Demo 17's scene into a complete Isaac Lab Mimic environment and BC-RNN pipeline.
- `18_franka_action_smoothness_eval` provides an offline smoothness report for Demo 17 trajectories.
- Demo `19` replicates the Franka workcell into a 4-6 station multi-env scene for easier parallel data collection.
- Demo `20` shows how to route ROS 2 control/camera links and local recording to one workcell or all workcells.
- Demos `13-15` include offline walkthroughs plus real GR00T N1.7 `PolicyClient` examples.
- Real GR00T observation and action keys must match the embodiment configuration selected on the policy server.
