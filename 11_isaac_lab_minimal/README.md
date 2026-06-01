# 11 Isaac Lab Minimal

This is the first detailed Isaac Lab example.

What it shows:

- `AppLauncher` based startup
- `SimulationContext`
- `InteractiveSceneCfg`
- spawning a ground plane and light through scene configuration
- stepping physics while updating the scene abstraction

This follows the structure shown in the official Isaac Lab tutorials:

- [Creating an empty scene](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/create_empty.html)
- [Using the Interactive Scene](https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/02_scene/create_scene.html)

Files:

- `demo.py`: smallest ground-plane and light example
- `camera_scene_demo.py`: Franka, red cube, and a `CameraCfg` RGB camera in one `InteractiveSceneCfg`

Run the richer scene from an Isaac Lab environment:

```powershell
python camera_scene_demo.py
```

The camera output is batched because Isaac Lab is designed to scale to many parallel environments.
