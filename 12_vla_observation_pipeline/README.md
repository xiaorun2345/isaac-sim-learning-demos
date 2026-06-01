# 12 VLA Observation Pipeline

This demo expands the observation path into explicit stages:

1. capture camera image
2. read robot joint state
3. define language instruction
4. normalize and package the observation

Output fields:

- `rgb`
- `instruction`
- `joint_positions`
- `joint_velocities`

Why this matters:

`GR00T` and other VLA policies do not consume raw simulator objects directly. They consume a carefully packaged observation dictionary or tensor batch.

Files:

- `demo.py`: dependency-light walkthrough using generated RGB pixels and Franka-shaped state vectors
- `isaac_lab_observation_adapter.py`: reusable adapter for `scene["camera"]` and `scene["robot"]`

The adapter exposes two formats:

- `read_isaac_lab_observation()`: easy-to-print learning format
- `read_groot_policy_observation()`: batched nested structure for `PolicyClient.get_action()`
