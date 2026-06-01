# 14 GR00T Pick Place

This demo expands the policy-control loop into repeated steps:

1. read the latest scene observation
2. query the policy
3. convert policy output into robot targets
4. send those targets to the robot
5. repeat

Important concept:

Real VLA policies often emit action chunks or end-effector deltas, not final joint positions. The control stack must convert those outputs into something the robot can actually execute.

Files:

- `demo.py`: offline readable loop
- `closed_loop_client.py`: real `PolicyClient` loop with a small `FrankaControlAdapter`
- `isaac_lab_policy_loop.py`: Isaac Lab scene, camera reads, Franka state reads, GR00T server calls, and joint target writes in one loop

The adapter deliberately isolates three robot-specific operations:

- `capture_rgb()`
- `read_joint_positions()`
- `apply_joint_targets()`

Replace those methods with reads and writes from the Isaac Lab scene in Demo 11. The rest of the GR00T control loop stays unchanged.

Set the action key to match your checkpoint:

```powershell
$env:GROOT_ACTION_KEY="joints"
python closed_loop_client.py
```

Run the integrated Isaac Lab version after starting the policy server:

```powershell
$env:GROOT_HOST="127.0.0.1"
$env:GROOT_PORT="5555"
$env:GROOT_ACTION_KEY="joints"
python isaac_lab_policy_loop.py
```

The selected GR00T checkpoint must return a 9-value Franka joint action under the configured action key. For other embodiments, change the action decoder and controller mapping explicitly.
