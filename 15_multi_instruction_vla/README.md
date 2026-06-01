# 15 Multi-Instruction VLA

This final demo shows a reusable task runner that handles multiple language instructions.

What it demonstrates:

- instruction-conditioned observation creation
- repeated policy calls
- per-episode logging
- a clean path toward task evaluation

This is the smallest example in the series that looks like a real evaluation harness rather than a one-off script.

Files:

- `demo.py`: offline multi-instruction evaluation walkthrough
- `ros2_vla_node.py`: integrated ROS 2 and GR00T node

The integrated node closes the loop:

1. subscribe `/isaac/camera/rgb`
2. subscribe `/franka/joint_state`
3. subscribe `/vla/instruction`
4. send image, language, and simulated robot feedback to the GR00T policy server
5. decode the first action from the returned action chunk
6. publish `/franka/joint_command`
7. let Demo 10 apply those commands to Franka

Required ROS 2 packages:

- `rclpy`
- `cv_bridge`
- `sensor_msgs`
- `std_msgs`

Start the components in separate terminals:

```powershell
python.bat ..\09_ros2_basics\camera_publisher.py
python.bat ..\10_ros2_franka_control\demo.py
python ros2_vla_node.py
ros2 topic pub /vla/instruction std_msgs/msg/String "{data: 'pick the red cube'}"
```

Also start the GR00T policy server described in Demo 13 before starting `ros2_vla_node.py`.
