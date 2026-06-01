# 10 ROS 2 Franka Control

This demo turns Franka into a ROS 2 controlled robot inside Isaac Sim.

Topics:

- subscribe `/franka/joint_command` as `std_msgs/msg/Float64MultiArray`
- publish `/franka/joint_state` as `sensor_msgs/msg/JointState`

Data flow:

1. A ROS 2 node publishes 9 desired joint positions.
2. This demo receives them with `rclpy`.
3. The positions are converted into an Isaac Sim `ArticulationAction`.
4. Franka moves inside the simulation.
5. The latest simulated joint state is published back to ROS 2.

Example publisher command:

```powershell
ros2 topic pub /franka/joint_command std_msgs/msg/Float64MultiArray "{data: [0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04]}"
```

You can also use the included Python publisher:

```powershell
python C:\Users\Administrator\Documents\isaac_sim学习\10_ros2_franka_control\command_publisher.py
```

Run from an Isaac Sim Python environment that already has ROS 2 sourced:

```powershell
python.bat C:\Users\Administrator\Documents\isaac_sim学习\10_ros2_franka_control\demo.py
```

Official references:

- [ROS 2 Generic Publisher and Subscriber](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/index.html)
- [ROS 2 Publish Real Time Factor](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_rtf.html)
