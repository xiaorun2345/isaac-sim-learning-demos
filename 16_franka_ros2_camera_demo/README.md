# 16 Franka ROS 2 Camera Demo

This demo packages a complete beginner-friendly ROS 2 camera workflow around a Franka scene.

Goal:

- build a standalone Isaac Sim Franka scene
- create a front overview camera and a wrist camera
- publish the front camera RGB stream over ROS 2
- subscribe to the image stream from an external ROS 2 Python node

Topics:

- publish `/front_camera/rgb` as `sensor_msgs/msg/Image`

Files:

- `demo.py`: Isaac Sim side scene builder and ROS 2 camera publisher
- `subscriber.py`: external ROS 2 side image subscriber

Suggested workflow:

1. Source ROS 2 in the terminal that launches Isaac Sim.
2. Start the Isaac Sim scene publisher:

```bash
python demo.py
```

3. In another ROS 2 sourced terminal, check the topic:

```bash
ros2 topic list
ros2 topic info /front_camera/rgb
```

4. Start the subscriber:

```bash
python subscriber.py
```

5. Optionally inspect the image visually:

```bash
ros2 run rqt_image_view rqt_image_view
```

Expected communication chain:

- Franka scene camera -> Isaac Sim ROS 2 bridge -> `/front_camera/rgb` -> `subscriber.py`

Notes:

- Run `demo.py` from an Isaac Sim Python environment.
- Run `subscriber.py` from a ROS 2 Python environment with `rclpy` and `sensor_msgs`.
- If your ROS 2 domain is customized, keep the Isaac Sim terminal and subscriber terminal on the same `ROS_DOMAIN_ID`.

Official references:

- [ROS 2 Cameras](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html)
- [Publishing Camera Data](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera_publishing.html)
