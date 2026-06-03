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

1. Start Isaac Sim side from a clean environment:

```bash
bash run_demo.sh
```

2. Start the Isaac Sim scene publisher:

```bash
bash run_demo.sh
```

3. In another terminal, use the ROS 2 subscriber environment:

```bash
bash run_subscriber.sh
```

4. Check the topic:

```bash
ros2 topic list
ros2 topic info /front_camera/rgb
```

5. Optionally inspect the image visually:

```bash
ros2 run rqt_image_view rqt_image_view
```

Expected communication chain:

- Franka scene camera -> Isaac Sim ROS 2 bridge -> `/front_camera/rgb` -> `subscriber.py`

Notes:

- `run_demo.sh` clears external ROS 2 Python variables and launches `demo.py` with `/home/mkls/xiao_run/.conda-isaac-openvla`.
- `run_subscriber.sh` sources `/opt/ros/humble/setup.bash` and runs `subscriber.py` with the system ROS 2 environment.

Official references:

- [ROS 2 Cameras](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html)
- [Publishing Camera Data](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera_publishing.html)
