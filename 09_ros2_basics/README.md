# 09 ROS 2 Video Subscriber

This demo is the first real ROS 2 communication example in the series.

Goal:

- subscribe to the RGB image stream published from Isaac Sim
- inspect each incoming `sensor_msgs/msg/Image`
- print frame size, encoding, frame id, and approximate FPS

How it connects to Demo 04:

1. `04_camera_capture` creates the camera in Isaac Sim.
2. Isaac Sim ROS 2 Bridge publishes that camera to a topic such as `/isaac/camera/rgb`.
3. This demo subscribes to that topic using `rclpy`.

Required ROS 2 packages:

- `rclpy`
- `sensor_msgs`

Suggested workflow:

1. Source ROS 2 before launching Isaac Sim.
2. Start the included Isaac Sim publisher:

```powershell
python.bat C:\Users\Administrator\Documents\isaac_sim学习\09_ros2_basics\camera_publisher.py
```

3. Confirm that the topics exist:

```powershell
ros2 topic list
ros2 topic info /isaac/camera/rgb
```

4. In a ROS 2 sourced terminal, start the subscriber:

```powershell
python C:\Users\Administrator\Documents\isaac_sim学习\09_ros2_basics\demo.py
```

5. Optionally view the image:

```powershell
ros2 run rqt_image_view rqt_image_view
```

Expected communication chain:

- Isaac Sim camera -> ROS 2 bridge -> `/isaac/camera/rgb` -> this subscriber

Files:

- `camera_publisher.py`: Isaac Sim side, enables `isaacsim.ros2.bridge` and attaches ROS 2 Replicator writers
- `demo.py`: external ROS 2 side, subscribes with `rclpy`

Official references:

- [ROS 2 Cameras](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html)
- [Publishing Camera's Data](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/ros2_tutorials/tutorial_ros2_camera_publishing.html)
