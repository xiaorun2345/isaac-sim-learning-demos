# 16 Franka ROS 2 相机示例

这个示例基于 Franka 场景，演示一条完整但尽量简洁的 ROS 2 相机链路：

- 在 Isaac Sim 中搭建 Franka 场景
- 创建前视相机和腕部相机
- 通过 ROS 2 发布前视相机图像
- 在订阅端接收图像并直接显示画面

## 话题

- 发布话题：`/front_camera/rgb`
- 消息类型：`sensor_msgs/msg/Image`

## 文件说明

- `demo.py`：Isaac Sim 侧场景搭建与 ROS 2 图像发布
- `subscriber.py`：ROS 2 图像订阅与画面显示

## 运行步骤

1. 打开一个新终端，激活 Isaac Sim 使用的 conda 环境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/mkls/xiao_run/.conda-isaac-openvla
```

2. 进入当前示例目录并启动发布端：

```bash
cd /home/mkls/xiao_run/isaac_openvla/isaac-sim-learning-demos/16_franka_ros2_camera_demo
python demo.py
```

3. 在 Isaac Sim 界面中点击 `Play`，让 ROS 2 图像发布开始工作。

4. 再打开一个新终端，激活同一个 conda 环境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/mkls/xiao_run/.conda-isaac-openvla
```

5. 进入当前示例目录并启动订阅端：

```bash
cd /home/mkls/xiao_run/isaac_openvla/isaac-sim-learning-demos/16_franka_ros2_camera_demo
python subscriber.py
```

## 预期效果

- `demo.py` 运行后，会在 Isaac Sim 中创建 Franka 场景与相机
- 点击 `Play` 后，前视相机会持续向 `/front_camera/rgb` 发布图像
- `subscriber.py` 会弹出图像窗口，并实时显示订阅到的画面

## 注意事项

- `demo.py` 和 `subscriber.py` 都应在 `/home/mkls/xiao_run/.conda-isaac-openvla` 环境中运行
- 不要在运行本示例前执行 `source /opt/ros/humble/setup.bash`
- 如果当前终端已经自动加载过系统 ROS 2 环境，请重新开一个新终端再运行

## 官方参考

- [ROS 2 Cameras](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html)
- [Publishing Camera Data](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera_publishing.html)
