# 20_franka_multi_station_ros2_collection

这个 demo 解决的是多工位里最常见的两个工程问题：

1. ROS 2 到底该连哪一台机械臂、哪一路相机
2. 数据采集到底只采一个工位，还是同时采全部工位

它建立在 `19_franka_multi_station_scene` 的多工位布局之上，但把重点放在
**路由和选择逻辑**，而不是专家策略本身。

## 这个 demo 能做什么

### 1. 机械臂 ROS 2 路由

支持三种模式：

- `--ros2-arm off`
  不启用机械臂 ROS 2 桥
- `--ros2-arm single --ros2-arm-env 2`
  只让 `env_02` 的 Franka 跟 ROS 2 通信
- `--ros2-arm all`
  让全部工位的 Franka 都跟 ROS 2 通信

话题规则：

- 单工位命令：`/env_XX/franka/joint_command`
- 单工位状态：`/env_XX/franka/joint_state`
- 全工位广播命令：`/all_envs/franka/joint_command`

命令消息类型：

- `std_msgs/msg/Float64MultiArray`
- 数据长度固定 9
- 含义：7 个机械臂关节 + 2 个夹爪关节

状态消息类型：

- `sensor_msgs/msg/JointState`

### 2. 相机 ROS 2 路由

支持三种模式：

- `--ros2-camera off`
- `--ros2-camera single --ros2-camera-env 1`
- `--ros2-camera all`

并且支持指定哪一路相机：

- `--ros2-camera-name front`
- `--ros2-camera-name wrist`
- `--ros2-camera-name both`

话题规则：

- `/env_XX/front_camera/rgb`
- `/env_XX/wrist_camera/rgb`

消息类型：

- `sensor_msgs/msg/Image`

### 3. 本地采集路由

支持三种模式：

- `--collect off`
- `--collect single --collect-env 3`
- `--collect all`

采集内容：

- `observation.images.front`
- `observation.images.wrist`
- `observation.state`
- `command.joint_position`
- `timestamp_sec`

保存方式：

- 每个工位各自保存一个 `.npz`
- 即使你是 `--collect all`，也不会把所有工位混成一个文件
- 文件目录形如：`outputs/raw/run_YYYYMMDD_HHMMSS/env_XX.npz`

## 最常用运行示例

### 只让某一个机械臂和 ROS 2 通信

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --ros2-arm single \
  --ros2-arm-env 2
```

这时你只需要对：

- `/env_02/franka/joint_command`

发命令即可。

### 让全部机械臂都跟 ROS 2 通信

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --ros2-arm all
```

这时你有两种控制方式：

1. 分别控制：
   - `/env_00/franka/joint_command`
   - `/env_01/franka/joint_command`
   - `/env_02/franka/joint_command`
   - `/env_03/franka/joint_command`
2. 广播同一条命令到全部工位：
   - `/all_envs/franka/joint_command`

### 只发布某一个工位的前视相机

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --ros2-camera single \
  --ros2-camera-env 1 \
  --ros2-camera-name front
```

对应话题：

- `/env_01/front_camera/rgb`

### 发布全部工位的前视和手腕相机

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --ros2-camera all \
  --ros2-camera-name both
```

### 只采某一个工位

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --collect single \
  --collect-env 2 \
  --collect-steps 300
```

### 同时采全部工位

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --collect all \
  --collect-steps 300
```

### 一边用 ROS 2 控一个工位，一边只采这个工位

```bash
python isaac-sim-learning-demos/20_franka_multi_station_ros2_collection/demo.py \
  --ros2-arm single \
  --ros2-arm-env 1 \
  --ros2-camera single \
  --ros2-camera-env 1 \
  --ros2-camera-name both \
  --collect single \
  --collect-env 1 \
  --collect-steps 300
```

这就是最接近“针对某一台机械臂做 ROS 2 控制和数据采集”的模式。

## 这个 demo 的定位

它不是 `17` 那种自动专家采集器，而是一个：

- 多工位路由示例
- ROS 2 命名规范示例
- 单工位 / 全工位选择逻辑示例

如果你后面要继续扩展成真正的并行专家采集器，最自然的方向是：

1. 保留本 demo 的 `env_XX` 路由规则
2. 把 `17` 的专家轨迹生成逻辑移植到每个工位
3. 再决定是每个工位独立 episode，还是批量并行 episode
