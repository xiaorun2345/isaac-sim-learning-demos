# 18_franka_isaac_lab_mimic

这个示例把 Demo 17 的 **Franka 抓取红色方块并放入木质托盘** 场景，完整迁移为
Isaac Lab `ManagerBasedRLMimicEnv`。

它不是把 Demo 17 的 `World + SingleManipulator + RMPFlowController` 直接复制过来，而是使用
Isaac Lab Manager-Based 架构重新表达同一个任务，使官方工具能够直接完成：

```text
人工或程序化种子演示
        ↓
自动标注子任务边界
        ↓
Isaac Lab Mimic 空间变换与轨迹拼接
        ↓
生成大规模 HDF5 专家数据
        ↓
Robomimic BC-RNN 训练
```

> Isaac Lab Mimic 当前只支持 Linux。建议在 Ubuntu 的 Isaac Lab 环境中运行本示例。

## 与 Demo 17 的关系

保留的场景语义：

- Franka Panda 机械臂和双指夹爪
- 目标物体为红色方块 `cube_2`
- 蓝色、绿色方块作为干扰物
- 红色方块初始范围：`x=0.42~0.44`、`y=-0.02~0.02`
- 木质托盘中心：`x=0.64`、`y=0.18`
- 前视相机与腕部相机
- 成功条件仍然是红色方块进入托盘内部

关键变化：

| Demo 17 | Demo 18 |
|---|---|
| Isaac Sim `World` | Isaac Lab Manager-Based 环境 |
| `SingleManipulator` | `ArticulationCfg` |
| RMPFlow 绝对 XYZ 目标 | 相对 IK 七维动作 |
| `[target_x, target_y, target_z, gripper]` | `[dx, dy, dz, dRx, dRy, dRz, gripper]` |
| 自定义 NPZ | 官方 Mimic HDF5 |
| 脚本阶段固定 | Mimic 子任务轨迹变换与拼接 |

## 文件结构

```text
18_franka_isaac_lab_mimic/
├── demo.py                         # 预览和验证环境
├── run_official.py                 # 注册自定义任务后运行官方 Isaac Lab 脚本
├── README.md
└── franka_tray_mimic/
    ├── __init__.py                 # Gym task 注册
    ├── scene_contract.py           # 与 Demo 17 对齐的纯 Python 场景参数
    ├── mdp.py                      # 托盘观测与成功判定
    ├── env_cfg.py                  # Manager-Based 场景、相机、事件、Mimic 配置
    └── mimic_env.py                # ManagerBasedRLMimicEnv 接口实现
```

自定义任务 ID：

```text
Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0
```

## Mimic 子任务设计

本任务只拆分为三个子任务。子任务过多会增加轨迹拼接次数，并降低生成成功率。

| 序号 | 子任务 | 参考物体 | 自动结束信号 |
|---|---|---|---|
| 1 | 抓取红色方块 | `cube_2` | `grasp` |
| 2 | 把红色方块移动进托盘 | `tray` | `placed_in_tray` |
| 3 | 释放方块并撤退 | `tray` | 最终任务成功，无需单独信号 |

`mimic_env.py` 实现或继承了 Mimic 所需接口：

```python
get_robot_eef_pose()
target_eef_pose_to_action()
action_to_target_eef_pose()
actions_to_gripper_actions()
get_object_poses()
get_subtask_term_signals()
```

其中前四项继承自官方 Franka 相对 IK Mimic 环境；本示例实现托盘参考位姿和自定义子任务信号。

## 1. 预览环境

在 Isaac Lab 根目录运行，并把本仓库路径替换为实际路径：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/demo.py \
    --num_envs 1
```

该脚本使用零动作运行环境，主要检查：

- 自定义任务是否成功注册
- Franka、方块、托盘和双相机是否加载
- 动作空间是否为相对 IK 动作
- observation 中是否存在 `policy` 和 `subtask_terms`

## 2. 采集种子专家演示

`run_official.py` 会先注册自定义 Gym task，再执行 Isaac Lab 官方脚本。

先创建数据目录：

```bash
mkdir -p datasets/franka_tray
```

使用键盘采集 20 条成功演示：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/tools/record_demos.py \
    --task Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0 \
    --viz kit \
    --dataset_file ./datasets/franka_tray/source.hdf5 \
    --num_demos 20 \
    --teleop_device keyboard
```

键盘控制：

```text
W/S：沿 X 移动
A/D：沿 Y 移动
Q/E：沿 Z 移动
Z/X、T/G、C/V：旋转末端
K：开关夹爪
R：丢弃当前失败演示并重置
```

当前 Demo 17 的 NPZ 动作是绝对 XYZ，不可以直接当作本环境的相对七维 IK 动作使用。若要复用
Demo 17 的程序化专家，需要在本环境中重新执行专家目标，并记录 `env.step()` 使用的相对七维动作。

## 3. 回放种子演示

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/tools/replay_demos.py \
    --task Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0 \
    --viz kit \
    --num_envs 1 \
    --reset_sim_buffer_each_episode \
    --dataset_file ./datasets/franka_tray/source.hdf5
```

## 4. 自动标注子任务

环境 observation 中包含：

```text
subtask_terms.grasp
subtask_terms.placed_in_tray
```

因此可以使用 `--auto` 自动寻找信号的上升沿并切分子任务：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0 \
    --viz kit \
    --auto \
    --input_file ./datasets/franka_tray/source.hdf5 \
    --output_file ./datasets/franka_tray/annotated.hdf5
```

建议先回放 `annotated.hdf5`，确认 `grasp` 与 `placed_in_tray` 的边界符合预期。

## 5. 小规模生成 Mimic 数据

先使用少量环境和少量轨迹进行可视化检查：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --viz kit \
    --num_envs 10 \
    --generation_num_trials 20 \
    --input_file ./datasets/franka_tray/annotated.hdf5 \
    --output_file ./datasets/franka_tray/generated_small.hdf5
```

重点检查：

- 抓取阶段是否始终相对红色方块变换
- 搬运阶段是否始终相对托盘变换
- 插值段是否碰撞托盘壁
- 夹爪闭合和释放时机是否稳定

## 6. 批量生成专家数据

小规模结果稳定后再无界面并行生成：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --headless \
    --num_envs 128 \
    --generation_num_trials 1000 \
    --input_file ./datasets/franka_tray/annotated.hdf5 \
    --output_file ./datasets/franka_tray/generated.hdf5
```

显存不足时降低 `--num_envs`。如果成功率较低，优先调整：

```python
subtask_term_offset_range
action_noise
num_interpolation_steps
```

这些参数位于 `franka_tray_mimic/env_cfg.py` 的 `FrankaTrayIKRelMimicEnvCfg`。

## 7. 训练 BC-RNN

安装 Robomimic：

```bash
sudo apt install cmake build-essential
./isaaclab.sh -i robomimic
```

训练：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0 \
    --algo bc \
    --dataset ./datasets/franka_tray/generated.hdf5
```

运行训练好的策略：

```bash
./isaaclab.sh -p /path/to/isaac-sim-learning-demos/18_franka_isaac_lab_mimic/run_official.py \
    scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Pick-Red-Cube-To-Tray-Franka-IK-Rel-Mimic-v0 \
    --viz kit \
    --num_rollouts 50 \
    --checkpoint /path/to/model_epoch_xxx.pth
```

不同 epoch 的成功率可能差异较大，不要只测试最后一个 checkpoint。

## 8. 后续接入 SmolVLA

建议先使用 BC-RNN 验证 Mimic 数据是否真的可学习。BC-RNN 能稳定完成任务后，再进行：

```text
Mimic HDF5
    ↓
转换为 LeRobotDataset
    ↓
保留 front_camera、wrist_camera、机器人状态、七维相对动作和任务文本
    ↓
训练 SmolVLA
```

SmolVLA 的语言指令可以使用：

```text
Pick up the red cube and place it into the wooden tray.
```

## 常见问题

### 为什么不继续使用 RMPFlow？

Mimic 需要能够把动作和末端目标位姿互相转换。官方 Franka 相对 IK 环境已经完整实现这组接口，
因此更适合用于 Mimic 的刚体变换和轨迹拼接。Demo 17 的 RMPFlow 专家仍适合生成种子轨迹，但需要
改写为在本环境中输出相对七维动作。

### 为什么托盘不是可运动的 RigidObject？

托盘在 Demo 17 中是固定目标。Mimic 只需要一个稳定的 `tray` 参考坐标系来变换搬运轨迹，
因此本示例把托盘壁建成静态碰撞体，并在 `get_object_poses()` 中提供托盘参考位姿。

### 为什么仍然保留蓝色和绿色方块？

它们与 Demo 17 一样作为视觉和运动干扰物，也让后续训练视觉策略时不至于只看到单一红方块场景。
