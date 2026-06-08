# 会话摘要

更新时间：2026-06-03

## 当前目录

- 代码目录：`/home/mkls/xiao_run/isaac_openvla/isaac-sim-learning-demos/17_franka_smolvla_data_collection`
- 主脚本：`demo.py`
- 说明文件：`README.md`

## 当前目标

这个 demo 用于：

1. 在 Isaac Sim 中搭建 Franka 抓取场景
2. 用专家策略自动采集 SmolVLA 风格数据
3. 记录双相机图像、机器人状态和动作真值
4. 为后续转换成 LeRobot / SmolVLA 训练数据做准备

## 已完成内容

### 1. 新 demo 已创建并改名

原先临时目录名是：

- `17_frannkaxx`

后来已经改成正式目录名：

- `17_franka_smolvla_data_collection`

### 2. 采集脚本已经写好

当前采集脚本特性：

1. 不使用键盘控制
2. 使用 Isaac 内置 `PickPlaceController` 做专家采集
3. 随机生成可抓取范围内的方块位置
4. 保存前视相机和手腕相机图像
5. 保存状态和动作真值
6. 每个 episode 输出一个压缩 `npz`

### 3. 当前状态定义

当前 `observation.state` 为 11 维：

1. `panda_joint1`
2. `panda_joint2`
3. `panda_joint3`
4. `panda_joint4`
5. `panda_joint5`
6. `panda_joint6`
7. `panda_joint7`
8. `ee_pos_x`
9. `ee_pos_y`
10. `ee_pos_z`
11. `gripper_width`

### 4. 当前动作真值定义

当前 `action` 已经不是关节目标，而是 4 维任务空间动作：

1. `target_ee_pos_x`
2. `target_ee_pos_y`
3. `target_ee_pos_z`
4. `target_gripper_closed`

说明：

1. 这是根据专家控制器阶段推导出来的任务空间真值
2. 当前是“末端绝对目标位置 + 夹爪闭合标记”
3. 还没有改成 `delta_ee_xyz`

## 已采集的数据

本地已经采集了 2 条 raw 数据：

1. `outputs/raw/episode_00000.npz`
2. `outputs/raw/episode_00001.npz`

验证结果：

1. 文件已正常生成
2. 每条数据包含双相机图像、状态、动作、reward、done
3. 当前两条 `success=False`

大致形状：

- `observation.images.front`: `(360, 480, 640, 3)`
- `observation.images.wrist`: `(360, 480, 640, 3)`
- `observation.state`: `(360, 11)`
- `action`: `(360, 4)`

## Git 与远端状态

### 已提交并推送

- 仓库：`/home/mkls/xiao_run/isaac_openvla/isaac-sim-learning-demos`
- 远端：`https://github.com/xiaorun2345/isaac-sim-learning-demos`
- commit：`d0169af`
- commit message：`Add Franka SmolVLA data collection demo`

### 已上传内容

已上传：

1. `17_franka_smolvla_data_collection/demo.py`
2. `17_franka_smolvla_data_collection/README.md`
3. `.gitignore`

未上传：

1. `16_franka_ros2_camera_demo/run_demo.sh`
2. `16_franka_ros2_camera_demo/run_subscriber.sh`
3. `17_franka_smolvla_data_collection/outputs/raw/*.npz`

## 当前忽略规则

`.gitignore` 已加入：

- `17_franka_smolvla_data_collection/outputs/`

因此后续采集的数据不会误传到 GitHub。

## 关键结论

1. SmolVLA 不固定要求输出关节增量
2. 它学习的是你数据集里定义的 `action`
3. 如果训练集动作定义变了，推理时也必须按同样语义解释
4. 训练好的权重不能直接“扔进 Isaac”，需要写单独的推理脚本接上观测和控制器

## 建议的下一步

优先顺序建议如下：

1. 先提高专家成功率，采集一批成功轨迹
2. 再写 `raw npz -> LeRobot` 转换脚本
3. 再决定是否把动作从“绝对末端位置”改成“末端增量”
4. 最后写 Isaac 版 `run_trained_policy.py`

## 下次继续时可以直接说

你下次可以直接说下面任意一句：

1. `继续 17_franka_smolvla_data_collection，先调专家成功率`
2. `继续 17_franka_smolvla_data_collection，先做 npz 转换脚本`
3. `继续 17_franka_smolvla_data_collection，把 action 改成 delta_ee_xyz`
