# 17_franka_smolvla_data_collection

这个目录用于做 **Franka + 双相机 + 专家策略** 的示教数据采集，目标是为后续
**SmolVLA / LeRobot 风格训练** 准备一个干净、稳定、容易再转换的数据源。

当前版本先解决第一步：

1. 有一个稳定的 Isaac Sim 场景
2. 方块会随机出现在机械臂可抓取范围
3. 不用键盘控制
4. 由内置专家控制器自动抓取和放置
5. 采集两路图像、机器人状态、任务空间动作真值
6. 每个 episode 保存为一个 `.npz`

## 运行方式

请使用 Isaac Sim 自己的 Python 环境运行：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/demo.py
```

如果你想无界面采集：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/demo.py --headless
```

如果你想多采几条：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/demo.py --episodes 50
```

## 动作回放

如果你想在 Isaac Sim 里重新执行某条已采集的轨迹，可以运行：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py
```

回放指定 episode：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py --episode 1
```

循环回放：

```bash
python isaac-sim-learning-demos/17_franka_smolvla_data_collection/replay_episode.py --episode 1 --loop
```

说明：

1. 当前回放的是 `npz` 里的动作序列，不是直接播放图像
2. 也就是说，Isaac 会重新搭场景，并按保存下来的 `action` 逐帧驱动 Franka
3. 当前 `action` 语义是：`target_ee_xyz + target_gripper_closed`
4. 因为 raw 数据里没有保存每一帧完整物理真值，所以它是“动作重放”，不是严格视频级状态复刻

## 当前采集内容

每个 episode 会保存以下核心字段：

- `observation.images.front`
  前视相机 RGB 图像序列，形状大致为 `(T, H, W, 3)`
- `observation.images.wrist`
  手腕相机 RGB 图像序列，形状大致为 `(T, H, W, 3)`
- `observation.state`
  状态序列，当前定义为 15 维：
  - 7 个机械臂关节
  - 3 个末端位置 xyz
  - 4 个末端姿态四元数 `wxyz`
  - 1 个夹爪开口宽度
- `action`
  动作序列，当前定义为 4 维：
  - 末端目标位置 `x y z`
  - 夹爪闭合标记 `0/1`
- `next.reward`
  简单奖励，当前只用“方块是否已经进入托盘”做 0/1 标记
- `next.done`
  结束标记
- `state_names`
  状态维度名字
- `action_names`
  动作维度名字
- `task`
  文本任务描述
- `success`
  这个 episode 最终是否放置成功
- `metadata_json`
  采集配置说明

## 为什么先保存成 NPZ

因为你当前主要目标是先把 Isaac 中的数据稳定采下来，而不是一开始就把采集器和
训练框架死绑。

这样做有几个好处：

1. 先把“数据是否采对”单独验证清楚
2. 后续你想转 LeRobot、SmolVLA、自定义 replay buffer 都很方便
3. 采集脚本不会被某个训练库的 Python 版本或依赖反向卡死

## 当前动作真值设计

当前版本里，训练真值不再保存关节目标，而是直接保存更贴近任务语义的：

1. `target_ee_pos_x`
2. `target_ee_pos_y`
3. `target_ee_pos_z`
4. `target_gripper_closed`

这样做的原因是：

1. VLA 更容易学习“手要去哪里、夹爪该不该闭合”
2. 末端空间动作比关节空间动作更稳定、更通用
3. 后续如果你切控制器或做 sim2real，任务空间标签更容易复用

## 当前专家策略

当前不是键盘控制，而是使用脚本生成的任务空间专家轨迹：

- `target_ee_xyz + target_gripper_closed`
- `RMPFlowController`
- 真实物理抓取和释放

它会按更细的阶段完成：

1. 移到方块高悬停点
2. 移到方块低悬停点
3. 慢速下探到抓取高度
4. 停顿并合拢夹爪
5. 保持夹爪闭合等待接触稳定
6. 慢速抬起
7. 转移到托盘上方
8. 慢速下降到放置高度
9. 张开夹爪释放
10. 停顿等待方块自然落稳
11. 抬起撤退

这比键盘遥操作更稳定，也比旧版本的“吸附式抓取”更接近真实物理过程。

## 当前随机化范围

当前只对训练方块做随机化，范围故意收得比较稳：

- `x` 在 `0.42 ~ 0.44`
- `y` 在 `-0.02 ~ 0.02`

原因很简单：第一版优先保证成功率和数据质量，不急着一开始就把随机化拉满。

## 下一步建议

这个版本完成后，下一步最合理的是继续做这几件事：

1. 增加更多任务字段，比如语言指令、episode 级标签
2. 增加导出脚本，把当前 NPZ 转成 LeRobot 数据集目录结构
3. 加入多种方块颜色、大小和位置随机化
4. 再决定是否扩展到双物体、障碍物、失败轨迹混采

如果你接下来要继续往 **SmolVLA 真正训练数据集** 走，最优先的下一步就是：

**写一个“NPZ -> LeRobot/SmolVLA 训练格式”的转换脚本。**
