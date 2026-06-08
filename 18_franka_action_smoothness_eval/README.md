# 18_franka_action_smoothness_eval

这个 demo 专门解决一个很实际的问题：

**为什么采集出来的数据拿去训练后，机械臂动作会抖？**

很多时候原因不只一个，常见有两类：

1. `action` 标签本身跳变太硬，模型学到的就是“突然切目标”
2. 数据回放后的真实末端轨迹已经有反向修正、小幅来回摆动或夹爪抖动

这个目录里的脚本就是用来做这件事的：

1. 读取 `17_franka_smolvla_data_collection` 产出的 `.npz`
2. 逐条 episode 评估平滑性
3. 区分“标签不平滑”和“执行不平滑”
4. 输出一个可排序的分数表，帮你先清理坏数据，再去训练

## 运行方式

默认直接评估 `17` 目录下现有的 raw 数据：

```bash
python isaac-sim-learning-demos/18_franka_action_smoothness_eval/demo.py
```

只评估某一条：

```bash
python isaac-sim-learning-demos/18_franka_action_smoothness_eval/demo.py --episode 1
```

## 输出内容

脚本会在当前目录下生成：

- `reports/smoothness_report.csv`
- `reports/smoothness_report.json`
- `reports/plots/smoothness_scores.png`
- `reports/plots/episode_xxxxx_smoothness.png`

其中每条数据都会包含：

- `smoothness_score`
  平滑性总分，范围 `0 ~ 100`
- `smoothness_label`
  等级标签：`优秀 / 良好 / 警惕 / 抖动明显`
- `action_step_p95_mm`
  标签目标位姿逐帧变化的 P95
- `action_jerk_p95`
  标签的 jerk 指标，越大说明标签切换越硬
- `ee_step_p95_mm`
  真实末端执行轨迹逐帧位移的 P95
- `ee_accel_p95`
  真实末端加速度的 P95
- `ee_jerk_p95`
  真实末端 jerk 的 P95
- `ee_reverse_ratio`
  真实末端轨迹中“方向反转”的比例
- `gripper_switch_count`
  夹爪开闭切换次数
- `gripper_chatter_count`
  夹爪短时间反复切换次数
- `suggestions`
  脚本自动生成的中文诊断建议

同时，每条 episode 还会生成一张总览图，图里会直接画出：

- 动作标签目标位置曲线
- 真实末端执行位置曲线
- 逐帧位移大小
- 加速度近似
- jerk 近似
- 夹爪目标与实际宽度
- 俯视路径
- 最终平滑性分数和中文建议

所以你不需要只看数字，直接看图就能判断：

1. 是标签切换太硬
2. 还是执行轨迹本身在来回抖
3. 抖动主要出现在哪一个时间段

## 怎么理解结果

你可以直接按这个顺序看：

1. 先按 `smoothness_score` 升序排序
2. 先删掉 `抖动明显` 的数据
3. 再重点看 `警惕` 级别的数据

如果你看到：

- `action_jerk_p95` 很大，但 `ee_jerk_p95` 还可以  
  说明专家标签本身切换太硬，后续更适合做动作插值或标签平滑

- `ee_jerk_p95` 和 `ee_reverse_ratio` 都大  
  说明回放出来的运动已经在抖，这种数据更不适合直接训练

- `gripper_chatter_count` 大于 0  
  说明夹爪有短时间来回开关，训练后很容易出现抓取犹豫

## 推荐用法

最实用的流程是：

1. 先采一批数据
2. 用这个脚本跑一次平滑性评估
3. 删掉分数最低的坏轨迹
4. 再把剩余轨迹送去做 SmolVLA 微调

这样通常比“盲目多采几百条再训练”更有效，因为训练抖动往往首先是数据质量问题。
