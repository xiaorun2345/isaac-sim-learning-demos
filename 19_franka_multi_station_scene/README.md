# 19_franka_multi_station_scene

这个 demo 把原来的单工位 `build_franka_scene_only.py` 扩展成了一个
**多 Franka 复制工位场景**，用于后续更方便地做并行观察、录屏或数据采集。

每个工位都包含：

1. 一台 Franka 机械臂
2. 一张桌子
3. 一组和原脚本一致的方块
4. 一个放置盒
5. 一个前视相机
6. 一个手腕相机

## 设计目标

这个版本不是把所有物体简单堆在一起，而是把每个工位按统一模板复制，并放到
一个间距合理的网格里：

1. 工位默认 4 个
2. 支持切换到 5 个或 6 个
3. 每个工位的局部结构与原单工位版本一致
4. 工位之间预留足够侧向间距，降低机械臂互相碰撞和画面遮挡的概率

## 运行方式

请使用 Isaac Sim 自己的 Python 环境运行：

```bash
python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py
```

无界面模式：

```bash
python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py --headless
```

切换工位数量：

```bash
python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py --num-envs 4
python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py --num-envs 5
python isaac-sim-learning-demos/19_franka_multi_station_scene/demo.py --num-envs 6
```

## 布局规则

- `4` 个工位时使用 `2 x 2`
- `5` 或 `6` 个工位时使用 `3 x 2`
- 所有工位使用同一朝向，便于后续统一控制和统一采集逻辑
- 每个工位都放在 `/World/Envs/env_XX` 下面，便于脚本按编号索引

## 适合后续怎么接

如果你后面要把它接进采集脚本，最自然的做法是：

1. 逐个遍历 `/World/Envs/env_00` 到 `/World/Envs/env_XX`
2. 为每个工位各自绑定控制器和相机传感器
3. 按工位编号分别保存 episode 或合并成批量采集任务

当前这个 demo 先只负责把多工位场景稳定搭起来，不直接做采集逻辑。
