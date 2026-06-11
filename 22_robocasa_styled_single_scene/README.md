# 22_robocasa_styled_single_scene

这个 demo 是单场景装修版，不会改动原始的 `build_franka_scene_only.py`。

特点：

1. 保留单 Franka、桌子、托盘、方块和双相机
2. 房间、地面、柜体、吊灯、桌面细节更丰富
3. 支持切换多种 RoboCasa 风格装修主题
4. 如果本机有本地 RoboCasa USD 资产目录，会尝试挂一部分装饰件

## 运行

```bash
python isaac-sim-learning-demos/22_robocasa_styled_single_scene/demo.py
```

```bash
python isaac-sim-learning-demos/22_robocasa_styled_single_scene/demo.py --theme warm_walnut
```

```bash
python isaac-sim-learning-demos/22_robocasa_styled_single_scene/demo.py \
  --theme coastal_bright \
  --robocasa-asset-root /path/to/robocasa/usd_assets
```
