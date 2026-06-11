# 21_robocasa_styled_multi_room_scene

这个 demo 把 Franka 多工位场景做成一个“多装修主题展厅”：

1. 每个工位仍然保留 Franka、桌子、托盘、方块和双相机
2. 每个工位放进不同装修风格的小房间里
3. 装修配色和层次参考 RoboCasa 常见的家居 / 厨房场景语言
4. 如果本机提供了 RoboCasa 的本地 USD 资产目录，会额外挂一部分装饰件

## 运行

```bash
python isaac-sim-learning-demos/21_robocasa_styled_multi_room_scene/demo.py
```

```bash
python isaac-sim-learning-demos/21_robocasa_styled_multi_room_scene/demo.py --num-envs 6
```

```bash
python isaac-sim-learning-demos/21_robocasa_styled_multi_room_scene/demo.py \
  --num-envs 4 \
  --robocasa-asset-root /path/to/robocasa/assets
```

## 说明

- 如果没有提供 `--robocasa-asset-root`，脚本会使用程序化装修件，不会报错
- 当前可选真实资产只会尝试挂载本地目录里的 `.usd/.usda/.usdc`
- 由于 RoboCasa 官方厨房资产本身需要额外下载，本 demo 默认不假设这些文件已经存在
