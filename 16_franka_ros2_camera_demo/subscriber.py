"""
Franka 前视相机的 ROS 2 订阅示例。

这个脚本的目标很单纯：
- 订阅 Isaac Sim 发布的 `/front_camera/rgb`
- 每收到一帧图像，就打印一次图像尺寸、编码格式、坐标系和近似帧率

运行方式：
    python subscriber.py

说明：
- 需要在已经 source 好 ROS 2 的 Python 环境中运行。
- 这是一个“最小可读版”示例，因此不再依赖仓库里的额外工具函数。
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import tkinter as tk

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk


def get_writable_log_dir() -> Path:
    """返回一个尽量可写的 ROS 2 日志目录。"""

    candidates = []
    try:
        candidates.append(Path(tempfile.gettempdir()))
    except FileNotFoundError:
        pass
    candidates.append(Path("/dev/shm"))
    candidates.append(Path.cwd())

    for base in candidates:
        if base.exists() and os.access(base, os.W_OK):
            return base / "franka_ros2_camera_demo_logs"

    raise RuntimeError("找不到可写的 ROS 2 日志目录。")


def decode_image_message(msg) -> np.ndarray:
    """把 ROS 2 Image 消息转换成可显示的 RGB/灰度图像。"""

    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("图像宽高无效。")

    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return data.reshape(msg.height, msg.width, 3)
    if encoding == "rgba8":
        image = data.reshape(msg.height, msg.width, 4)
        return image[:, :, :3]
    if encoding == "bgr8":
        image = data.reshape(msg.height, msg.width, 3)
        return image[:, :, ::-1]
    if encoding == "bgra8":
        image = data.reshape(msg.height, msg.width, 4)
        return image[:, :, [2, 1, 0]]
    if encoding == "mono8":
        return data.reshape(msg.height, msg.width)

    raise ValueError(f"暂不支持的图像编码：{msg.encoding}")


def main() -> None:
    """程序主入口。"""

    # 给 ROS 2 日志指定一个稳定且通常可写的临时目录，避免某些环境下
    # `~/.ros/log` 不可写时，`rclpy.init()` 直接失败。
    ros_log_dir = get_writable_log_dir()
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image as RosImage
    except ImportError:
        print("当前环境缺少订阅或显示图像所需的 Python 依赖。")
        print("需要的模块：rclpy、sensor_msgs、tkinter、PIL")
        return

    class CameraSubscriber(Node):
        """订阅前视相机图像的话题节点。"""

        def __init__(self, root: tk.Tk) -> None:
            super().__init__("franka_front_camera_subscriber")

            # 记录累计帧数，用于直观看到订阅是否持续进行。
            self.frame_count = 0

            # 使用 monotonic 时钟计算帧间隔，避免系统时间跳变影响 FPS。
            self.last_stamp = time.monotonic()
            self.root = root
            self.window_name = "Franka Front Camera"
            self.image_label = tk.Label(root, text="Waiting for image...")
            self.image_label.pack(fill="both", expand=True)
            self.photo_image = None

            # 创建话题订阅。
            # 参数含义分别是：
            # 1. 消息类型：sensor_msgs/msg/Image
            # 2. 话题名：/front_camera/rgb
            # 3. 回调函数：收到消息后执行 handle_image
            # 4. 队列深度：10，表示缓存最近 10 条未处理消息
            self.subscription = self.create_subscription(
                RosImage,
                "/front_camera/rgb",
                self.handle_image,
                10,
            )

        def handle_image(self, msg: RosImage) -> None:
            """处理每一帧图像消息。"""

            self.frame_count += 1

            # 计算两帧之间的时间间隔，得到近似帧率。
            now = time.monotonic()
            dt = max(now - self.last_stamp, 1e-6)
            fps = 1.0 / dt
            self.last_stamp = now

            try:
                frame = decode_image_message(msg)
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return

            if frame.ndim == 2:
                pil_image = PILImage.fromarray(frame, mode="L")
            else:
                pil_image = PILImage.fromarray(frame, mode="RGB")

            self.photo_image = ImageTk.PhotoImage(image=pil_image)
            self.image_label.configure(image=self.photo_image, text="")
            self.root.title(f"{self.window_name} | {msg.width}x{msg.height} | {msg.encoding} | {fps:.2f} FPS")

            # sensor_msgs/msg/Image 常用字段说明：
            # - width / height：图像宽高
            # - encoding：像素编码格式，例如 rgb8、rgba8、bgr8
            # - header.frame_id：消息所属坐标系名字
            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"frame={self.frame_count} "
                    f"size={msg.width}x{msg.height} "
                    f"encoding={msg.encoding} "
                    f"frame_id={msg.header.frame_id} "
                    f"approx_fps={fps:.2f}"
                )

    print("Demo 16: 订阅 Franka 前视相机 ROS 2 图像流")
    print("Expected Isaac Sim topic: /front_camera/rgb")

    # 初始化 ROS 2 客户端库。
    rclpy.init()

    root = tk.Tk()
    root.title("Franka Front Camera")
    root.geometry("1280x800")

    # 创建节点并进入事件循环。
    node = CameraSubscriber(root)
    keep_running = True

    def handle_close() -> None:
        nonlocal keep_running
        keep_running = False

    root.protocol("WM_DELETE_WINDOW", handle_close)
    try:
        while rclpy.ok() and keep_running:
            rclpy.spin_once(node, timeout_sec=0.05)
            root.update_idletasks()
            root.update()
    except KeyboardInterrupt:
        # 允许 Ctrl+C 正常退出。
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()


if __name__ == "__main__":
    main()
