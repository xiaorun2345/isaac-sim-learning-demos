from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.ros2_utils import summarize_image_message


def main():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError:
        print("This demo requires ROS 2 Python packages in the current environment.")
        print("Expected modules: rclpy, sensor_msgs")
        print("Source your ROS 2 environment before running this script.")
        return

    class CameraSubscriber(Node):
        def __init__(self):
            super().__init__("isaac_camera_subscriber")
            self.frame_count = 0
            self.last_stamp = time.time()
            self.subscription = self.create_subscription(
                Image,
                "/isaac/camera/rgb",
                self.handle_image,
                10,
            )

        def handle_image(self, msg):
            self.frame_count += 1
            info = summarize_image_message(msg)
            now = time.time()
            dt = max(now - self.last_stamp, 1e-6)
            fps = 1.0 / dt
            self.last_stamp = now
            self.get_logger().info(
                f"frame={self.frame_count} "
                f"size={info['width']}x{info['height']} "
                f"encoding={info['encoding']} "
                f"frame_id={info['frame_id']} "
                f"approx_fps={fps:.2f}"
            )

    print("Demo 09: subscribe to camera video over ROS 2")
    print("Expected Isaac Sim topic: /isaac/camera/rgb")
    print("Pair this with Demo 04 camera creation and an Isaac ROS 2 bridge publisher.")

    rclpy.init()
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
