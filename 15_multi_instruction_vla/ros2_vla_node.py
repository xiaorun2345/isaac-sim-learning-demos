import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.groot_utils import build_groot_policy_observation, extract_first_action


ACTION_KEY = os.environ.get("GROOT_ACTION_KEY", "joints")


def main():
    try:
        import rclpy
        from cv_bridge import CvBridge
        from gr00t.policy.server_client import PolicyClient
        from rclpy.node import Node
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float64MultiArray, String
    except ImportError:
        print("Required packages: rclpy, cv_bridge, sensor_msgs, std_msgs, Isaac-GR00T client.")
        return

    class Ros2VlaNode(Node):
        def __init__(self):
            super().__init__("ros2_vla_node")
            self.bridge = CvBridge()
            self.rgb = None
            self.instruction = "pick the red cube"
            self.joints = np.zeros(9, dtype=np.float32)
            self.policy = PolicyClient(
                host=os.environ.get("GROOT_HOST", "127.0.0.1"),
                port=int(os.environ.get("GROOT_PORT", "5555")),
                timeout_ms=15000,
                strict=False,
            )
            if not self.policy.ping():
                raise RuntimeError("Cannot reach GR00T policy server.")

            self.create_subscription(Image, "/isaac/camera/rgb", self.on_rgb, 10)
            self.create_subscription(JointState, "/franka/joint_state", self.on_joint_state, 10)
            self.create_subscription(String, "/vla/instruction", self.on_instruction, 10)
            self.command_pub = self.create_publisher(Float64MultiArray, "/franka/joint_command", 10)
            self.create_timer(0.5, self.run_policy_once)

        def on_rgb(self, msg):
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        def on_instruction(self, msg):
            self.instruction = msg.data
            self.get_logger().info(f"instruction={self.instruction}")

        def on_joint_state(self, msg):
            if len(msg.position) == 9:
                self.joints = np.asarray(msg.position, dtype=np.float32)

        def run_policy_once(self):
            if self.rgb is None:
                self.get_logger().info("waiting for /isaac/camera/rgb")
                return
            observation = build_groot_policy_observation(self.rgb, self.instruction, self.joints)
            action_dict, _ = self.policy.get_action(observation)
            self.joints = extract_first_action(action_dict, ACTION_KEY)
            msg = Float64MultiArray()
            msg.data = self.joints.tolist()
            self.command_pub.publish(msg)
            self.get_logger().info(f"published {len(msg.data)} joint targets")

    rclpy.init()
    node = Ros2VlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
