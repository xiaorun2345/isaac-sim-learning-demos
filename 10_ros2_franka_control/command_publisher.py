def main():
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
    except ImportError:
        print("This script requires ROS 2 Python packages in the current environment.")
        print("Expected modules: rclpy, std_msgs")
        return

    class JointCommandPublisher(Node):
        def __init__(self):
            super().__init__("franka_joint_command_publisher")
            self.publisher = self.create_publisher(Float64MultiArray, "/franka/joint_command", 10)
            self.timer = self.create_timer(2.0, self.publish_next)
            self.index = 0
            self.commands = [
                [0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04],
                [0.2, -0.5, 0.0, -1.7, 0.0, 2.0, 0.8, 0.04, 0.04],
                [0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.00, 0.00],
            ]

        def publish_next(self):
            msg = Float64MultiArray()
            msg.data = self.commands[self.index]
            self.publisher.publish(msg)
            self.get_logger().info(f"published command #{self.index}: {msg.data}")
            self.index = (self.index + 1) % len(self.commands)

    rclpy.init()
    node = JointCommandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
