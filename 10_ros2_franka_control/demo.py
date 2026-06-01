from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.world import World
from isaacsim.core.utils.types import ArticulationAction

from common.core_utils import add_basic_light
from common.franka_utils import FRANKA_PRIM_PATH, load_franka
from common.ros2_utils import decode_joint_command


def main():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError:
        print("This demo requires ROS 2 Python packages in the current environment.")
        print("Expected modules: rclpy, sensor_msgs, std_msgs")
        print("Source your ROS 2 environment before running this script.")
        simulation_app.close()
        return

    class FrankaRos2Bridge(Node):
        def __init__(self):
            super().__init__("franka_ros2_bridge")
            self.latest_command = None
            self.command_sub = self.create_subscription(
                Float64MultiArray,
                "/franka/joint_command",
                self.handle_command,
                10,
            )
            self.state_pub = self.create_publisher(JointState, "/franka/joint_state", 10)

        def handle_command(self, msg):
            try:
                self.latest_command = decode_joint_command(msg.data, expected_len=9)
                self.get_logger().info(f"received joint command: {self.latest_command.tolist()}")
            except ValueError as exc:
                self.get_logger().error(str(exc))

        def publish_state(self, robot, stamp_msg):
            joint_state = JointState()
            joint_state.header.stamp = stamp_msg
            joint_state.header.frame_id = FRANKA_PRIM_PATH
            joint_state.name = [f"joint_{index}" for index in range(9)]
            positions = robot.get_joint_positions()
            velocities = robot.get_joint_velocities()
            joint_state.position = [float(x) for x in positions]
            joint_state.velocity = [float(x) for x in velocities]
            self.state_pub.publish(joint_state)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_light(world.stage)
    robot = load_franka()

    world.reset()
    robot.initialize()

    rclpy.init()
    bridge = FrankaRos2Bridge()

    print("Demo 10: ROS 2 Franka control loop")
    print("Subscribe: /franka/joint_command (std_msgs/msg/Float64MultiArray)")
    print("Publish:   /franka/joint_state (sensor_msgs/msg/JointState)")

    try:
        for _ in range(100000):
            rclpy.spin_once(bridge, timeout_sec=0.0)
            if bridge.latest_command is not None:
                robot.apply_action(ArticulationAction(joint_positions=bridge.latest_command))
            world.step(render=True)
            stamp = bridge.get_clock().now().to_msg()
            bridge.publish_state(robot, stamp)
            if not simulation_app.is_running():
                break
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
