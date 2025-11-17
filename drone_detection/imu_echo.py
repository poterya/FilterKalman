import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import time


class ImuEcho(Node):
    def __init__(self):
        super().__init__('imu_echo')
        self.sub = self.create_subscription(Imu, '/imu', self.cb, 50)
        self.last_print = 0.0
        self.print_hz = 10.0

    def cb(self, msg: Imu):
        now = time.time()
        if now - self.last_print < 1.0 / self.print_hz:
            return
        self.last_print = now
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        self.get_logger().info(f'GYRO: ({gx:.3f}, {gy:.3f}, {gz:.3f})')


def main():
    rclpy.init()
    node = ImuEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


