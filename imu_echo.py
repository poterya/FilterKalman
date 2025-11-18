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
        self.last_value = None
        self.change_epsilon = 0.01  # Минимальное изменение для логирования
        self.keep_alive_sec = 1.0   # Печатаем хотя бы раз в секунду
        self.last_keep_alive = 0.0

    def cb(self, msg: Imu):
        now = time.time()
        if now - self.last_print < 1.0 / self.print_hz:
            return
        self.last_print = now
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        current = (gx, gy, gz)

        # Проверяем, изменилось ли значение
        changed = False
        if self.last_value is None:
            changed = True
        else:
            diff = max(abs(current[i] - self.last_value[i]) for i in range(3))
            if diff >= self.change_epsilon:
                changed = True

        if changed or (now - self.last_keep_alive) >= self.keep_alive_sec:
            self.get_logger().info(f'GYRO: ({gx:.3f}, {gy:.3f}, {gz:.3f})')
            self.last_value = current
            self.last_keep_alive = now


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


