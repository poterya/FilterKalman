import sys
import select
import termios
import tty
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


INSTRUCTIONS = """
Управление дроном с клавиатуры
==============================
W / S : движение вперед / назад
A / D : движение влево / вправо
R / F : вверх / вниз
Q / E : поворот влево / вправо (yaw)
Space : остановить дрон (все скорости = 0)
P / O : увеличить / уменьшить линейную скорость
L / K : увеличить / уменьшить угловую скорость
Ctrl-C: выход
"""


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.settings = termios.tcgetattr(sys.stdin)

        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.max_linear = 2.0
        self.max_angular = math.radians(90)  # ~1.57

        self.current_twist = Twist()
        self.timer = self.create_timer(0.1, self.publish_cmd)
        self.get_logger().info('🚀 Keyboard teleop запущен. Управляйте из этого терминала.')
        print(INSTRUCTIONS)

    def publish_cmd(self):
        self.publisher.publish(self.current_twist)

    def adjust_speed(self, linear_delta=0.0, angular_delta=0.0):
        self.linear_speed = max(0.05, min(self.max_linear, self.linear_speed + linear_delta))
        self.angular_speed = max(0.1, min(self.max_angular, self.angular_speed + angular_delta))
        self.get_logger().info(f'Скорости: linear={self.linear_speed:.2f} m/s, angular={self.angular_speed:.2f} rad/s')

    def handle_key(self, key):
        twist = Twist()

        if key == 'w':
            twist.linear.x = self.linear_speed
        elif key == 's':
            twist.linear.x = -self.linear_speed
        elif key == 'a':
            twist.linear.y = self.linear_speed
        elif key == 'd':
            twist.linear.y = -self.linear_speed
        elif key == 'r':
            twist.linear.z = self.linear_speed
        elif key == 'f':
            twist.linear.z = -self.linear_speed
        elif key == 'q':
            twist.angular.z = self.angular_speed
        elif key == 'e':
            twist.angular.z = -self.angular_speed
        elif key == 'p':
            self.adjust_speed(linear_delta=0.1)
            return
        elif key == 'o':
            self.adjust_speed(linear_delta=-0.1)
            return
        elif key == 'l':
            self.adjust_speed(angular_delta=math.radians(5))
            return
        elif key == 'k':
            self.adjust_speed(angular_delta=-math.radians(5))
            return
        elif key == ' ':
            self.get_logger().info('⏹ Дрон остановлен (все скорости обнулены)')
            self.current_twist = Twist()
            return
        else:
            return

        self.current_twist = twist

    def run(self):
        try:
            while rclpy.ok():
                key = get_key(self.settings)
                if key == '\x03':  # Ctrl+C
                    break
                if key:
                    self.handle_key(key)
                rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            self.current_twist = Twist()
            self.publish_cmd()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.get_logger().info('🛑 Keyboard teleop остановлен')


def main():
    rclpy.init()
    node = KeyboardTeleop()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


