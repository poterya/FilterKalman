#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile
import time


class EfficiencyMonitor(Node):
    def __init__(self):
        super().__init__('efficiency_monitor')
        
        qos = QoSProfile(depth=10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/propeller_efficiency',
            self.efficiency_callback,
            qos
        )
        
        self.last_print_time = time.time()
        self.print_interval = 0.1  # Печатать каждые 0.1 секунды (10 Hz)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Efficiency Monitor запущен')
        self.get_logger().info('Подписка на топик: /propeller_efficiency')
        self.get_logger().info('Ожидание данных...')
        self.get_logger().info('=' * 60)
        
    def efficiency_callback(self, msg: Float32MultiArray):
        now = time.time()
        
        # Ограничиваем частоту вывода
        if now - self.last_print_time < self.print_interval:
            return
        
        self.last_print_time = now
        
        if len(msg.data) >= 4:
            fr = msg.data[0]
            fl = msg.data[1]
            rr = msg.data[2]
            rl = msg.data[3]
            
            # Форматированный вывод
            print(f'\r[EFF] FR={fr:.3f}  FL={fl:.3f}  RR={rr:.3f}  RL={rl:.3f}  |  AVG={sum(msg.data)/len(msg.data):.3f}', end='', flush=True)
        else:
            self.get_logger().warn(f'Неверный формат данных: {len(msg.data)} элементов вместо 4')


def main(args=None):
    rclpy.init(args=args)
    
    node = EfficiencyMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n')  # Новая строка после Ctrl+C
        node.get_logger().info('Остановка монитора эффективности...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

