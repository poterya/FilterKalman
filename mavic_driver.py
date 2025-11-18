# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS2 Mavic 2 Pro driver."""

import math
import rclpy
from geometry_msgs.msg import Twist


K_VERTICAL_THRUST = 68.5    # with this thrust, the drone lifts.
K_VERTICAL_P = 3.0          # P constant of the vertical PID.
K_ROLL_P = 50.0             # P constant of the roll PID.
K_PITCH_P = 30.0            # P constant of the pitch PID.
K_YAW_P = 2.0
K_X_VELOCITY_P = 1
K_Y_VELOCITY_P = 1
K_X_VELOCITY_I = 0.01
K_Y_VELOCITY_I = 0.01
LIFT_HEIGHT = 1


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class MavicDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        # Sensors
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__imu = self.__robot.getDevice('inertial unit')

        # Propellers
        self.__propellers = [
            self.__robot.getDevice('front right propeller'),
            self.__robot.getDevice('front left propeller'),
            self.__robot.getDevice('rear right propeller'),
            self.__robot.getDevice('rear left propeller')
        ]
        for propeller in self.__propellers:
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)

        # State
        self.__target_twist = Twist()
        self.__vertical_ref = LIFT_HEIGHT
        self.__linear_x_integral = 0
        self.__linear_y_integral = 0

        # Коэффициенты эффективности винтов (все 100%)
        self.__damage_factors = [1.0, 1.0, 1.0, 1.0]  # FR, FL, RR, RL
        self.__step_counter = 0
        self.__last_collision_time = 0.0
        self.__collision_cooldown = 2.0  # секунды между коллизиями
        self.__impact_roll_thresh = 1.0
        self.__impact_pitch_thresh = 1.0
        self.__impact_delta_thresh = 3.0
        self.__prev_roll_velocity = 0.0
        self.__prev_pitch_velocity = 0.0

        # ROS interface
        rclpy.init(args=None)
        self.__node = rclpy.create_node('mavic_driver')
        self.__node.create_subscription(Twist, 'cmd_vel', self.__cmd_vel_callback, 1)
        # No simulated damage

    def __cmd_vel_callback(self, twist):
        self.__target_twist = twist

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        roll_ref = 0
        pitch_ref = 0

        # Read sensors
        roll, pitch, _ = self.__imu.getRollPitchYaw()
        x, _, vertical = self.__gps.getValues()
        roll_velocity, pitch_velocity, twist_yaw = self.__gyro.getValues()
        velocity = self.__gps.getSpeed()
        if math.isnan(velocity):
            return

        self.__handle_collisions(roll_velocity, pitch_velocity)

        # Allow high level control once the drone is lifted
        if vertical > 0.2:
            # Calculate estimated horizontal velocity from attitude and GPS speed
            denom = (abs(roll) + abs(pitch))
            velocity_x = (pitch / denom) * velocity if denom > 0 else 0.0
            velocity_y = - (roll / denom) * velocity if denom > 0 else 0.0

            # Desired velocities come from external controller via /cmd_vel
            desired_vx = self.__target_twist.linear.x
            desired_vy = self.__target_twist.linear.y

            # High level controller (linear velocity) tracking the desired velocity
            linear_y_error = desired_vy - velocity_y
            linear_x_error = desired_vx - velocity_x
            self.__linear_x_integral += linear_x_error
            self.__linear_y_integral += linear_y_error
            roll_ref = K_Y_VELOCITY_P * linear_y_error + K_Y_VELOCITY_I * self.__linear_y_integral
            pitch_ref = - K_X_VELOCITY_P * linear_x_error - K_X_VELOCITY_I * self.__linear_x_integral
            # Vertical reference can be adjusted by external controller through linear.z
            self.__vertical_ref = max(0.0, LIFT_HEIGHT + self.__target_twist.linear.z)
        vertical_input = K_VERTICAL_P * (self.__vertical_ref - vertical)

        # Low level controller (roll, pitch, yaw)
        yaw_ref = self.__target_twist.angular.z

        roll_input = K_ROLL_P * clamp(roll, -1, 1) + roll_velocity + roll_ref
        pitch_input = K_PITCH_P * clamp(pitch, -1, 1) + pitch_velocity + pitch_ref
        yaw_input = K_YAW_P * (yaw_ref - twist_yaw)

        m1 = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        m2 = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        m3 = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        m4 = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input

        # Apply control with possible damage factors
        self.__propellers[0].setVelocity(-m1 * self.__damage_factors[0])
        self.__propellers[1].setVelocity(m2 * self.__damage_factors[1])
        self.__propellers[2].setVelocity(m3 * self.__damage_factors[2])
        self.__propellers[3].setVelocity(-m4 * self.__damage_factors[3])

    def __handle_collisions(self, roll_velocity, pitch_velocity):
        now = self.__robot.getTime()
        if (now - self.__last_collision_time) < self.__collision_cooldown:
            return

        roll_side = None  # 'left' or 'right'
        roll_delta = roll_velocity - self.__prev_roll_velocity
        if roll_velocity > self.__impact_roll_thresh or roll_delta > self.__impact_delta_thresh:
            roll_side = 'right'
        elif roll_velocity < -self.__impact_roll_thresh or roll_delta < -self.__impact_delta_thresh:
            roll_side = 'left'

        pitch_side = None  # 'front' or 'rear'
        pitch_delta = pitch_velocity - self.__prev_pitch_velocity
        if pitch_velocity < -self.__impact_pitch_thresh or pitch_delta < -self.__impact_delta_thresh:
            pitch_side = 'front'
        elif pitch_velocity > self.__impact_pitch_thresh or pitch_delta > self.__impact_delta_thresh:
            pitch_side = 'rear'

        self.__prev_roll_velocity = roll_velocity
        self.__prev_pitch_velocity = pitch_velocity

        if roll_side is None and pitch_side is None:
            return

        # Определяем индекс винта: 0 FR, 1 FL, 2 RR, 3 RL
        damaged_index = None
        if pitch_side is not None and roll_side is not None:
            if pitch_side == 'front' and roll_side == 'right':
                damaged_index = 0  # FR
            elif pitch_side == 'front' and roll_side == 'left':
                damaged_index = 1  # FL
            elif pitch_side == 'rear' and roll_side == 'right':
                damaged_index = 2  # RR
            else:
                damaged_index = 3  # RL
        elif pitch_side is not None:
            damaged_index = 0 if pitch_side == 'front' else 2
        elif roll_side is not None:
            damaged_index = 0 if roll_side == 'right' else 1

        if damaged_index is None:
            return

        if self.__damage_factors[damaged_index] <= 0.35:
            return  # уже поврежден

        self.__damage_factors[damaged_index] = max(0.3, self.__damage_factors[damaged_index] * 0.5)
        self.__last_collision_time = now
        print(f'[MAVIC DRIVER] collision detected -> prop {damaged_index} efficiency reduced to {self.__damage_factors[damaged_index]:.2f}')
