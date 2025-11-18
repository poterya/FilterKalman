#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node


def generate_launch_description():
   

    # 1) Стандартный демо-лаунч Mavic 2 PRO из webots_ros2_mavic
    mavic_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('webots_ros2_mavic'),
            '/launch/robot_launch.py',
        ])
    )

    # 2) Калман-детектор повреждений
    kalman_detector = Node(
        package='drone_detection',
        executable='new',
        name='new',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'enable_detection': True,
            'damage_threshold': 0.85,  # Понижен с 0.70 до 0.85
            'detection_min_time_sec': 2.0,
        }],
    )

    # 3) Узел для echo IMU (для отладки)
    imu_echo = Node(
        package='drone_detection',
        executable='imu_echo',
        name='imu_echo',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        mavic_launch,
        kalman_detector,
        imu_echo,
    ])

