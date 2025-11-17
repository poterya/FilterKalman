import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String, Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix
import numpy as np
import math
import time
import os
import sys
from datetime import datetime


class KalmanFilter:
    def __init__(self, n_states=4, meas_dim=3):
        from filterpy.kalman import KalmanFilter as FP_KalmanFilter

        self.n = n_states
        self.meas_dim = meas_dim

        self.kf = FP_KalmanFilter(dim_x=n_states, dim_z=meas_dim)

        # Инициализация состояния и матриц, совместимых с прежней логикой
        self.kf.x = np.ones((n_states, 1))
        self.kf.F = np.eye(n_states)
        self.kf.P = np.eye(n_states) * 0.1
        self.kf.Q = np.eye(n_states) * 0.005
        self.kf.R = np.eye(meas_dim) * 0.02

        self.history = []
        self.history_size = 10

    def predict(self):
        self.kf.predict()

    def update(self, measurements, measurement_matrix):
        H = np.array(measurement_matrix, dtype=float)
        z = np.array(measurements, dtype=float).reshape(self.meas_dim, 1)

        self.kf.H = H
        self.kf.update(z)

        # Ограничиваем состояния в [0, 1], как и раньше
        self.kf.x = np.clip(self.kf.x, 0.0, 1.0)

    def get_states(self):
        return self.kf.x.flatten().copy()

    def get_covariance(self):
        return np.diag(self.kf.P).copy()


class KalmanDamageDetector(Node):
    def __init__(self):
        super().__init__('kalman_damage_detector')
        
        # Возвращаемся к одному фильтру, но улучшаем детекцию по паттерну
        self.kalman = KalmanFilter(n_states=4, meas_dim=3)
        
        # Публикаторы
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.propeller_health_pub = self.create_publisher(Bool, '/propeller_health', 10)
        self.vibration_pub = self.create_publisher(Float32, '/vibration_level', 10)
        self.status_pub = self.create_publisher(String, '/drone_status', 10)
        self.detection_pub = self.create_publisher(String, '/damage_detection', 10)
        self.efficiency_pub = self.create_publisher(Float32MultiArray, '/propeller_efficiency', 10)
        
        # Подписки на датчики
        # Используем только гироскоп из IMU
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('enable_detection', False)
        self.declare_parameter('damage_threshold', 0.65)
        self.declare_parameter('detection_min_time_sec', 9.8)
        self.enable_detection = bool(self.get_parameter('enable_detection').value)
        self.damage_threshold = float(self.get_parameter('damage_threshold').value)
        self.detection_min_time_sec = float(self.get_parameter('detection_min_time_sec').value)
        imu_topic = str(self.get_parameter('imu_topic').value)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odometry', self.odom_callback, 10)
        # Резервные подписки под именованным роботом Webots
        self.imu_sub2 = self.create_subscription(Imu, '/Mavic_2_PRO/imu', self.imu_callback, 10)
        self.odom_sub2 = self.create_subscription(Odometry, '/Mavic_2_PRO/odometry', self.odom_callback, 10)
        # GPS как резерв высоты
        self.gps_sub = self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.gps_point_sub = self.create_subscription(PointStamped, '/Mavic_2_PRO/gps', self.gps_point_callback, 10)
        self.sim_time_sub = self.create_subscription(Float32, '/Mavic_2_PRO/sim_time', self.sim_time_callback, 10)
        
        self.current_imu = None
        self.current_gyro_vec = None
        self.current_odom = None
        self.current_altitude_gps = None
        self.baseline_gyro = None
        self.gyro_lpf = None
        self.gyro_alpha = 0.2
        self.imu_msg_count = 0
        self.odom_msg_count = 0
        self.gps_msg_count = 0
        self.sim_time_start = None
        self.sim_time_rel = 0.0
        self.sim_time_ready = False
        self.gyro_calib_samples = []
        self.gyro_calibration_complete = False
        self.gyro_cal_required_samples = 20
        self.k_roll = 0.3
        self.k_pitch = 0.3
        self.k_yaw = 0.2
        self.current_gyro_scores = np.zeros(4, dtype=float)
        self.damage_source = [None, None, None, None]
        self.is_airborne = False
        self.takeoff_time = None
        self.airborne_time = None
        self.flight_time = 0.0
        self.detection_enable_delay_sec = 3.0
        self.propeller_efficiency = np.ones(4)
        self.damage_detected = [False, False, False, False]
        self.damage_clear_threshold = 0.85
        self.min_damage_interval_sec = 3.0
        self.last_damage_time = 0.0
        self.confirm_hold_sec = 0.3
        self.anomaly_confirm_threshold = 0.35
        self.anomaly_above_since = [None, None, None, None]
        self.last_altitude = 0.0
        self.altitude_drop_threshold = 0.5
        self.pattern_history = []  # список (prop_id, score, timestamp)
        self.pattern_window_sec = 2.0  # окно накопления
        self.create_timer(0.1, self.control_loop)
        self.create_timer(0.05, self.kalman_update_loop)
        self.create_timer(1.0, self.publish_status)
        self.create_timer(0.25, self.log_status_brief)
        self.start_time = time.time()
        self.create_timer(1.0, self.sensor_watchdog)
        self.create_timer(1.0, self.report_tick)
        
        # Лог-файл симуляции (сенсоры + краткий статус)
        self.log_fp = None
        self.last_residual = None
        try:
            log_dir = os.path.expanduser('~/ros2_ws/log/sim')
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file_path = os.path.join(log_dir, f'sim_{ts}.log')
            self.log_fp = open(self.log_file_path, 'w', buffering=1, encoding='utf-8')
            self.log_fp.write('# time(s); phase; damaged_count; flight_time(s); imu_ax; imu_ay; imu_az; imu_gx; imu_gy; imu_gz; odom_z\n')
        except Exception as e:
            self.get_logger().warning(f'Не удалось открыть лог-файл: {e}')
        # Файл пересекундного отчёта
        try:
            rep_dir = os.path.expanduser('~/ros2_ws/log/report')
            os.makedirs(rep_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.report_file_path = os.path.join(rep_dir, f'report_{ts}.csv')
            self.report_fp = open(self.report_file_path, 'w', buffering=1, encoding='utf-8')
            self.report_fp.write('t_abs_s,t_rel_s,flight_s,p,q,r,p_lpf,q_lpf,r_lpf,'
                                 'eff_FL,eff_FR,eff_BL,eff_BR,'
                                 'score_FL,score_FR,score_BL,score_BR,'
                                 'dam_FL,dam_FR,dam_BL,dam_BR\n')
        except Exception as e:
            self.report_fp = None
            self.get_logger().warning(f'Не удалось открыть файл отчёта: {e}')
        
        self.get_logger().info('[DETECTOR] Калман-детектор повреждений запущен')
        
    def imu_callback(self, msg):
        self.current_imu = msg
        self.imu_msg_count += 1
        try:
            sim_now = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if self.sim_time_start is None and sim_now > 0.0:
                self.sim_time_start = sim_now
            if self.sim_time_start is not None:
                self.sim_time_rel = max(0.0, sim_now - self.sim_time_start)
        except Exception:
            pass
        if not self.is_airborne and self.imu_msg_count >= 20:
            self.is_airborne = True
            self.last_altitude = 0.0
            self.airborne_time = time.time()
        self.current_gyro_vec = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=float
        )
        # Калибровка гироскопа
        if self.is_airborne and not self.gyro_calibration_complete and self.current_gyro_vec is not None:
            if self.gyro_lpf is None:
                self.gyro_lpf = self.current_gyro_vec.copy()
            else:
                self.gyro_lpf = self.gyro_alpha * self.current_gyro_vec + (1.0 - self.gyro_alpha) * self.gyro_lpf
            if len(self.gyro_calib_samples) < self.gyro_cal_required_samples:
                self.gyro_calib_samples.append(self.gyro_lpf.copy())
            else:
                self.baseline_gyro = np.mean(np.vstack(self.gyro_calib_samples), axis=0)
                self.gyro_calibration_complete = True
    
    def sim_time_callback(self, msg: Float32):
        try:
            self.sim_time_rel = float(msg.data)
            self.sim_time_ready = True
        except Exception:
            pass
    
    def damage_armed_callback(self, msg: Bool):
        pass

    def _process_altitude_update(self, current_altitude: float):
        if not self.is_airborne and current_altitude is not None and current_altitude > 0.3:
            self.is_airborne = True
            self.last_altitude = current_altitude
            self.airborne_time = time.time()
            return
        
        if self.is_airborne and current_altitude is not None:
            self.last_altitude = current_altitude

    def odom_callback(self, msg):
        self.current_odom = msg
        self.odom_msg_count += 1
        current_altitude = msg.pose.pose.position.z
        self._process_altitude_update(current_altitude)

    def gps_callback(self, msg: NavSatFix):
        altitude = getattr(msg, 'altitude', None)
        if altitude is None:
            return
        self.current_altitude_gps = float(altitude)
        self.gps_msg_count += 1
        self._process_altitude_update(self.current_altitude_gps)
    
    def gps_point_callback(self, msg: PointStamped):
        try:
            self.current_altitude_gps = float(msg.point.z)
            self.gps_msg_count += 1
            self._process_altitude_update(self.current_altitude_gps)
        except Exception:
            pass
            
    def _build_gyro_H(self) -> np.ndarray:
        k_r = self.k_roll
        k_p = self.k_pitch
        k_y = self.k_yaw
        H = np.array([
            [+k_r, -k_r, +k_r, -k_r],
            [+k_p, +k_p, -k_p, -k_p],
            [+k_y, -k_y, -k_y, +k_y],
        ], dtype=float)
        return H
    
    def _get_gyro_measurement(self):
        if self.current_gyro_vec is None or not self.gyro_calibration_complete or self.baseline_gyro is None:
            return None
        return (self.current_gyro_vec - self.baseline_gyro).astype(float)
    
    def detect_damaged_propeller_by_pattern(self, gyro_error: np.ndarray):
        """
        
        Порядок винтов в mavic_driver: [FR, FL, RR, RL] (индексы 0,1,2,3)
        Каждый винт создаёт уникальный паттерн возмущений:
        - FR (0): roll↑, pitch↓, yaw↑   [+1, -1, +1]
        - FL (1): roll↓, pitch↓, yaw↓   [-1, -1, -1]  
        - RR (2): roll↑, pitch↑, yaw↓   [+1, +1, -1]
        - RL (3): roll↓, pitch↑, yaw↑   [-1, +1, +1]
        """
        patterns = {
            0: np.array([+1.0, -1.0, +1.0]),  # FR (front-right
            1: np.array([-1.0, -1.0, -1.0]),  # FL (front-left)  
            2: np.array([+1.0, +1.0, -1.0]),  # RR (rear-right)  
            3: np.array([-1.0, +1.0, +1.0]),  # RL (rear-left)  
        }
        
        # Нормализуем вектор ошибки гироскопа
        error_norm = np.linalg.norm(gyro_error)
        if error_norm < 0.05:  # слишком малые возмущения
            return -1, 0.0
        
        norm_error = gyro_error / error_norm
        
        # Находим паттерн с максимальным скалярным произведением
        best_match = -1
        best_score = -1.0
        scores = {}
        
        for prop_id, pattern in patterns.items():
            # Скалярное произведение показывает схожесть направлений
            score = float(norm_error.dot(pattern))
            scores[prop_id] = score
            if score > best_score:
                best_score = score
                best_match = prop_id
        
        # Накопляем паттерн в историю для усреднения
        now = time.time()
        if best_score > 0.5 and error_norm > 0.2:  # значимое возмущение
            self.pattern_history.append((best_match, best_score, now))
        
        # Удаляем старые записи за пределами окна
        self.pattern_history = [(p, s, t) for (p, s, t) in self.pattern_history if (now - t) < self.pattern_window_sec]
        
        # Усредняем по окну: какой винт чаще всего определяется
        if len(self.pattern_history) >= 10:  # минимум 10 наблюдений
            prop_votes = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            for (prop_id, score, _) in self.pattern_history:
                prop_votes[prop_id] += score
            
            # Находим винт с максимальным накопленным score
            accumulated_best = max(prop_votes, key=prop_votes.get)
            accumulated_score = prop_votes[accumulated_best] / len(self.pattern_history)
            
            # Логируем для отладки
            if error_norm > 0.2:
                score_str = " ".join([f"P{i}:{s:.2f}" for i, s in scores.items()])
                votes_str = " ".join([f"P{i}:{v:.1f}" for i, v in prop_votes.items()])
                self.get_logger().info(
                    f'[PATTERN] [{norm_error[0]:.2f},{norm_error[1]:.2f},{norm_error[2]:.2f}] '
                    f'inst={score_str} hist={votes_str} -> P{accumulated_best}(acc={accumulated_score:.2f})'
                )
            
            return accumulated_best, accumulated_score
        
        return best_match, best_score
        
    def kalman_update_loop(self):
        if not self.is_airborne:
            return
        
        self.kalman.predict()
        z_gyro = self._get_gyro_measurement()
        
        if z_gyro is not None:
            # ГЛАВНОЕ ИЗМЕНЕНИЕ: детекция по паттерну ПРИОРИТЕТНЕЕ Калмана
            damaged_prop, pattern_score = self.detect_damaged_propeller_by_pattern(z_gyro)
            
            H_gyro = self._build_gyro_H()
            
            try:
                # Обновляем Калман
                self.kalman.update(z_gyro, H_gyro)
                self.propeller_efficiency = self.kalman.get_states()
                
                # Но для scores используем ТОЛЬКО паттерн!
                scores = np.zeros(4, dtype=float)
                
                if damaged_prop >= 0 and pattern_score > 0.5:
                    # Паттерн указывает на конкретный винт
                    scores[damaged_prop] = pattern_score
                    # Остальные получают низкий score
                    for i in range(4):
                        if i != damaged_prop:
                            scores[i] = 0.0
                else:
                    # Если паттерн неуверенный, используем residual
                    residual = z_gyro - H_gyro.dot(self.propeller_efficiency)
                    for i in range(4):
                        col = H_gyro[:, i]
                        denom = np.linalg.norm(col) if np.linalg.norm(col) > 1e-6 else 1.0
                        scores[i] = abs(col.dot(residual)) / denom
                    if np.max(scores) > 0:
                        scores = scores / (np.max(scores) + 1e-6)
                
                self.current_gyro_scores = np.clip(scores, 0.0, 1.0)
                
            except Exception as e:
                self.get_logger().warning(f'Не удалось обновить по гироскопу: {e}')
                self.propeller_efficiency = self.kalman.get_states()
        else:
            self.propeller_efficiency = self.kalman.get_states()
        
        if self.enable_detection:
            self._update_damage_flags(gyro_scores=self.current_gyro_scores)
    
    def _update_damage_flags(self, gyro_scores: np.ndarray):
        for i in range(4):
            if self.damage_detected[i] and self.propeller_efficiency[i] > self.damage_clear_threshold:
                self.damage_detected[i] = False
        if self.airborne_time is None or (time.time() - self.airborne_time) < self.detection_enable_delay_sec:
            return
        if any(self.damage_detected):
            return
        now = time.time()
        if now - self.last_damage_time < self.min_damage_interval_sec:
            return
        if self.sim_time_rel < self.detection_min_time_sec:
            return
        gyro = gyro_scores if gyro_scores is not None else np.zeros(4)
        
        # НОВАЯ ЛОГИКА: Детектируем винт с МАКСИМАЛЬНЫМ gyro score (из паттерна)
        idx = int(np.argmax(gyro))
        max_gyro_score = gyro[idx]
        
        # Проверяем условия для винта с максимальным score
        if max_gyro_score > 0.4:  # Порог уверенности паттерна
            # Также проверяем эффективность
            eff_low = self.propeller_efficiency[idx] < self.damage_threshold
            
            if eff_low or max_gyro_score > 0.7:  # Либо низкая эффективность, либо очень уверенный паттерн
                self.damage_detected[idx] = True
                self.last_damage_time = now
                source = 'PATTERN' if max_gyro_score > 0.7 else 'KALMAN'
                self.damage_source[idx] = source
                # Map internal order [FR, FL, RR, RL] to logical order [FR(1), FL(2), RR(3), RL(4)]
                det_to_driver = [1, 2, 3, 4]
                reported_id = det_to_driver[idx]
                t_out = self.sim_time_rel
                self.get_logger().error(
                    f'[DAMAGE DETECTED] ВИНТ_{reported_id}_ПОВРЕЖДЕН [{source}] '
                    f't={t_out:.2f}s eff={self.propeller_efficiency[idx]:.2f} '
                    f'pattern_score={max_gyro_score:.2f}'
                )
                try:
                    if self.log_fp:
                        self.log_fp.write(
                            f'{t_out:.3f}; DAMAGE_DETECTED_P{reported_id}; '
                            f'SRC={source}; EFF={self.propeller_efficiency[idx]:.3f}; '
                            f'PATTERN={max_gyro_score:.3f}\n'
                        )
                        self.log_fp.flush()
                except Exception:
                    pass
                detection_msg = String()
                detection_msg.data = (
                    f"ВИНТ_{reported_id}_ПОВРЕЖДЕН[{source}] "
                    f"t={t_out:.2f}s eff={self.propeller_efficiency[idx]:.2f} "
                    f"pattern={max_gyro_score:.2f}"
                )
                self.detection_pub.publish(detection_msg)
                try:
                    print(
                        f'[!] DETECT: P{reported_id} [{source}] '
                        f't={t_out:.2f}s eff={self.propeller_efficiency[idx]:.2f} '
                        f'pattern={max_gyro_score:.2f}', 
                        flush=True
                    )
                except Exception:
                    pass
                
    def calculate_vibration_level(self):
        base_vibration = 0.05
        damage_vibration = 0.0
        for eff in self.propeller_efficiency:
            if eff < 1.0:
                damage_vibration += (1.0 - eff) * 0.25
        return min(base_vibration + damage_vibration, 1.0)
    def calculate_compensation(self):
        thrust_comp = 0.0
        yaw_comp = 0.0
        roll_comp = 0.0
        pitch_comp = 0.0
        for i, eff in enumerate(self.propeller_efficiency):
            if eff < 0.95:
                loss = 1.0 - eff
                thrust_comp += loss * 0.2
                if i == 0:
                    roll_comp -= loss * 0.1
                    pitch_comp -= loss * 0.1
                    yaw_comp += loss * 0.15
                elif i == 1:
                    roll_comp += loss * 0.1
                    pitch_comp -= loss * 0.1
                    yaw_comp -= loss * 0.15
                elif i == 2:
                    roll_comp -= loss * 0.1
                    pitch_comp += loss * 0.1
                    yaw_comp -= loss * 0.1
                elif i == 3:
                    roll_comp += loss * 0.1
                    pitch_comp += loss * 0.1
                    yaw_comp += loss * 0.1
        return thrust_comp, roll_comp, pitch_comp, yaw_comp
    def control_loop(self):
        cmd_msg = Twist()
        self.flight_time += 0.1
        thrust_comp, roll_comp, pitch_comp, yaw_comp = self.calculate_compensation()

        # Команда взлёта/поддержания высоты:
        # активный взлёт до момента, когда дрон в воздухе, дальше — высоту держит контроллер дрона
        if not self.is_airborne:
            cmd_msg.linear.z = 1.0  # взлёт
        else:
            cmd_msg.linear.z = 0.0  # без аварийной посадки, высота стабилизируется самим контроллером

        cmd_msg.angular.x = roll_comp
        cmd_msg.angular.y = pitch_comp
        cmd_msg.angular.z = yaw_comp

        # Базовая траектория: прямолинейный полёт вперёд
        if self.is_airborne:
            v = 0.8  # м/с вперёд
            cmd_msg.linear.x = v
            cmd_msg.linear.y = 0.0

        # В режиме активной компенсации НЕ выполняем аварийную посадку,
        # а только меняем roll/pitch/yaw через calculate_compensation()

        self.cmd_vel_pub.publish(cmd_msg)
    def publish_status(self):
        health_msg = Bool()
        health_msg.data = not any(self.damage_detected)
        self.propeller_health_pub.publish(health_msg)
        vibration_msg = Float32()
        vibration_msg.data = self.calculate_vibration_level()
        self.vibration_pub.publish(vibration_msg)
        
        # Публикация эффективности винтов (для проверки алгоритма)
        eff_msg = Float32MultiArray()
        eff_msg.data = [float(e) for e in self.propeller_efficiency]
        self.efficiency_pub.publish(eff_msg)
        
        status_msg = String()
        damaged_count = sum(self.damage_detected)
        avg_efficiency = np.mean(self.propeller_efficiency)
        if not self.is_airborne:
            status_msg.data = "ВЗЛЕТ_В_ПРОЦЕССЕ"
        elif not self.gyro_calibration_complete:
            status_msg.data = f"КАЛИБРОВКА_ГИРОСКОПА... {len(self.gyro_calib_samples)}/{self.gyro_cal_required_samples}"
        else:
            status_msg.data = (
                f"ПОЛЕТ:{self.flight_time:.1f}с "
                f"ПОВРЕЖДЕНО:{damaged_count}/4 "
                f"СР_ЭФФ:{avg_efficiency:.2%} "
                f"ВИБРАЦИЯ:{vibration_msg.data:.2f}"
            )
            eff_str = " | ".join([f"В{i+1}:{eff:.2%}" for i, eff in enumerate(self.propeller_efficiency)])
            status_msg.data += f" | {eff_str}"
            
            # Детальное логирование для проверки алгоритма (порядок: FR, FL, RR, RL)
            if self.enable_detection and self.flight_time > 0:
                self.get_logger().info(
                    f'🔍 Эффективность: FR={self.propeller_efficiency[0]:.3f} '
                    f'FL={self.propeller_efficiency[1]:.3f} '
                    f'RR={self.propeller_efficiency[2]:.3f} '
                    f'RL={self.propeller_efficiency[3]:.3f}'
                )
        
        self.status_pub.publish(status_msg)

    def log_status_brief(self):
        try:
            if self.log_fp:
                damaged_count = sum(self.damage_detected)
                t_rel = time.time() - self.start_time
                self.log_fp.write(f'{t_rel:.3f}; TICK; {damaged_count}; {self.flight_time:.2f}\n')
        except Exception:
            pass
    
    def sensor_watchdog(self):
        if not self.enable_detection:
            return
        now = time.time()
        elapsed = now - self.start_time
        if elapsed < 8.0:
            return

    def report_tick(self):
        if self.enable_detection:
            try:
                if not getattr(self, 'report_fp', None):
                    return
                t_abs = time.time()
                t_rel = t_abs - self.start_time
                p = q = r = float('nan')
                p_l = q_l = r_l = float('nan')
                if self.current_gyro_vec is not None:
                    p = float(self.current_gyro_vec[0])
                    q = float(self.current_gyro_vec[1])
                    r = float(self.current_gyro_vec[2])
                if self.gyro_lpf is not None:
                    p_l = float(self.gyro_lpf[0])
                    q_l = float(self.gyro_lpf[1])
                    r_l = float(self.gyro_lpf[2])
                eff = self.propeller_efficiency if self.propeller_efficiency is not None else np.ones(4)
                sc = self.current_gyro_scores if self.current_gyro_scores is not None else np.zeros(4)
                dam = self.damage_detected if self.damage_detected is not None else [False, False, False, False]
                self.report_fp.write(
                    f'{t_abs:.3f},{t_rel:.3f},{self.flight_time:.2f},'
                    f'{p:.6f},{q:.6f},{r:.6f},{p_l:.6f},{q_l:.6f},{r_l:.6f},'
                    f'{eff[0]:.4f},{eff[1]:.4f},{eff[2]:.4f},{eff[3]:.4f},'
                    f'{sc[0]:.4f},{sc[1]:.4f},{sc[2]:.4f},{sc[3]:.4f},'
                    f'{int(dam[0])},{int(dam[1])},{int(dam[2])},{int(dam[3])}\n'
                )
            except Exception:
                pass
            # Проверяем данные только после достаточного времени ожидания (15 секунд)
            elapsed = time.time() - self.start_time
            if elapsed < 15.0:
                return  # Слишком рано для проверки, ждём данных
            
            gyro_ok = (self.current_gyro_vec is not None) or (self.imu_msg_count > 0)
            alt_ok = (self.current_odom is not None) or (self.current_altitude_gps is not None) or (self.odom_msg_count > 0) or (self.gps_msg_count > 0)
            if not gyro_ok:
                self.get_logger().warn(f'[WARNING] Нет данных гироскопа/IMU после {elapsed:.1f}с. Продолжаем ожидание...')
                return  # Не завершаем узел, продолжаем ожидать
            if not alt_ok:
                self.get_logger().warn(f'[WARNING] Нет данных высоты после {elapsed:.1f}с. Продолжаем ожидание...')
                return  # Не завершаем узел, продолжаем ожидать


def main():
    rclpy.init()
    node = KalmanDamageDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Работа детектора завершена')
    finally:
        try:
            if getattr(node, 'log_fp', None):
                node.log_fp.flush()
                node.log_fp.close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        try:
            if getattr(node, 'report_fp', None):
                node.report_fp.flush()
                node.report_fp.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()


