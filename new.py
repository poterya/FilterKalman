import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, Float32, String, Float32MultiArray
from sensor_msgs.msg import Imu

import numpy as np
from filterpy.kalman import KalmanFilter as FP_KalmanFilter
import time


class KalmanFilterWrapper:
    def __init__(self, n_states=4, meas_dim=3, node=None):
        self.node = node
        self.n = n_states
        self.meas_dim = meas_dim
        self.kf = FP_KalmanFilter(dim_x=n_states, dim_z=meas_dim)
        self.kf.x = np.ones((n_states, 1))
        self.kf.F = np.eye(n_states)
        self.kf.P = np.eye(n_states) * 0.1
        # Увеличенные ковариации для большей стабильности (меньше резких скачков)
        self.kf.Q = np.eye(n_states) * 0.05  # Увеличено с 0.01 для большей инерции
        self.kf.R = np.eye(meas_dim) * 0.1   # Увеличено с 0.01 для меньшей чувствительности к шуму

    def predict(self):
        self.kf.predict()

    def update(self, measurements, H):
        H = np.array(H, dtype=float)
        z = np.array(measurements, dtype=float).reshape(self.meas_dim, 1)
        self.kf.H = H
        self.kf.update(z)
        self.kf.x = np.clip(self.kf.x, 0.0, 1.0)

        # residual = z - np.dot(H, self.kf.x)  # Для отладки, если нужно

    def get_states(self):
        return self.kf.x.flatten().copy()


class KalmanDamageController:
    def __init__(self, node):
        self.node = node
        self.kalman = KalmanFilterWrapper(n_states=4, meas_dim=3, node=node)  # Только гироскоп (акселерометр не работает в Webots)
        self.propeller_efficiency = np.ones(4)
        self.propeller_efficiency_smoothed = np.ones(4)  # Сглаженные эффективности
        self.efficiency_smoothing_alpha = 0.3  # Коэффициент сглаживания (0.0-1.0, меньше = больше сглаживания)
        self.max_efficiency_change_per_update = 0.1  # Максимальное изменение за одно обновление
        self.efficiency_history_with_activity = []
        self.max_history_size = 50
        self.history_window = 4
        self.prev_efficiency = np.ones(4)
        self.spike_threshold_up = 0.10
        self.spike_threshold_down = -0.08
        self.spike_diff_threshold = 0.35
        self.spike_cooldown_sec = 1.0
        self.last_spike_time = [0.0, 0.0, 0.0, 0.0]
        self.suspected_prop = None
        self.suspected_expiry = 0.0
        self.suspect_window_sec = 2.0
        self.current_gyro = None
        self.baseline_gyro = None
        self.gyro_calibration_complete = False
        self.gyro_calib_samples = []
        self.gyro_cal_required_samples = 100
        self.gyro_alpha = 0.05
        self.gyro_lpf = None
        self.calibration_started = False
        self.stable_gyro_threshold = 0.1
        self.stable_samples_required = 20
        self.stable_samples_count = 0
        self.damage_detected = [False]*4
        self.last_damage_time = 0.0
        self.damage_threshold = 0.65
        self.pattern_window_sec = 2.0
        self.pattern_history = []
        self.enable_detection = True
        self.k_roll = 0.3
        self.k_pitch = 0.3
        self.k_yaw = 0.2

        self.efficiency_pub = self.node.create_publisher(Float32MultiArray, '/propeller_efficiency', QoSProfile(depth=10))
        self.propeller_health_pub = self.node.create_publisher(Bool, '/propeller_health', QoSProfile(depth=10))
        self.detection_pub = self.node.create_publisher(String, '/damage_detection', QoSProfile(depth=10))
        self.status_pub = self.node.create_publisher(String, '/drone_status', QoSProfile(depth=10))

    def _detect_efficiency_spikes(self, deltas: np.ndarray):
        if deltas is None or deltas.shape[0] != 4:
            return
        pairs = [
            (0, 1),  # FR main, FL opposite
            (1, 0),  # FL main, FR opposite
            (2, 3),  # RR main, RL opposite
            (3, 2),  # RL main, RR opposite
        ]
        now = time.time()
        for main_idx, opp_idx in pairs:
            main_delta = deltas[main_idx]
            opp_delta = deltas[opp_idx]
            dominance = self.propeller_efficiency[main_idx] - self.propeller_efficiency[opp_idx]
            quick_drop = (self.propeller_efficiency[opp_idx] < 0.35 and self.propeller_efficiency[main_idx] > 0.6 and dominance >= self.spike_diff_threshold)
            if (main_delta >= self.spike_threshold_up and opp_delta <= self.spike_threshold_down) or quick_drop:
                if (now - self.last_spike_time[main_idx]) < self.spike_cooldown_sec:
                    continue
                self.last_spike_time[main_idx] = now
                self.suspected_prop = main_idx
                self.suspected_expiry = now + self.suspect_window_sec
                if not self.damage_detected[main_idx] and (now - self.last_damage_time) > 1.0:
                    self.damage_detected[main_idx] = True
                    self.last_damage_time = now
                    msg = String()
                    msg.data = f"ВИНТ_{main_idx + 1}_ПОВРЕЖДЕН[SPIKE] Δmain={main_delta:.3f} Δopp={opp_delta:.3f}"
                    self.detection_pub.publish(msg)
                    self.node.get_logger().error(msg.data)

    def imu_callback(self, msg: Imu):
        self.current_gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=float)
        
        # Логируем первые несколько сообщений для отладки
        if not hasattr(self, '_imu_debug_count'):
            self._imu_debug_count = 0
        if self._imu_debug_count < 3:
            self.node.get_logger().info(
                f'[IMU DEBUG #{self._imu_debug_count}] gyro=({self.current_gyro[0]:.3f}, {self.current_gyro[1]:.3f}, {self.current_gyro[2]:.3f})'
            )
            self._imu_debug_count += 1

        if not self.gyro_calibration_complete:
            gyro_norm = np.linalg.norm(self.current_gyro)
            if not self.calibration_started:
                if gyro_norm < self.stable_gyro_threshold:
                    self.stable_samples_count += 1
                    if self.stable_samples_count >= self.stable_samples_required:
                        self.calibration_started = True
                        self.node.get_logger().info('[KALMAN] Дрон стабилен. Начинаем калибровку baseline...')
                else:
                    self.stable_samples_count = 0
            if self.calibration_started:
                if self.gyro_lpf is None:
                    self.gyro_lpf = self.current_gyro.copy()
                else:
                    self.gyro_lpf = self.gyro_alpha * self.current_gyro + (1-self.gyro_alpha) * self.gyro_lpf
                if len(self.gyro_calib_samples) < self.gyro_cal_required_samples:
                    self.gyro_calib_samples.append(self.gyro_lpf.copy())
                    if len(self.gyro_calib_samples) % 20 == 0:
                        self.node.get_logger().info(f'[KALMAN] Калибровка: {len(self.gyro_calib_samples)}/{self.gyro_cal_required_samples}')
                else:
                    self.baseline_gyro = np.mean(np.vstack(self.gyro_calib_samples), axis=0)
                    self.gyro_calibration_complete = True
                    self.node.get_logger().info(f'[KALMAN] Калибровка завершена! Baseline gyro: {self.baseline_gyro}')

    def _build_gyro_H(self):
        return np.array([
            [+self.k_roll, -self.k_roll, +self.k_roll, -self.k_roll],
            [+self.k_pitch, +self.k_pitch, -self.k_pitch, -self.k_pitch],
            [+self.k_yaw, -self.k_yaw, -self.k_yaw, +self.k_yaw],
        ])
    
    def detect_damaged_propeller_by_pattern(self, gyro_error):
        patterns = {
            0: np.array([+1.0, -1.0, +1.0]),
            1: np.array([-1.0, -1.0, -1.0]),
            2: np.array([+1.0, +1.0, -1.0]),
            3: np.array([-1.0, +1.0, +1.0]),
        }
        norm = np.linalg.norm(gyro_error)
        if norm < 0.15:
            return -1, 0.0
        norm_error = gyro_error / norm
        best_match = -1
        best_score = -1.0
        for pid, pattern in patterns.items():
            score = float(norm_error.dot(pattern))
            if score > best_score:
                best_score = score
                best_match = pid
        now = time.time()
        if best_score > 0.7 and norm > 0.3:
            self.pattern_history.append((best_match, best_score, now))
        self.pattern_history = [(p, s, t) for (p, s, t) in self.pattern_history if (now - t) < self.pattern_window_sec]
        if len(self.pattern_history) >= 15:
            votes = {0:0.0,1:0.0,2:0.0,3:0.0}
            for p,s,_ in self.pattern_history:
                votes[p] += s
            accumulated_best = max(votes, key=votes.get)
            accumulated_score = votes[accumulated_best] / len(self.pattern_history)
            if accumulated_score > 0.6:
                return accumulated_best, accumulated_score
        return best_match, best_score

    def update(self, time_sec: float, period_sec: float):
        if self.current_gyro is None:
            if not hasattr(self, '_no_imu_logged'):
                self.node.get_logger().warn('[KALMAN] ⚠️ Нет данных гироскопа!')
                self._no_imu_logged = True
            return

        if not self.gyro_calibration_complete or self.baseline_gyro is None:
            if not hasattr(self, '_waiting_calib_logged'):
                self.node.get_logger().info(f'[KALMAN] Ожидание калибровки... ({len(self.gyro_calib_samples)}/{self.gyro_cal_required_samples})')
                self._waiting_calib_logged = True
            return
        
        # Сбрасываем флаги после начала работы
        if hasattr(self, '_waiting_calib_logged'):
            delattr(self, '_waiting_calib_logged')
        if hasattr(self, '_no_imu_logged'):
            delattr(self, '_no_imu_logged')

        # Вычисляем отклонения от baseline
        gyro_err = self.current_gyro - self.baseline_gyro
        gyro_err_norm = np.linalg.norm(gyro_err)
        
        H_gyro = self._build_gyro_H()  # 3x4 матрица
        
        # Обновляем фильтр если есть значимое отклонение гироскопа
        min_update_norm = 0.05
        if gyro_err_norm > min_update_norm:
            self.kalman.predict()
            try:
                # Сохраняем старые эффективности для сравнения
                old_eff = self.propeller_efficiency.copy()
                
                self.kalman.update(gyro_err, H_gyro)
                raw_eff = self.kalman.get_states()
                
                # СГЛАЖИВАНИЕ: ограничиваем скорость изменения эффективностей
                for i in range(4):
                    change = raw_eff[i] - self.propeller_efficiency_smoothed[i]
                    # Ограничиваем максимальное изменение за одно обновление
                    change = np.clip(change, -self.max_efficiency_change_per_update, self.max_efficiency_change_per_update)
                    # Применяем сглаживание (low-pass filter)
                    self.propeller_efficiency_smoothed[i] += change * self.efficiency_smoothing_alpha
                    # Ограничиваем значения [0, 1]
                    self.propeller_efficiency_smoothed[i] = np.clip(self.propeller_efficiency_smoothed[i], 0.0, 1.0)
                
                # Используем сглаженные эффективности
                self.propeller_efficiency = self.propeller_efficiency_smoothed.copy()
                self._detect_efficiency_spikes(self.propeller_efficiency - self.prev_efficiency)
                self.prev_efficiency = self.propeller_efficiency.copy()

                self.efficiency_history_with_activity.append(self.propeller_efficiency.copy())
                if len(self.efficiency_history_with_activity) > self.max_history_size:
                    self.efficiency_history_with_activity.pop(0)
                
                # Логируем обновление фильтра (только при значительных изменениях)
                max_change = np.max(np.abs(self.propeller_efficiency - old_eff))
                if max_change > 0.05:  # Логируем только если изменение > 5%
                    self.node.get_logger().info(
                        f'[KALMAN] Обновление: gyro_err={gyro_err_norm:.3f} | '
                        f'ЭФФ: FR={self.propeller_efficiency[0]:.3f} FL={self.propeller_efficiency[1]:.3f} '
                        f'RR={self.propeller_efficiency[2]:.3f} RL={self.propeller_efficiency[3]:.3f}'
                    )
            except Exception as e:
                self.node.get_logger().warn(f'[KALMAN] Ошибка обновления фильтра: {e}')

        damaged_prop, pattern_score = self.detect_damaged_propeller_by_pattern(gyro_err)

        if self.enable_detection:
            now = time.time()
            if self.suspected_prop is not None and now >= self.suspected_expiry:
                self.suspected_prop = None

            if len(self.efficiency_history_with_activity) >= self.history_window:
                window = self.history_window
                recent_active = np.array(self.efficiency_history_with_activity[-window:])
                min_efficiency = np.min(recent_active, axis=0)
                avg_efficiency = np.mean(recent_active, axis=0)
                eff_for_detection = np.minimum(min_efficiency, avg_efficiency * 0.9)
            else:
                eff_for_detection = self.propeller_efficiency

            limit_to_prop = self.suspected_prop if (self.suspected_prop is not None and now < self.suspected_expiry) else None

            # ОТНОСИТЕЛЬНАЯ детекция: детектируем только винт, который ЗНАЧИТЕЛЬНО хуже остальных
            # Вычисляем среднюю эффективность всех винтов
            avg_all_eff = np.mean(eff_for_detection)
            
            # Находим винт с минимальной эффективностью
            worst_pid = np.argmin(eff_for_detection)
            worst_eff = eff_for_detection[worst_pid]
            
            # Детектируем только если:
            # 1. Эффективность худшего винта ниже порога
            # 2. И он ЗНАЧИТЕЛЬНО хуже среднего (минимум на 25% ниже)
            # 3. И он хуже остальных минимум на 20%
            relative_threshold = 0.25  # Должен быть на 25% хуже среднего
            min_diff_from_others = 0.20  # Должен быть минимум на 20% хуже остальных
            
            if worst_eff < self.damage_threshold:
                # Проверяем относительное отклонение от среднего
                relative_diff = (avg_all_eff - worst_eff) / avg_all_eff if avg_all_eff > 0.01 else 0.0
                
                # Проверяем разницу с остальными винтами
                other_effs = [eff_for_detection[i] for i in range(4) if i != worst_pid]
                min_other_eff = min(other_effs) if other_effs else 1.0
                diff_from_others = (min_other_eff - worst_eff) / min_other_eff if min_other_eff > 0.01 else 0.0
                
                # Логируем лишь общую статистику, без строк "Худший..." каждую секунду
                if not hasattr(self, '_last_detection_log_time') or (now - self._last_detection_log_time) > 2.0:
                    self.node.get_logger().info(
                        f'[KALMAN] Анализ: FR={eff_for_detection[0]:.3f} FL={eff_for_detection[1]:.3f} '
                        f'RR={eff_for_detection[2]:.3f} RL={eff_for_detection[3]:.3f}'
                    )
                    self._last_detection_log_time = now
                
                # Детектируем только если винт ЗНАЧИТЕЛЬНО хуже остальных
                if (relative_diff >= relative_threshold and diff_from_others >= min_diff_from_others):
                    if limit_to_prop is not None and worst_pid != limit_to_prop:
                        pass
                    elif not self.damage_detected[worst_pid]:
                        if len(self.efficiency_history_with_activity) >= self.history_window:
                            recent_effs = [h[worst_pid] for h in self.efficiency_history_with_activity[-self.history_window:]]
                            low_count = sum(1 for e in recent_effs if e < self.damage_threshold)
                            required_lows = max(2, self.history_window - 1)
                            if low_count >= required_lows and (now - self.last_damage_time) > 2.0:
                                self.damage_detected[worst_pid] = True
                                self.last_damage_time = now
                                msg = String()
                                msg.data = f"ВИНТ_{worst_pid+1}_ПОВРЕЖДЕН[KALMAN] eff={worst_eff:.3f} (отн.={relative_diff:.1%}, разница={diff_from_others:.1%}, {low_count}/{self.history_window} низких)"
                                self.detection_pub.publish(msg)
                                self.node.get_logger().error(msg.data)
                                if self.suspected_prop == worst_pid:
                                    self.suspected_prop = None
                        elif gyro_err_norm > min_update_norm and (now - self.last_damage_time) > 2.0:
                            self.damage_detected[worst_pid] = True
                            self.last_damage_time = now
                            msg = String()
                            msg.data = f"ВИНТ_{worst_pid+1}_ПОВРЕЖДЕН[KALMAN] eff={worst_eff:.3f} (отн.={relative_diff:.1%}, разница={diff_from_others:.1%})"
                            self.detection_pub.publish(msg)
                            self.node.get_logger().error(msg.data)
                            if self.suspected_prop == worst_pid:
                                self.suspected_prop = None

            if damaged_prop >= 0 and pattern_score > 0.7:
                if self.propeller_efficiency[damaged_prop] < 0.8:
                    if not self.damage_detected[damaged_prop] and (now - self.last_damage_time) > 3.0:
                        if limit_to_prop is None or damaged_prop == limit_to_prop:
                            self.damage_detected[damaged_prop] = True
                            self.last_damage_time = now
                            msg = String()
                            msg.data = f"ВИНТ_{damaged_prop+1}_ПОВРЕЖДЕН[PATTERN+KALMAN] score={pattern_score:.2f} eff={self.propeller_efficiency[damaged_prop]:.3f}"
                            self.detection_pub.publish(msg)
                            self.node.get_logger().error(msg.data)
                            if self.suspected_prop == damaged_prop:
                                self.suspected_prop = None


        status_msg = String()
        damaged_count = sum(self.damage_detected)
        avg_eff = np.mean(self.propeller_efficiency)
        status_msg.data = f"ПОВРЕЖДЕНО {damaged_count}/4 | СР. ЭФФ: {avg_eff:.2%}"
        self.status_pub.publish(status_msg)

        eff_msg = Float32MultiArray()
        eff_msg.data = self.propeller_efficiency.tolist()
        self.efficiency_pub.publish(eff_msg)

        health_msg = Bool()
        health_msg.data = not any(self.damage_detected)
        self.propeller_health_pub.publish(health_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Node('kalman_damage_controller_node')
    
    # Читаем параметры из launch файла
    node.declare_parameter('damage_threshold', 0.65)
    node.declare_parameter('enable_detection', True)
    
    controller = KalmanDamageController(node)
    
    qos = QoSProfile(depth=10)
    controller.damage_threshold = float(node.get_parameter('damage_threshold').value)
    controller.enable_detection = bool(node.get_parameter('enable_detection').value)

    node.create_subscription(Imu, '/imu', controller.imu_callback, qos)
    node.create_subscription(Imu, '/Mavic_2_PRO/imu', controller.imu_callback, qos)

    timer_period = 0.01  # 100 Hz
    timer = node.create_timer(timer_period, lambda: controller.update(node.get_clock().now().nanoseconds / 1e9, timer_period))

    node.get_logger().info('='*60)
    node.get_logger().info('[KALMAN] Kalman Damage Controller started')
    node.get_logger().info(f'[KALMAN] Damage threshold: {controller.damage_threshold}')
    node.get_logger().info(f'[KALMAN] Detection enabled: {controller.enable_detection}')
    node.get_logger().info('[KALMAN] Подписка на топики: /imu и /Mavic_2_PRO/imu')
    node.get_logger().info('[KALMAN] Ожидание калибровки IMU (100 измерений)...')
    node.get_logger().info('='*60)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
