#!/usr/bin/env python3
"""
Контроллер для автономного прохождения полигона с препятствиями.

Стратегия:
  EXPLORING  — обход препятствий по depth-камере, поиск текущей цели
  APPROACHING — цель видна, рулим к ней по центру bbox + depth
  REACHED    — цель достигнута (<= REACH_DISTANCE_M), стоим, подтверждаем
  FINISHED   — все цели пройдены

Логирование:
  - Каждый новый обнаруженный объект — с секундомером
  - Периодический статус: состояние, цель, глубина, посещения
  - Детектор пишет свои логи (класс, conf, dist, FPS, RAM/VRAM)
"""
import logging
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, Imu, JointState

# ── CV-модуль ──────────────────────────────────────────────────────────────
_CV_DIR = Path(__file__).resolve().parent / "cv"
if str(_CV_DIR) not in sys.path:
    sys.path.insert(0, str(_CV_DIR))

from detector_eugene import (  # noqa: E402
    LABEL_RU,
    OBJECT_CLASSES,
    REACH_DISTANCE_M,
    Detection,
    ObjectDetector,
)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

_LOG = logging.getLogger(__name__)

# ── Последовательность целей (можно задать через аргументы запуска) ─────────
TARGET_SEQUENCE: List[str] = ["cup", "laptop", "bottle", "chair", "backpack"]

# ── Параметры навигации ──────────────────────────────────────────────────────
FWD_SPEED = 0.5          # м/с — движение вперёд при исследовании
TURN_SPEED = 0.7         # рад/с — поворот при обходе препятствия
SAFE_DIST = 1.2          # м — начало манёвра уклонения
HARD_DIST = 0.55         # м — жёсткий останов и поворот на месте

# Управление при подходе к цели
KP_STEER = 1.2           # P-коэффициент руления по bbox
APPROACH_FWD = 0.35      # м/с при дальнем подходе
APPROACH_FWD_CLOSE = 0.15  # м/с при подходе вплотную (dist < 0.8 м)

# Запускать детектор не чаще 1 раза в DETECT_PERIOD секунд
DETECT_PERIOD = 0.33     # ≈3 fps


class NavState(Enum):
    EXPLORING  = "EXPLORING"
    APPROACHING = "APPROACHING"
    REACHED    = "REACHED"
    FINISHED   = "FINISHED"


def _depth_sector(patch: np.ndarray) -> float:
    """Медиана валидных значений глубины патча; 999.0 если нет данных."""
    valid = patch[(patch > 0.05) & np.isfinite(patch)]
    return float(np.median(valid)) if valid.size > 0 else 999.0


class HLInterfaceController(Node):
    """
    Контроллер участника для автономного прохождения полигона.

    Публикует команды скорости в /cmd_vel, подписывается на топики робота,
    использует ObjectDetector для обнаружения объектов-целей.
    """

    def __init__(self) -> None:
        super().__init__("controller")

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.vel_sub = self.create_subscription(
            TwistStamped, "/aliengo/base_velocity", self._vel_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, "/aliengo/joint_states", self._joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, "/aliengo/imu", self._imu_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, "/aliengo/camera/color/image_raw", self._rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/aliengo/camera/depth/image_raw", self._depth_callback, 10)

        # ── Кэш состояния сенсоров ────────────────────────────────────────────
        self.latest_base_velocity: Dict = {
            "vx": 0.0, "vy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_joint_state: Dict = {
            "names": [], "position": [], "velocity": [],
            "name_to_index": {}, "stamp_sec": None}
        self.latest_imu: Dict = {
            "wx": 0.0, "wy": 0.0, "wz": 0.0, "stamp_sec": None}
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_info: Dict = {
            "height": 0, "width": 0, "encoding": None, "stamp_sec": None}
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_info: Dict = {
            "height": 0, "width": 0, "encoding": None, "stamp_sec": None}

        # ── Детектор объектов ─────────────────────────────────────────────────
        self._detector = ObjectDetector(log_detections=True)
        self._last_detect_time: float = 0.0
        self._last_detections: List[Detection] = []

        # label -> время первого обнаружения (секунды с запуска)
        self._first_seen: Dict[str, float] = {}

        # ── Навигация ─────────────────────────────────────────────────────────
        self._target_seq: List[str] = list(TARGET_SEQUENCE)
        self._target_idx: int = 0
        self._nav_state: NavState = NavState.EXPLORING
        self._visited: Set[str] = set()

        # Время, когда была достигнута текущая цель (для подтверждения)
        self._reached_at: Optional[float] = None
        self._reached_confirm_sec: float = 1.5

        # ── Секундомер ────────────────────────────────────────────────────────
        self._start_time: float = time.perf_counter()

        # ── Периодический лог статуса ────────────────────────────────────────
        self._log_period: float = 2.0
        self._last_log_time: float = 0.0

        self.create_timer(0.05, self._main_loop)

        self.get_logger().info(
            f"[{self.elapsed():.1f}с] Контроллер запущен. "
            f"Последовательность целей: {self._target_seq}"
        )

    # =========================================================================
    # Высокоуровневое API
    # =========================================================================

    def send_command(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.send_command(0.0, 0.0, 0.0)

    def get_base_velocity(self) -> Dict:
        return dict(self.latest_base_velocity)

    def get_vx(self) -> float:
        return float(self.latest_base_velocity["vx"])

    def get_vy(self) -> float:
        return float(self.latest_base_velocity["vy"])

    def get_wz(self) -> float:
        return float(self.latest_base_velocity["wz"])

    def get_joint_names(self) -> List[str]:
        return list(self.latest_joint_state["names"])

    def get_joint_positions(self) -> Dict[str, float]:
        names = self.latest_joint_state["names"]
        pos = self.latest_joint_state["position"]
        return {n: float(v) for n, v in zip(names, pos)}

    def get_joint_velocities(self) -> Dict[str, float]:
        names = self.latest_joint_state["names"]
        vel = self.latest_joint_state["velocity"]
        return {n: float(v) for n, v in zip(names, vel)}

    def get_joint_position(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        return None if idx is None else float(self.latest_joint_state["position"][idx])

    def get_joint_velocity(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        return None if idx is None else float(self.latest_joint_state["velocity"][idx])

    def get_imu(self) -> Dict:
        return dict(self.latest_imu)
    
    def publish_detected_object(self, object_id):
        # Существующая логика ROS (публикация в топик)
        msg = Int32()
        msg.data = int(object_id)
        self.object_pub.publish(msg)
        
        # Дополнительно: отправка напрямую в бридж для isaac_controller.py
        # Если ваш бридж поддерживает прямую отправку:
        # self.bridge.send_object_id(object_id)

    def get_rgb_image(self) -> Optional[np.ndarray]:
        return self.latest_rgb.copy() if self.latest_rgb is not None else None

    def get_depth_image(self) -> Optional[np.ndarray]:
        return self.latest_depth.copy() if self.latest_depth is not None else None

    def get_depth_center(self) -> Optional[float]:
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        v = self.latest_depth[h // 2, w // 2]
        return float(v) if np.isfinite(v) else None

    def robot_state_ready(self) -> bool:
        return (
            self.latest_base_velocity["stamp_sec"] is not None
            and self.latest_joint_state["stamp_sec"] is not None
            and self.latest_imu["stamp_sec"] is not None
        )

    def elapsed(self) -> float:
        """Секунды с момента запуска контроллера."""
        return time.perf_counter() - self._start_time

    # =========================================================================
    # Основной цикл
    # =========================================================================

    def _main_loop(self) -> None:
        self.run_user_code()

        now = self._now_sec()
        if now - self._last_log_time >= self._log_period:
            self._last_log_time = now
            self._log_status()

    def run_user_code(self) -> None:
        """Главная логика: детекция → навигация по состоянию."""
        if self._nav_state == NavState.FINISHED:
            self.stop_robot()
            return

        self._try_detect()

        target_label = self._current_target_label()

        if self._nav_state == NavState.EXPLORING:
            self._state_explore(target_label)
        elif self._nav_state == NavState.APPROACHING:
            self._state_approach(target_label)
        elif self._nav_state == NavState.REACHED:
            self._state_reached()

    # =========================================================================
    # Состояния навигации
    # =========================================================================

    def _state_explore(self, target_label: Optional[str]) -> None:
        """
        Исследование: реактивный обход препятствий.
        Если текущая цель появляется в детекциях — переходим в APPROACHING.
        """
        target_det = self._find_detection(target_label)
        if target_det is not None:
            self._nav_state = NavState.APPROACHING
            ru = LABEL_RU.get(target_label or "", target_label or "")
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] Цель найдена: {ru} ({target_label}) "
                f"на {target_det.distance_m:.2f}м → APPROACHING"
            )
            return

        vx, wz = self._obstacle_avoid_cmd(safe_dist=SAFE_DIST)
        self.send_command(vx, 0.0, wz)

    def _state_approach(self, target_label: Optional[str]) -> None:
        """
        Подход к цели: рулим по центру bbox, контролируем расстояние.
        При потере цели возвращаемся в EXPLORING.
        """
        target_det = self._find_detection(target_label)
        if target_det is None:
            self._nav_state = NavState.EXPLORING
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] Цель {target_label} потеряна → EXPLORING"
            )
            return

        if target_det.is_reached:
            self._nav_state = NavState.REACHED
            self._reached_at = self.elapsed()
            ru = LABEL_RU.get(target_label or "", target_label or "")
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] ✓ ДОСТИГНУТ: {ru} ({target_label}) "
                f"dist={target_det.distance_m:.2f}м"
            )
            self.stop_robot()
            return

        # Рулевое управление по горизонтальному смещению центра bbox
        x1, _y1, x2, _y2 = target_det.bbox
        img_w = self.latest_rgb_info["width"] or 640
        bbox_cx = (x1 + x2) / 2.0
        err = (bbox_cx - img_w / 2.0) / (img_w / 2.0)  # [-1, +1]
        wz = float(np.clip(-KP_STEER * err, -TURN_SPEED, TURN_SPEED))

        dist = target_det.distance_m if target_det.distance_m > 0 else 2.0
        vx = APPROACH_FWD_CLOSE if dist < 0.8 else APPROACH_FWD

        # Если впереди критическое препятствие — уступаем obstacle avoidance
        _, wz_obs = self._obstacle_avoid_cmd(safe_dist=HARD_DIST)
        if wz_obs != 0.0:
            wz = wz_obs
            vx = 0.0

        self.send_command(vx, 0.0, wz)

    def _state_reached(self) -> None:
        """
        Подтверждение достижения: стоим REACHED_CONFIRM_SEC секунд,
        затем переходим к следующей цели.
        """
        self.stop_robot()
        if self._reached_at is None:
            self._reached_at = self.elapsed()
        if self.elapsed() - self._reached_at >= self._reached_confirm_sec:
            self._advance_target()

    def _advance_target(self) -> None:
        """Зачесть текущую цель и перейти к следующей."""
        cur = self._current_target_label()
        if cur:
            self._visited.add(cur)
            ru = LABEL_RU.get(cur, cur)
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] ★ ЗАЧТЕНО: {ru} ({cur}) "
                f"| посещено: {len(self._visited)}/{len(self._target_seq)}"
            )

        self._target_idx += 1
        if self._target_idx >= len(self._target_seq):
            self._nav_state = NavState.FINISHED
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] ✓✓✓ ВСЕ ЦЕЛИ ПРОЙДЕНЫ "
                f"за {self.elapsed():.1f} секунд!"
            )
        else:
            self._nav_state = NavState.EXPLORING
            self._reached_at = None
            nxt = self._target_seq[self._target_idx]
            self.get_logger().info(
                f"[{self.elapsed():.1f}с] → Следующая цель: "
                f"{LABEL_RU.get(nxt, nxt)} ({nxt})"
            )

    # =========================================================================
    # Обход препятствий
    # =========================================================================

    def _obstacle_avoid_cmd(
        self, safe_dist: float = SAFE_DIST
    ) -> tuple[float, float]:
        """
        Возвращает (vx, wz) на основе данных depth-камеры.

        Кадр делится на 3 горизонтальных сектора (левый/центр/правый).
        Используется центральная полоса по вертикали [h/4 .. 3h/4],
        чтобы не реагировать на пол/потолок.

        Логика:
          - Центр свободен (> safe_dist) → прямо
          - Центр занят → поворот в сторону с большей дистанцией
          - Всё занято (< HARD_DIST) → вращение на месте
        """
        depth = self.latest_depth
        if depth is None:
            return FWD_SPEED, 0.0

        h, w = depth.shape
        ym, yM = h // 4, 3 * h // 4
        strip = depth[ym:yM, :]

        left_d   = _depth_sector(strip[:, : w // 3])
        center_d = _depth_sector(strip[:, w // 3 : 2 * w // 3])
        right_d  = _depth_sector(strip[:, 2 * w // 3 :])

        # Путь свободен
        if center_d > safe_dist and left_d > HARD_DIST and right_d > HARD_DIST:
            return FWD_SPEED, 0.0

        # Поворот в сторону с большей дистанцией
        wz = TURN_SPEED if left_d >= right_d else -TURN_SPEED
        vx = 0.0 if center_d < HARD_DIST else FWD_SPEED * 0.4
        return vx, wz

    # =========================================================================
    # Детекция
    # =========================================================================

    def _try_detect(self) -> None:
        """Запустить детектор (не чаще DETECT_PERIOD с)."""
        now = time.perf_counter()
        if now - self._last_detect_time < DETECT_PERIOD:
            return
        rgb = self.get_rgb_image()
        if rgb is None:
            return
        depth = self.get_depth_image()
        self._last_detect_time = now
        self._last_detections = self._detector.detect(rgb, depth)

        # Логируем первое появление каждого объекта
        for det in self._last_detections:
            if det.label not in self._first_seen:
                self._first_seen[det.label] = self.elapsed()
                ru = LABEL_RU.get(det.label, det.label)
                self.get_logger().info(
                    f"[{self.elapsed():.1f}с] [НОВЫЙ ОБЪЕКТ] "
                    f"{ru} ({det.label})  "
                    f"conf={det.confidence:.2f}  "
                    f"dist={det.distance_m:.2f}м  "
                    f"src={det.source}"
                )

    def _find_detection(self, label: Optional[str]) -> Optional[Detection]:
        """Найти лучшую детекцию для метки (первая = ближайшая)."""
        if label is None:
            return None
        for d in self._last_detections:
            if d.label == label:
                return d
        return None

    def _current_target_label(self) -> Optional[str]:
        if self._target_idx < len(self._target_seq):
            return self._target_seq[self._target_idx]
        return None

    # =========================================================================
    # Callbacks сенсоров
    # =========================================================================

    def _vel_callback(self, msg: TwistStamped) -> None:
        self.latest_base_velocity = {
            "vx": float(msg.twist.linear.x),
            "vy": float(msg.twist.linear.y),
            "wz": float(msg.twist.angular.z),
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _joint_callback(self, msg: JointState) -> None:
        self.latest_joint_state = {
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "name_to_index": {n: i for i, n in enumerate(msg.name)},
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _imu_callback(self, msg: Imu) -> None:
        self.latest_imu = {
            "wx": float(msg.angular_velocity.x),
            "wy": float(msg.angular_velocity.y),
            "wz": float(msg.angular_velocity.z),
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _rgb_callback(self, msg: Image) -> None:
        try:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
        except ValueError:
            self.get_logger().warning("Failed to reshape RGB image.")
            return
        self.latest_rgb = image.copy()
        self.latest_rgb_info = {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)
            )
        except ValueError:
            self.get_logger().warning("Failed to reshape depth image.")
            return
        self.latest_depth = depth.copy()
        self.latest_depth_info = {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    # =========================================================================
    # Логирование статуса
    # =========================================================================

    def _log_status(self) -> None:
        vel = self.get_base_velocity()
        depth_center = self.get_depth_center()
        depth_text = "н/д" if depth_center is None else f"{depth_center:.2f}м"

        target = self._current_target_label()
        target_ru = LABEL_RU.get(target, target) if target else "—"

        seen_str = (
            ", ".join(
                f"{LABEL_RU.get(lbl, lbl)}({t:.0f}с)"
                for lbl, t in sorted(self._first_seen.items(), key=lambda x: x[1])
            )
            or "—"
        )

        # Глубина по секторам для диагностики
        sector_info = ""
        if self.latest_depth is not None:
            h, w = self.latest_depth.shape
            strip = self.latest_depth[h // 4 : 3 * h // 4, :]
            ld = _depth_sector(strip[:, : w // 3])
            cd = _depth_sector(strip[:, w // 3 : 2 * w // 3])
            rd = _depth_sector(strip[:, 2 * w // 3 :])
            sector_info = f" | depth L={ld:.2f} C={cd:.2f} R={rd:.2f}"

        self.get_logger().info(
            f"[{self.elapsed():.1f}с] "
            f"state={self._nav_state.value} | "
            f"цель={target_ru} | "
            f"vx={vel['vx']:.2f} wz={vel['wz']:.2f} | "
            f"depth_center={depth_text}"
            f"{sector_info} | "
            f"посещено={len(self._visited)}/{len(self._target_seq)} | "
            f"обнаружено=[{seen_str}]"
        )

    # =========================================================================
    # Утилиты
    # =========================================================================

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _msg_time_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HLInterfaceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
