#!/usr/bin/env python3
import math
import random
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, Imu, JointState

from std_msgs.msg import Int32
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, PointStamped
import tf2_geometry_msgs
import rclpy.time

from detector import ObjectDetector


class HLInterfaceController(Node):
    """
    Prototype controller for competitors.

    What it does:
    - publishes velocity commands to /cmd_vel
    - subscribes to robot state topics published by bridge_node
    - exposes high-level helper functions that can reused
    - sends random commands periodically as a demo
    """

    def __init__(self):
        super().__init__("controller")

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.vel_sub = self.create_subscription(
            TwistStamped,
            "/aliengo/base_velocity",
            self._vel_callback,
            10,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            "/aliengo/joint_states",
            self._joint_callback,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/aliengo/imu",
            self._imu_callback,
            10,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            "/aliengo/camera/color/image_raw",
            self._rgb_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/aliengo/camera/depth/image_raw",
            self._depth_callback,
            10,
        )

        # ---------------- Cached state ----------------
        self.latest_base_velocity = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
            "stamp_sec": None,
        }

        self.latest_joint_state = {
            "names": [],
            "position": [],
            "velocity": [],
            "name_to_index": {},
            "stamp_sec": None,
        }

        self.latest_imu = {
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "stamp_sec": None,
        }

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_info = {
            "height": 0,
            "width": 0,
            "encoding": None,
            "stamp_sec": None,
        }

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_info = {
            "height": 0,
            "width": 0,
            "encoding": None,
            "stamp_sec": None,
        }

        # ---------------- Demo behavior ----------------
        self.command_duration = 2.0
        self.last_command_change_time = self._now_sec()
        self.current_demo_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.demo_enabled = True

        self.log_period = 1.0
        self.last_log_time = 0.0

        self.create_timer(0.05, self._main_loop)

        self.get_logger().info("Controller started.")
        self.get_logger().info("This node publishes random demo commands and exposes HL helper functions.")

        self.detected_object_pub = self.create_publisher(Int32, "/competition/detected_object", 10)
        self.nav = BasicNavigator()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.object_queue = [1, 2] # Твое задание: сначала 1, потом 2
        self.current_target_idx = 0
        self.is_navigating = False

        self.detector = ObjectDetector(log_detections=False)

        self.object_queue = [1, 2]   # порядок обхода
        self.current_target_idx = 0

        self.reported_targets = set()
        self.known_targets = {}

        self.camera_frame = "camera_link_optical"
        self.map_frame = "map"

        self.search_timer = self._now_sec()
        self.scan_dir = 1.0
    
    def get_3d_pose(self, u, v):
        if self.latest_depth is None: return None
        z = self.latest_depth[v, u]
        if not np.isfinite(z) or z <= 0: return None

        # Параметры камеры (fx, fy, cx, cy). Уточни на месте, если 640x480:
        fx = fy = 450.0 
        cx, cy = 320.0, 240.0

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        p = PoseStamped()
        p.header.frame_id = "camera_link_optical" # Имя фрейма из вашего TF-дерева
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = float(z)
        p.pose.position.y = float(-x)
        p.pose.position.z = float(-y)

        try:
            transform = self.tf_buffer.lookup_transform("map", p.header.frame_id, rclpy.time.Time())
            return tf2_geometry_msgs.do_transform_pose(p.pose, transform)
        except:
            return None
        
    def publish_detected_object(self, object_id: int):
        msg = Int32()
        msg.data = int(object_id)
        self.detected_object_pub.publish(msg)

    def _wrap_angle(self, a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _get_camera_pose_map(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.camera_frame, rclpy.time.Time())
        except Exception:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        return np.array([t.x, t.y, yaw], dtype=np.float32)
    
    def _transform_point_to_map(self, xyz_camera):
        pt = PointStamped()
        pt.header.frame_id = self.camera_frame
        pt.header.stamp = self.get_clock().now().to_msg()

        pt.point.x = float(xyz_camera[0])
        pt.point.y = float(xyz_camera[1])
        pt.point.z = float(xyz_camera[2])

        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.camera_frame, rclpy.time.Time())
            pt_map = tf2_geometry_msgs.do_transform_point(pt, tf)

            return np.array([
                pt_map.point.x,
                pt_map.point.y,
                pt_map.point.z
            ], dtype=np.float32)

        except Exception:
            return None
    
    def _drive_to_goal(self, goal_xy):
        pose = self._get_camera_pose_map()
        if pose is None:
            self.stop_robot()
            return

        dx = float(goal_xy[0] - pose[0])
        dy = float(goal_xy[1] - pose[1])

        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        err = self._wrap_angle(target_yaw - float(pose[2]))

        if dist < 0.45:
            self.stop_robot()
            return

        vx = min(0.6, 0.35 * dist)
        wz = max(-1.0, min(1.0, 1.8 * err))

        if abs(err) > 0.9:
            vx = 0.0

        self.send_command(vx, 0.0, wz)
    
    def _explore(self):
        now = self._now_sec()

        if now - self.search_timer > 2.5:
            self.scan_dir *= -1.0
            self.search_timer = now

        self.send_command(0.12, 0.0, 0.55 * self.scan_dir)

    def _update_occupancy_from_depth(self, depth):
        h, w = depth.shape[:2]

        fx = fy = 450.0
        cx, cy = w / 2.0, h / 2.0

        if self._get_camera_pose_map() is None:
            return

        step = 24

        for v in range(h // 4, h - 10, step):
            for u in range(0, w, step):
                z = float(depth[v, u])

                if not np.isfinite(z) or z <= 0.05:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                self._transform_point_to_map((x, y, z))

    # =====================================================================
    # High-level API competitors can use
    # =====================================================================
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

    def get_base_velocity(self) -> Dict[str, float]:
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
        return {name: float(value) for name, value in zip(names, pos)}

    def get_joint_velocities(self) -> Dict[str, float]:
        names = self.latest_joint_state["names"]
        vel = self.latest_joint_state["velocity"]
        return {name: float(value) for name, value in zip(names, vel)}

    def get_joint_position(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        if idx is None:
            return None
        return float(self.latest_joint_state["position"][idx])

    def get_joint_velocity(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        if idx is None:
            return None
        return float(self.latest_joint_state["velocity"][idx])

    def get_imu(self) -> Dict[str, float]:
        return dict(self.latest_imu)

    def get_rgb_image(self) -> Optional[np.ndarray]:
        if self.latest_rgb is None:
            return None
        return self.latest_rgb.copy()

    def get_depth_image(self) -> Optional[np.ndarray]:
        if self.latest_depth is None:
            return None
        return self.latest_depth.copy()

    def get_depth_center(self) -> Optional[float]:
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        center_value = self.latest_depth[h // 2, w // 2]
        if not np.isfinite(center_value):
            return None
        return float(center_value)

    def robot_state_ready(self) -> bool:
        return (
            self.latest_base_velocity["stamp_sec"] is not None
            and self.latest_joint_state["stamp_sec"] is not None
            and self.latest_imu["stamp_sec"] is not None
        )

    # =====================================================================
    # Example competitor entry point
    # =====================================================================
    def run_user_code(self):

        if self.current_target_idx >= len(self.object_queue):
            self.stop_robot()
            return

        if self.latest_rgb is None or self.latest_depth is None:
            self._explore()
            return

        target_id = self.object_queue[self.current_target_idx]

        # обновляем карту
        self._update_occupancy_from_depth(self.latest_depth)

        # YOLO
        detections = self.detector.detect(self.latest_rgb, self.latest_depth)

        target_det = next((d for d in detections if d.object_id == target_id), None)

        # если видим нужный объект
        if target_det is not None:

            target_map = self._transform_point_to_map(target_det.xyz_camera)

            if target_map is not None:
                self.known_targets[target_id] = target_map

                if target_id not in self.reported_targets:
                    self.publish_detected_object(target_id)
                    self.reported_targets.add(target_id)

                cam_pose = self._get_camera_pose_map()

                if cam_pose is not None:
                    dist = math.hypot(
                        float(target_map[0] - cam_pose[0]),
                        float(target_map[1] - cam_pose[1])
                    )

                    if dist < 0.45:
                        self.stop_robot()
                        self.current_target_idx += 1
                        return

                self._drive_to_goal(target_map[:2])
                return

        # если уже знаем где объект
        if target_id in self.known_targets:
            self._drive_to_goal(self.known_targets[target_id][:2])
            return

        # иначе ищем
        self._explore()

    # =====================================================================
    # Internal callbacks
    # =====================================================================
    def _vel_callback(self, msg: TwistStamped) -> None:
        self.latest_base_velocity = {
            "vx": float(msg.twist.linear.x),
            "vy": float(msg.twist.linear.y),
            "wz": float(msg.twist.angular.z),
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _joint_callback(self, msg: JointState) -> None:
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        self.latest_joint_state = {
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "name_to_index": name_to_index,
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
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
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
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
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

    # =====================================================================
    # Main loop
    # =====================================================================
    def _main_loop(self) -> None:
        if self.demo_enabled:
            self.run_user_code()

        now = self._now_sec()
        if now - self.last_log_time >= self.log_period:
            self.last_log_time = now
            self._log_status()

    def _log_status(self) -> None:
        vel = self.get_base_velocity()
        imu = self.get_imu()
        depth_center = self.get_depth_center()

        fl_hip = self.get_joint_position("FL_hip_joint")
        fr_hip = self.get_joint_position("FR_hip_joint")

        depth_text = "None" if depth_center is None else f"{depth_center:.3f}"
        fl_text = "None" if fl_hip is None else f"{fl_hip:.3f}"
        fr_text = "None" if fr_hip is None else f"{fr_hip:.3f}"

        self.get_logger().info(
            "state | "
            f"ready={self.robot_state_ready()} | "
            f"cmd=(vx={self.current_demo_cmd['vx']:.3f}, vy={self.current_demo_cmd['vy']:.3f}, wz={self.current_demo_cmd['wz']:.3f}) | "
            f"vel=(vx={vel['vx']:.3f}, vy={vel['vy']:.3f}, wz={vel['wz']:.3f}) | "
            f"imu_wz={imu['wz']:.3f} | "
            f"depth_center={depth_text} | "
            f"FL_hip={fl_text} | "
            f"FR_hip={fr_text}"
        )

    # =====================================================================
    # Utilities
    # =====================================================================
    def _sample_random_command(self) -> Dict[str, float]:
        vx = random.uniform(-0.8, 0.8)
        vy = random.uniform(-0.2, 0.2)
        wz = random.uniform(-0.8, 0.8)

        if abs(vx) < 0.08:
            vx = 0.0
        if abs(vy) < 0.05:
            vy = 0.0
        if abs(wz) < 0.08:
            wz = 0.0

        return {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
        }

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _msg_time_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None):
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
