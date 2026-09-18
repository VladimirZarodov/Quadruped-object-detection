import socket
import json
import struct

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, JointState, Imu


CMD_IP = "127.0.0.1"
CMD_PORT = 5005

STATE_IP = "127.0.0.1"
STATE_PORT = 5006

RGB_IP = "127.0.0.1"
RGB_PORT = 5007

DEPTH_IP = "127.0.0.1"
DEPTH_PORT = 5008

JOINT_STATE_IP = "127.0.0.1"
JOINT_STATE_PORT = 5009

IMU_IP = "127.0.0.1"
IMU_PORT = 5010

class BridgeNode(Node):
    def __init__(self):
        super().__init__("bridge_node")

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind((STATE_IP, STATE_PORT))
        self.state_sock.setblocking(False)

        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock.bind((RGB_IP, RGB_PORT))
        self.rgb_sock.setblocking(False)

        # self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.depth_sock.bind((DEPTH_IP, DEPTH_PORT))
        # self.depth_sock.setblocking(False)

        self.depth_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.depth_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.depth_server_sock.bind((DEPTH_IP, DEPTH_PORT))
        self.depth_server_sock.listen(1)
        self.depth_server_sock.setblocking(False)

        self.depth_conn = None

        self.joint_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.joint_state_sock.bind((JOINT_STATE_IP, JOINT_STATE_PORT))
        self.joint_state_sock.setblocking(False)

        self.imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu_sock.bind((IMU_IP, IMU_PORT))
        self.imu_sock.setblocking(False)

        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10,
        )

        self.vel_pub = self.create_publisher(
            TwistStamped,
            "/aliengo/base_velocity",
            10,
        )

        self.rgb_pub = self.create_publisher(
            Image,
            "/aliengo/camera/color/image_raw",
            10,
        )

        self.depth_pub = self.create_publisher(
            Image,
            "/aliengo/camera/depth/image_raw",
            10,
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            "/aliengo/joint_states",
            10,
        )

        self.imu_pub = self.create_publisher(
            Imu,
            "/aliengo/imu",
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("ROS bridge node started.")

    def cmd_callback(self, msg: Twist):
        payload = {
            "vx": msg.linear.x,
            "vy": msg.linear.y,
            "wz": msg.angular.z,
        }

        data = json.dumps(payload).encode("utf-8")
        self.cmd_sock.sendto(data, (CMD_IP, CMD_PORT))
    
    def recv_exact(self, sock, size):
        chunks = []
        received = 0
        while received < size:
            chunk = sock.recv(size - received)
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def timer_callback(self):
        try:
            data, _ = self.state_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))

            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = "base"

            out.twist.linear.x = float(msg.get("vx", 0.0))
            out.twist.linear.y = float(msg.get("vy", 0.0))
            out.twist.angular.z = float(msg.get("wz", 0.0))

            self.vel_pub.publish(out)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"state receive error: {e}")

        try:
            data, _ = self.rgb_sock.recvfrom(65535)

            np_arr = np.frombuffer(data, dtype=np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_bgr is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "front_camera"
                msg.height = image_rgb.shape[0]
                msg.width = image_rgb.shape[1]
                msg.encoding = "rgb8"
                msg.is_bigendian = 0
                msg.step = image_rgb.shape[1] * 3
                msg.data = image_rgb.tobytes()

                self.rgb_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"rgb receive error: {e}")

        # try:
        #     data, _ = self.depth_sock.recvfrom(65535)

        #     np_arr = np.frombuffer(data, dtype=np.uint8)
        #     depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        #     if depth_image is not None:
        #         msg = Image()
        #         msg.header.stamp = self.get_clock().now().to_msg()
        #         msg.header.frame_id = "front_camera_depth"
        #         msg.height = depth_image.shape[0]
        #         msg.width = depth_image.shape[1]
        #         msg.encoding = "32FC1"
        #         msg.is_bigendian = 0
        #         msg.step = depth_image.shape[1] * 4
        #         msg.data = depth_image.astype(np.float32).tobytes()

        #         self.depth_pub.publish(msg)

        # try:
        #     data, _ = self.depth_sock.recvfrom(2000000)

        #     if len(data) < 8:
        #         raise ValueError("depth packet too small")

        #     h, w = struct.unpack("II", data[:8])
        #     depth_bytes = data[8:]

        #     expected_size = h * w * 4
        #     if len(depth_bytes) != expected_size:
        #         raise ValueError(
        #             f"depth payload size mismatch: got {len(depth_bytes)}, expected {expected_size}"
        #         )

        #     depth_image = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))

        #     msg = Image()
        #     msg.header.stamp = self.get_clock().now().to_msg()
        #     msg.header.frame_id = "front_camera_depth"
        #     msg.height = h
        #     msg.width = w
        #     msg.encoding = "32FC1"
        #     msg.is_bigendian = 0
        #     msg.step = w * 4
        #     msg.data = depth_image.tobytes()

        #     self.depth_pub.publish(msg)

        # except BlockingIOError:
        #     pass
        # except Exception as e:
        #     self.get_logger().error(f"depth receive error: {e}")

        try:
            if self.depth_conn is None:
                try:
                    self.depth_conn, _ = self.depth_server_sock.accept()
                    self.depth_conn.setblocking(True)
                    self.get_logger().info("Depth TCP client connected.")
                except BlockingIOError:
                    pass
            else:
                header = self.recv_exact(self.depth_conn, 4)
                payload_size = struct.unpack("I", header)[0]

                payload = self.recv_exact(self.depth_conn, payload_size)

                h, w = struct.unpack("II", payload[:8])
                depth_bytes = payload[8:]

                expected_size = h * w * 4
                if len(depth_bytes) != expected_size:
                    raise ValueError(
                        f"depth payload size mismatch: got {len(depth_bytes)}, expected {expected_size}"
                    )

                depth_image = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))

                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "front_camera_depth"
                msg.height = h
                msg.width = w
                msg.encoding = "32FC1"
                msg.is_bigendian = 0
                msg.step = w * 4
                msg.data = depth_image.tobytes()

                self.depth_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"depth receive error: {e}")
            if self.depth_conn is not None:
                try:
                    self.depth_conn.close()
                except Exception:
                    pass
                self.depth_conn = None

        try:
            data, _ = self.joint_state_sock.recvfrom(65535)

            msg_in = json.loads(data.decode("utf-8"))

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.header.frame_id = "base"

            js.name = msg_in.get("names", [])
            js.position = msg_in.get("position", [])
            js.velocity = msg_in.get("velocity", [])

            self.joint_state_pub.publish(js)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"joint_state receive error: {e}")

        try:
            data, _ = self.imu_sock.recvfrom(4096)

            msg_in = json.loads(data.decode("utf-8"))

            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.header.frame_id = "imu_link"

            imu.angular_velocity.x = float(msg_in.get("wx", 0.0))
            imu.angular_velocity.y = float(msg_in.get("wy", 0.0))
            imu.angular_velocity.z = float(msg_in.get("wz", 0.0))

            imu.linear_acceleration.x = float(msg_in.get("ax", 0.0))
            imu.linear_acceleration.y = float(msg_in.get("ay", 0.0))
            imu.linear_acceleration.z = float(msg_in.get("az", 0.0))

            self.imu_pub.publish(imu)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"imu receive error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()