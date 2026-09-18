import socket
import json
import time
import cv2
import struct
import numpy as np


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

class SimBridgeClient:
    def __init__(self):
        self.latest_cmd = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind((CMD_IP, CMD_PORT))
        self.cmd_sock.setblocking(False)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        self.depth_sock.connect((DEPTH_IP, DEPTH_PORT)) # TCP
        self.joint_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def receive_cmd(self):
        try:
            data, _ = self.cmd_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            self.latest_cmd["vx"] = float(msg.get("vx", 0.0))
            self.latest_cmd["vy"] = float(msg.get("vy", 0.0))
            self.latest_cmd["wz"] = float(msg.get("wz", 0.0))
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"receive_cmd error: {e}")

        return self.latest_cmd.copy()

    def send_state(self, vx, vy, wz):
        msg = {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.state_sock.sendto(data, (STATE_IP, STATE_PORT))

    def send_rgb(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(".jpg", bgr)
        # success, encoded = cv2.imencode(".jpg", rgb)
        if not success:
            print("send_rgb error: JPEG encoding failed")
            return

        data = encoded.tobytes()
        self.rgb_sock.sendto(data, (RGB_IP, RGB_PORT))

    # UDP with PNG encoding
    # def send_depth(self, depth):
    #     depth_clipped = depth.copy()
    #     depth_clipped = depth_clipped.astype("float32")

    #     success, encoded = cv2.imencode(".png", depth_clipped)
    #     if not success:
    #         print("send_depth error: PNG encoding failed")
    #         return

    #     data = encoded.tobytes()
    #     self.depth_sock.sendto(data, (DEPTH_IP, DEPTH_PORT))

    # UDP without PNG encoding
    # def send_depth(self, depth):
    #     depth = np.asarray(depth, dtype=np.float32)

    #     h, w = depth.shape[:2]
    #     header = struct.pack("II", h, w)
    #     payload = header + depth.tobytes()

    #     self.depth_sock.sendto(payload, (DEPTH_IP, DEPTH_PORT))

    # TCP
    def send_depth(self, depth):
        depth = np.asarray(depth, dtype=np.float32)

        h, w = depth.shape[:2]
        payload = struct.pack("II", h, w) + depth.tobytes()
        packet = struct.pack("I", len(payload)) + payload

        self.depth_sock.sendall(packet)

    def send_joint_states(self, names, position, velocity):
        msg = {
            "names": list(names),
            "position": [float(x) for x in position],
            "velocity": [float(x) for x in velocity],
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.joint_state_sock.sendto(data, (JOINT_STATE_IP, JOINT_STATE_PORT))

    def send_imu(self, ang_vel, lin_acc):
        msg = {
            "wx": float(ang_vel[0]),
            "wy": float(ang_vel[1]),
            "wz": float(ang_vel[2]),
            "ax": float(lin_acc[0]),
            "ay": float(lin_acc[1]),
            "az": float(lin_acc[2]),
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.imu_sock.sendto(data, (IMU_IP, IMU_PORT))
