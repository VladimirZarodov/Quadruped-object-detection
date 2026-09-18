from __future__ import annotations

from datetime import datetime
import os

import torch


def _unwrap_env(env):
    while hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def get_base_pose_xy_yaw(env):
    base_env = _unwrap_env(env)
    x = base_env.root_states[0, 0].item()
    y = base_env.root_states[0, 1].item()
    quat = base_env.root_states[0, 3:7]
    yaw = torch.atan2(
        2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
        1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
    ).item()
    return x, y, yaw


class CompetitionRunLogger:
    
    def __init__(self, env, seed=0, log_root=None):
        from aliengo_gym import MINI_GYM_ROOT_DIR

        self.env = env
        self.base_env = _unwrap_env(env)
        self.seed = int(seed)
        self.detected_objects = {}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = log_root or os.path.join(MINI_GYM_ROOT_DIR, "logs")
        self.log_dir = os.path.join(root, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, f"log_seed_{self.seed}.txt")
        self.log_file = open(self.log_path, "w+")

        print(f"[LOG] Saving log to: {self.log_path}")
        self._write_header()

    def _write_header(self):
        sequence_of_objects = getattr(self.env, "SEQUENCE_OF_OBJECTS", None)
        object_positions = getattr(self.env, "detectable_object_positions", [])

        self.log_file.write(f"seed={self.seed}\n\n")
        self.log_file.write(f"SEQUENCE_OF_OBJECTS = {sequence_of_objects}\n")

        for obj in object_positions:
            self.log_file.write(
                f"object {obj['id']}: "
                f"cell=({obj['cell_x']}, {obj['cell_y']}), "
                f"world=({obj['x']:.2f}, {obj['y']:.2f})\n"
            )

        self.log_file.write("\n")
        self.log_file.write("detected_objects = {}\n")
        self.log_file.write("\nt,x,y,yaw\n")
        self.log_file.flush()

    def log_pose(self, t, x, y, yaw):
        self.log_file.write(f"{t:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n")
        self.log_file.flush()

    def log_step(self, t):
        x, y, yaw = get_base_pose_xy_yaw(self.base_env)
        self.log_pose(t, x, y, yaw)
        return x, y, yaw

    def log_detected_object(self, object_id, t, x, y, yaw):
        if object_id in self.detected_objects:
            return

        self.detected_objects[object_id] = {
            "t": round(t, 3),
            "x": round(x, 4),
            "y": round(y, 4),
            "yaw": round(yaw, 4),
        }

        self.log_file.seek(0)
        lines = self.log_file.readlines()

        new_block = "detected_objects = {\n"
        for key, value in self.detected_objects.items():
            new_block += f"{key}: {value},\n"
        new_block += "}\n"

        start, end = None, None
        for i, line in enumerate(lines):
            if line.startswith("detected_objects"):
                start = i
                if line.strip().endswith("}"):
                    end = i
                else:
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() == "}":
                            end = j
                            break
                break

        if start is None or end is None:
            return

        lines[start : end + 1] = [new_block]

        self.log_file.seek(0)
        self.log_file.writelines(lines)
        self.log_file.truncate()
        self.log_file.flush()

    def log_detected_object_at_time(self, object_id, t):
        x, y, yaw = get_base_pose_xy_yaw(self.base_env)
        self.log_detected_object(object_id, t, x, y, yaw)

    def close(self):
        if getattr(self, "log_file", None) is not None and not self.log_file.closed:
            self.log_file.close()
