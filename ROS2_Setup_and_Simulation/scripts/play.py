import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2

import glob
import pickle as pkl
from pathlib import Path

from aliengo_gym.envs import *
from aliengo_gym.envs.base.legged_robot_config import Cfg
from aliengo_gym.envs.aliengo.aliengo_config import config_aliengo
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
from datetime import datetime
import os

DEFAULT_RUN_LABEL = "gait-conditioned-agility/aliengo-v0/train"
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False, seed=0):
    dirs = glob.glob(str(RUNS_DIR / label / "*"))
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.terrain_length = 10.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    Cfg.env.episode_length_s = 300

    Cfg.env.front_camera_enabled = True
    Cfg.env.front_camera_attach_body_name = "trunk"
    Cfg.env.front_camera_color_width_px = 640
    Cfg.env.front_camera_color_height_px = 360
    Cfg.env.front_camera_depth_width_px = 848
    Cfg.env.front_camera_depth_height_px = 480
    Cfg.env.front_camera_color_fov_h_deg = 70.0
    Cfg.env.front_camera_depth_fov_h_deg = 86.0
    Cfg.env.front_camera_offset_xyz = [0.315, 0.0, 0.052]
    Cfg.env.front_camera_pitch_deg = -4.0

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(seed=seed, sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)

    return env, policy


def play_aliengo(headless=True):
    from pathlib import Path
    from aliengo_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = DEFAULT_RUN_LABEL
    seed = 0

    env, policy = load_env(label, headless=headless, seed=seed)
    SEQUENCE_OF_OBJECTS = env.SEQUENCE_OF_OBJECTS

    num_eval_steps = 1000000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    command_change_interval = 100
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(MINI_GYM_ROOT_DIR, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"log_seed_{seed}.txt")
    log_file = open(log_path, "w+")

    print(f"[LOG] Saving log to: {log_path}")

    object_positions = env.detectable_object_positions
    
    log_file.write(f"seed={seed}\n\n")
    log_file.write(f"SEQUENCE_OF_OBJECTS = {SEQUENCE_OF_OBJECTS}\n")

    for obj in object_positions:
        log_file.write(
            f"object {obj['id']}: "
            f"cell=({obj['cell_x']}, {obj['cell_y']}), "
            f"world=({obj['x']:.2f}, {obj['y']:.2f})\n"
        )

    log_file.write("\n")
    log_file.write("detected_objects = {}\n")
    log_file.write("\nt,x,y,yaw\n")
    log_file.flush()

    detected_objects = {}

    def log_detected_object(object_id):
        nonlocal detected_objects, log_file, t, x, y, yaw

        if object_id in detected_objects:
            return

        detected_objects[object_id] = {
            "t": round(t, 3),
            "x": round(x, 4),
            "y": round(y, 4),
            "yaw": round(yaw, 4),
        }

        log_file.seek(0)
        lines = log_file.readlines()

        new_block = "detected_objects = {\n"
        for k, v in detected_objects.items():
            new_block += f"{k}: {v},\n"
        new_block += "}\n"

        start, end = None, None

        for i, line in enumerate(lines):
            if line.startswith("detected_objects"):
                start = i
                if line.strip().endswith("}"):
                    end = i
                else:
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() == "}":
                            end = j
                            break
                break

        lines[start:end+1] = [new_block]

        log_file.seek(0)
        log_file.writelines(lines)
        log_file.truncate()
        log_file.flush()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)

        if i % command_change_interval == 0:
            x_vel_cmd = np.random.uniform(-0.5, 2.0)
            y_vel_cmd = np.random.uniform(-0.5, 0.5)
            yaw_vel_cmd = np.random.uniform(-1.0, 1.0)

        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        t = i * env.dt

        x = env.root_states[0, 0].item()
        y = env.root_states[0, 1].item()
        quat = env.root_states[0, 3:7]

        yaw = torch.atan2(
            2.0 * (quat[3]*quat[2] + quat[0]*quat[1]),
            1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        ).item()

        log_file.write(f"{t:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n")
        log_file.flush()

        # IMPORTANT! Add each detected object here:
        # if YOUR_CONDITION:
        #     log_detected_object(DETECTED_OBJECT_ID)

        camera_data = env.get_front_camera_data(0)
        if camera_data is not None:
            rgb = camera_data["image"]
            depth = camera_data["depth"]

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            depth_vis = depth.copy()
            depth_vis = np.clip(depth_vis, 0.0, 5.0)
            depth_vis = (255.0 * depth_vis / 5.0).astype(np.uint8)

            cv2.imshow("Front RGB", rgb_bgr)
            cv2.imshow("Front Depth", depth_vis)
            cv2.waitKey(1)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu().numpy()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_aliengo(headless=False)
