from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState, ImuState, JointState, RobotState


@dataclass
class StepResult:
    observation: torch.Tensor
    reward: torch.Tensor | None
    done: torch.Tensor | None
    info: dict


class SimAliengoRobot(AliengoRobotInterface):
    CMD_VX = 0
    CMD_VY = 1
    CMD_VW = 2
    CMD_BODY_HEIGHT = 3
    CMD_GAIT_FREQUENCY = 4
    CMD_GAIT_PHASE = 5
    CMD_GAIT_OFFSET = 6
    CMD_GAIT_BOUND = 7
    CMD_GAIT_DURATION = 8
    CMD_FOOTSWING_HEIGHT = 9
    CMD_BODY_PITCH = 10
    CMD_BODY_ROLL = 11
    CMD_STANCE_WIDTH = 12
    CMD_STANCE_LENGTH = 13
    CMD_AUX_REWARD = 14

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self._speed = torch.zeros(3, device=self.env.device)
        self._command_template = None
        self._step_index = 0
        self._sim_time_s = 0.0
        self._prev_base_lin_vel = None
        self._latest_state = None
        self._last_result = StepResult(
            observation=self.env.get_observations(),
            reward=None,
            done=None,
            info={},
        )
        self._refresh_command_template()
        self._latest_state = self._extract_state(reset=True)

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value).copy()

    def _get_control_dt(self) -> float:
        base_env = self._unwrap_env()
        dt = getattr(base_env, "dt", None)
        try:
            dt_value = float(dt)
            if dt_value > 0.0:
                return dt_value
        except (TypeError, ValueError):
            pass
        return 0.02

    def _extract_camera_state(self) -> CameraState:
        camera = self.get_camera()
        if not isinstance(camera, dict):
            return CameraState(rgb=None, depth=None)
        rgb = camera.get("image")
        depth = camera.get("depth")
        return CameraState(
            rgb=None if rgb is None else np.asarray(rgb).copy(),
            depth=None if depth is None else np.asarray(depth).copy(),
        )

    def _extract_state(self, *, reset: bool) -> RobotState:
        base_env = self._unwrap_env()
        dt = self._get_control_dt()

        num_dof = int(base_env.dof_pos.shape[1])
        num_actuated_dof = int(getattr(base_env, "num_actuated_dof", num_dof))
        num_actuated_dof = max(0, min(num_actuated_dof, num_dof))

        dof_pos = base_env.dof_pos[0, :num_actuated_dof]
        dof_vel = base_env.dof_vel[0, :num_actuated_dof]

        default_dof_pos = getattr(base_env, "default_dof_pos", None)
        if torch.is_tensor(default_dof_pos):
            if default_dof_pos.ndim >= 2:
                default_dof_pos_ref = default_dof_pos[0, :num_actuated_dof]
            else:
                default_dof_pos_ref = default_dof_pos[:num_actuated_dof]
        else:
            default_dof_pos_ref = torch.zeros_like(dof_pos)

        # Participant-facing joint measurement: position error relative to the
        # nominal standing pose, without observation scaling.
        joint_positions = self._tensor_to_numpy(dof_pos - default_dof_pos_ref)
        joint_velocities = self._tensor_to_numpy(dof_vel)
        base_lin_vel = self._tensor_to_numpy(base_env.base_lin_vel[0])
        base_ang_vel = self._tensor_to_numpy(base_env.base_ang_vel[0])

        joint_names = tuple(getattr(base_env, "dof_names", ()))[:num_actuated_dof]
        camera_state = self._extract_camera_state()

        return RobotState(
            step_index=self._step_index,
            sim_time_s=self._sim_time_s,
            dt=dt,
            joints=JointState(
                names=joint_names,
                positions=joint_positions,
                velocities=joint_velocities,
            ),
            imu=ImuState(
                angular_velocity_xyz=base_ang_vel,
            ),
            base_linear_velocity_xyz=base_lin_vel,
            base_angular_velocity_xyz=base_ang_vel,
            camera=camera_state,
        )

    def _unwrap_env(self):
        env = self.env
        while hasattr(env, "env") and getattr(env, "env") is not env:
            env = env.env
        return env

    def _refresh_command_template(self) -> None:
        base_env = self._unwrap_env()
        template = None
        default_command = getattr(base_env, "default_command", None)
        if torch.is_tensor(default_command):
            template = default_command.detach().clone()
        elif hasattr(base_env, "commands") and torch.is_tensor(base_env.commands):
            template = base_env.commands[0].detach().clone()

        if template is None:
            self._command_template = None
            return

        # Keep trot command deterministic across runs. Only vx/vy/vw are
        # controlled externally by the participant controller; pitch is fixed.
        fixed_values = {
            self.CMD_BODY_HEIGHT: 0.0,
            self.CMD_GAIT_FREQUENCY: 3.0,
            self.CMD_GAIT_PHASE: 0.5,
            self.CMD_GAIT_OFFSET: 0.0,
            self.CMD_GAIT_BOUND: 0.0,
            self.CMD_GAIT_DURATION: 0.5,
            self.CMD_FOOTSWING_HEIGHT: 0.08,
            self.CMD_BODY_PITCH: 0.0,
            self.CMD_BODY_ROLL: 0.0,
            self.CMD_STANCE_WIDTH: 0.25,
            self.CMD_STANCE_LENGTH: 0.45,
            self.CMD_AUX_REWARD: 0.0,
        }
        for index, value in fixed_values.items():
            if template.shape[0] > index:
                template[index] = float(value)

        self._command_template = template

    def _apply_command(self) -> None:
        if hasattr(self.env, "set_command"):
            self.env.set_command(
                float(self._speed[0].item()),
                float(self._speed[1].item()),
                float(self._speed[2].item()),
                0.0,
            )
            return

        base_env = self._unwrap_env()
        if self._command_template is None:
            self._refresh_command_template()
        if self._command_template is None or not hasattr(base_env, "commands"):
            raise AttributeError("Environment does not expose a controllable command interface.")

        command = self._command_template.clone()
        command[self.CMD_VX] = float(self._speed[0].item())
        command[self.CMD_VY] = float(self._speed[1].item())
        command[self.CMD_VW] = float(self._speed[2].item())
        if command.shape[0] > self.CMD_BODY_PITCH:
            command[self.CMD_BODY_PITCH] = 0.0
        base_env.commands[:] = command.unsqueeze(0).repeat(base_env.num_envs, 1)

    def set_speed(self, vx: float, vy: float, vw: float) -> None:
        self._speed = torch.tensor([vx, vy, vw], device=self.env.device, dtype=torch.float32)
        self._apply_command()

    def stop(self) -> None:
        self._speed.zero_()
        self._apply_command()

    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, privileged_obs = result
            info = {"privileged_obs": privileged_obs}
        else:
            obs = result
            info = {}
            if isinstance(obs, dict) and "privileged_obs" in obs:
                info["privileged_obs"] = obs["privileged_obs"]
        self._last_result = StepResult(observation=obs, reward=None, done=None, info=info)
        self._refresh_command_template()
        self._apply_command()
        self._step_index = 0
        self._sim_time_s = 0.0
        self._latest_state = self._extract_state(reset=True)
        return obs

    def step(self):
        obs = self.env.get_observations()
        policy_input = obs.detach() if hasattr(obs, "detach") else obs
        action = self.policy(policy_input)
        env_action = action.detach() if hasattr(action, "detach") else action
        result = self.env.step(env_action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, privileged_obs, reward, done, info = result
            info["privileged_obs"] = privileged_obs
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            if isinstance(obs, dict) and "privileged_obs" in obs:
                info["privileged_obs"] = obs["privileged_obs"]
        else:
            raise ValueError("Unexpected environment step() return format.")
        self._last_result = StepResult(observation=obs, reward=reward, done=done, info=info)
        self._step_index += 1
        self._sim_time_s += self._get_control_dt()
        self._latest_state = self._extract_state(reset=False)
        return obs, reward, done, info

    def get_camera(self):
        base_env = self._unwrap_env()
        camera_getter = getattr(base_env, "get_front_camera_data", None)
        if callable(camera_getter):
            return camera_getter(env_id=0)
        return None

    def get_observation(self):
        return self._last_result.observation

    def get_state(self) -> RobotState:
        if self._latest_state is None:
            self._latest_state = self._extract_state(reset=True)
        return self._latest_state

    def is_fallen(self) -> bool:
        if self._last_result.done is None:
            return False
        return bool(torch.any(self._last_result.done).item())
