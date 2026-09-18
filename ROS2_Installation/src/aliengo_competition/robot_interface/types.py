from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0


@dataclass
class JointState:
    names: Tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def name(self) -> Tuple[str, ...]:
        return self.names

    @property
    def position(self) -> np.ndarray:
        return self.positions

    @property
    def velocity(self) -> np.ndarray:
        return self.velocities


@dataclass
class ImuState:
    angular_velocity_xyz: np.ndarray

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.angular_velocity_xyz

    @property
    def wx(self) -> float:
        return float(self.angular_velocity_xyz[0])

    @property
    def wy(self) -> float:
        return float(self.angular_velocity_xyz[1])

    @property
    def wz(self) -> float:
        return float(self.angular_velocity_xyz[2])


@dataclass
class CameraState:
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]

    @property
    def image(self) -> Optional[np.ndarray]:
        return self.rgb


@dataclass
class RobotState:
    step_index: int
    sim_time_s: float
    dt: float
    joints: JointState
    imu: ImuState
    base_linear_velocity_xyz: np.ndarray
    base_angular_velocity_xyz: np.ndarray
    camera: CameraState

    @property
    def q(self) -> np.ndarray:
        return self.joints.positions

    @property
    def q_dot(self) -> np.ndarray:
        return self.joints.velocities

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return self.joints.names

    @property
    def linear_velocity_xyz(self) -> np.ndarray:
        return self.base_linear_velocity_xyz

    @property
    def joint_position(self) -> np.ndarray:
        return self.joints.position

    @property
    def joint_velocity(self) -> np.ndarray:
        return self.joints.velocity

    @property
    def base_velocity_xyz(self) -> np.ndarray:
        return self.base_linear_velocity_xyz

    @property
    def vx(self) -> float:
        return float(self.base_linear_velocity_xyz[0])

    @property
    def vy(self) -> float:
        return float(self.base_linear_velocity_xyz[1])

    @property
    def wz(self) -> float:
        return float(self.base_angular_velocity_xyz[2])
