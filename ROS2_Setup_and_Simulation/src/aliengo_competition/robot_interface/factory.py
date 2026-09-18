from __future__ import annotations

from aliengo_competition.robot_interface.sim import SimAliengoRobot
from scripts.play import DEFAULT_RUN_LABEL, load_env

DEFAULT_CAMERA_DEPTH_MAX_M = 4.0

def make_robot_interface(
    *,
    headless: bool = True,
    run_label: str = DEFAULT_RUN_LABEL,
    seed: int = 0,
):
    print(f"Loading controller low-level policy from label: {run_label}")
    env, policy = load_env(run_label, headless=headless, seed=int(seed))
    return SimAliengoRobot(env=env, policy=policy)
