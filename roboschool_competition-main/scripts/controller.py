from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from isaacgym import gymutil

from aliengo_competition.controllers.main_controller import run
from aliengo_competition.robot_interface.factory import (
    DEFAULT_CAMERA_DEPTH_MAX_M,
    make_robot_interface,
)


def get_controller_args():
    custom_parameters = [
        {"name": "--steps", "type": int, "default": 15000, "help": "Number of controller steps. 15000 steps ~= 5 minutes at 20 ms control dt."},
        {"name": "--seed", "type": int, "help": "Random seed."},
        {"name": "--no_render_camera", "action": "store_true", "help": "Disable front RGB+Depth camera rendering."},
    ]
    args = gymutil.parse_arguments(
        description="AlienGo competition controller",
        headless=True,
        custom_parameters=custom_parameters,
    )
    args.render_camera = not getattr(args, "no_render_camera", False)
    return args


def controller(args):
    seed = 0 if getattr(args, "seed", None) is None else int(args.seed)
    robot = make_robot_interface(
        headless=args.headless,
        seed=seed,
    )
    run(
        robot,
        steps=args.steps,
        render_camera=args.render_camera,
        camera_depth_max_m=DEFAULT_CAMERA_DEPTH_MAX_M,
        seed=seed,
    )


if __name__ == "__main__":
    controller(get_controller_args())
