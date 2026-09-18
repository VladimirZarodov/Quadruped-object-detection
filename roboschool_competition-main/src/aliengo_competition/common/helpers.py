from __future__ import annotations

import copy
import os
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
from isaacgym import gymapi, gymutil
import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, list):
            result[key] = [class_to_dict(item) for item in value]
        else:
            result[key] = class_to_dict(value)
    return result


def update_class_from_dict(obj, values: dict) -> None:
    for key, value in values.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, value)
        else:
            setattr(obj, key, value)


def set_seed(seed: int) -> None:
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg: dict) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    return sim_params


def get_load_path(root: str, load_run: str | int = -1, checkpoint: int = -1) -> str:
    runs = sorted([run for run in os.listdir(root) if run != "exported"])
    if not runs:
        raise ValueError(f"No runs in this directory: {root}")
    latest = os.path.join(root, runs[-1])
    run_dir = latest if load_run in (-1, None) else os.path.join(root, str(load_run))
    if checkpoint in (-1, None):
        models = sorted([file for file in os.listdir(run_dir) if file.startswith("model_") and file.endswith(".pt")])
        if not models:
            raise ValueError(f"No checkpoints in run directory: {run_dir}")
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"
    return os.path.join(run_dir, model)


def update_cfg_from_args(env_cfg, cfg_train, args):
    if env_cfg is not None and args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = True
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        load_run = getattr(args, "load_run", None)
        if load_run is not None:
            cfg_train.runner.load_run = load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
    return env_cfg, cfg_train


def export_policy_as_jit(actor_critic, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced = torch.jit.script(model)
    traced.save(os.path.join(path, "policy.pt"))


def namespace(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
