import os

from ray.air import session
import torch


def set_environ_vars():
    # Env vars necessary for HF to setup DDP
    os.environ["RANK"] = str(session.get_world_rank())
    os.environ["WORLD_SIZE"] = str(session.get_world_size())
    os.environ["LOCAL_RANK"] = str(session.get_local_rank())
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )

    # DeepSpeed env vars
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = "2"
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = "1"
    os.environ["ACCELERATE_GRADIENT_CLIPPING"] = "1.0"
    os.environ["ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"] = "cpu"
    os.environ["ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE"] = "cpu"
