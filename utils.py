import os

from ray.air import session
import torch


def set_environ_vars():
    # Env vars necessary for HF to setup DDP
    os.environ["RANK"] = str(session.get_world_rank())
    os.environ["WORLD_SIZE"] = str(session.get_world_size())
    os.environ["LOCAL_RANK"] = str(session.get_local_rank())

    # FSDP env vars
    os.environ["USE_FSDP"] = "true"
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "true"
    os.environ["FSDP_MIN_NUM_PARAMS"] = str(1e8)
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "SIZE_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"


def get_weight_dtype(accelerator):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
