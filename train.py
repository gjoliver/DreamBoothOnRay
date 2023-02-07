"""
python train.py   --model_dir=/mnt/shared_storage/jungong/dream_booth/models/models--CompVis--stable-diffusion-v1-4/snapshots/3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4/   --output_dir=/mnt/shared_storage/jungong/dream_booth/outputs/attempt_4/   --instance_images_dir=/mnt/shared_storage/jungong/dream_booth/images/sks/   --instance_prompt="a photo of sks lego car"   --class_images_dir=/mnt/shared_storage/jungong/dream_booth/images/lego/   --class_prompt="a photo of a lego car"
"""

import itertools
import math
from os import path

from accelerate import Accelerator
import bitsandbytes as bnb
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from ray.air import session, ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel

from dataset import collate, get_train_dataset
from flags import train_arguments
from utils import set_environ_vars


def prior_preserving_loss(model_pred, target, weight):
    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    # Compute instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    # Compute prior loss
    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

    # Add the prior loss to the instance loss.
    return loss + weight * prior_loss


def get_target(scheduler, noise):
    """Get the target for loss depending on the prediction type.
    """
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {pred_type}")


def train_fn(config):
    args = config["args"]
    set_environ_vars()

    accelerator = Accelerator(
        mixed_precision='bf16',  # Use bf16 to save GRAM.
    )

    # Load models
    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae")
    # We are not training VAE part of the model.
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet")
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # 8bit Adam to save GRAM.
    optimizer = torch.optim.Adagrad(unet.parameters(), lr=args.lr)  #bnb.optim.AdamW8bit

    train_dataset = session.get_dataset_shard("train")

    # Prepare everything with `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Use bf16 dtype to save GRAM.
    weight_dtype = torch.bfloat16

    # Move vae to device and cast weights to half-precision.
    # VAE is only used for inference, keeping weights in full precision is not required.
    text_encoder.to("cuda:1", dtype=weight_dtype)
    vae.to("cuda:1", dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("dream_booth", config=vars(args))

    # Train!
    num_update_steps_per_epoch = train_dataset.count()
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes

    print(f"Running {num_train_epochs} epochs. Max training steps {args.max_train_steps}.")

    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()

        for step, batch in enumerate(
            train_dataset.iter_torch_batches(
                batch_size=args.train_batch_size,
                device=accelerator.device,
            )
        ):
            batch = collate(batch)
            optimizer.zero_grad()

            # Convert images to latent space
            latents = vae.encode(
                batch["images"].to("cuda:1", dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["prompt_ids"].to("cuda:1"))[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = get_target(noise_scheduler, noise)

            loss = prior_preserving_loss(model_pred.to("cuda:1"), target, args.prior_loss_weight)
            accelerator.backward(loss)

            # Gradient clipping before optimizer stepping.
            accelerator.clip_grad_norm_(unet.parameters(), rgs.max_grad_norm)

            optimizer.step()

            global_step += 1
            results = {
                "step": global_step,
                "loss": loss.detach().item(),
            }
            session.report(results)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Create pipeline using the trained modules and save it.
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_dir,
            unet=accelerator.unwrap_model(unet),
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = train_arguments().parse_args()

    # Build training dataset.
    train_dataset = get_train_dataset(args)

    print(f"Loaded training dataset (size: {train_dataset.count()})")

    # Train with Ray AIR TorchTrainer.
    trainer = TorchTrainer(
        train_fn,
        train_loop_config={
            "args": args
        },
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=1,
            resources_per_worker={
                "GPU": 4,
            }
        ),
        datasets={
            "train": train_dataset,
        },
    )
    result = trainer.fit()

    print(result)
