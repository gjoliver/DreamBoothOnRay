import itertools
import math
from os import path

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from ray.air import session, ScalingConfig
from ray.train.torch import TorchTrainer
import torch
from transformers import CLIPTextModel

from data import get_train_dataset
from flags import train_arguments
from utils import get_weight_dtype, set_environ_vars


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


def train_fn(config):
    args = config["args"]
    set_environ_vars()

    accelerator = Accelerator(
        logging_dir=path.join(session.get_trial_dir(), "accelerator_logs"),
    )

    # Load models
    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae")
    # We are not training VAE part of the model.
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet")
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(
        itertools.chain(unet.parameters(), text_encoder.parameters()),
        lr=args.lr,
    )

    train_dataset = session.get_dataset_shard("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset.to_torch(),
        batch_size=args.train_batch_size,
    )

    # Prepare everything with `accelerator`.
    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader
    )

    # Move vae to device and cast weights to half-precision.
    # VAE is only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=get_weight_dtype(accelerator))

    # Train!
    num_update_steps_per_epoch = train_dataset.count()
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes

    print(f"Running {num_train_epochs} epochs. Max training steps {args.max_train_steps}.")

    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                optimizer.zero_grad()

                # Convert images to latent space
                latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
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
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = prior_preserving_loss(model_pred, target, args.prior_loss_weight)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1

            results = {
                "step": global_step,
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            session.report(logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_dir,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
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
        scaling_config=ScalingConfig(use_gpu=True, num_workers=1),
        datasets={
            "train": train_dataset,
        },
    )
    result = trainer.fit()

    print(result)