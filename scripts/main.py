from omegaconf import DictConfig, OmegaConf
import hydra

from diffusers import DiffusionPipeline
from lcm_rl_pytorch.diffusers_patch.lcm_scheduler import LCMScheduler
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from lcm_rl_pytorch.rewards import rewards
import wandb
import numpy as np
from peft import LoraConfig

from lcm_rl_pytorch.util.lora import cast_training_params
from lcm_rl_pytorch.core.training_loop import training_loop
from lcm_rl_pytorch.core import dataloader
logger = get_logger(__name__)

import os
from accelerate.utils import set_seed, ProjectConfiguration


@hydra.main(
    version_base=None, config_path="../lcm_rl_pytorch/conf", config_name="config"
)
def main(cfg: DictConfig) -> None:
    training_config = cfg.get(cfg.task, None)
    if training_config is None:
        raise ValueError(f"Invalid task name: {cfg.task}")

    # Checpoint naming
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.training.logdir, cfg.wandb.name),
        automatic_checkpoint_naming=True,
        total_limit=99999,
    )


    # load accelerator
    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_config,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps
        * training_config.num_inference_steps,
        mixed_precision="fp16",
    )

    # set_seed(cfg.training.seed, device_specific=True)

    # init wandb
    if cfg.wandb.log and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            init_kwargs={"wandb": {"name": cfg.wandb.name, "entity": cfg.wandb.entity}},
        )

    # load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        training_config.model,
        custom_pipeline="./../lcm_rl_pytorch/diffusers_patch",
    )
    pipeline.scheduler = LCMScheduler()
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.to(torch_device=accelerator.device, torch_dtype=torch.float16)
    if cfg.training.use_lora:
        # for all tasks but aesthetic, we use r=16, alpha =32. For aesthetic, we use r=8, alpha=8
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet.add_adapter(unet_lora_config)

        cast_training_params(pipeline.unet, torch.float32)
    else:
        print("Please use lora.")
        exit()
    # load reward function
    reward_fn = getattr(rewards, training_config.reward_fn)(cfg)
    dataset = getattr(dataloader, training_config.dataset)(training_config.sample_batch_size_per_gpu)

    # define save model hook:
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        accelerator.unwrap_model(pipeline.unet).save_pretrained(output_dir)
        weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    # start training loop
    training_loop(accelerator, cfg, training_config, pipeline, reward_fn, dataset)


if __name__ == "__main__":
    main()
