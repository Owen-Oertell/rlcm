from omegaconf import DictConfig, OmegaConf
import hydra
import sys

from diffusers import DiffusionPipeline
from lcm_rl_pytorch.core.dataloader import SimpleAnimals
from lcm_rl_pytorch.diffusers_patch.lcm_scheduler import LCMScheduler
from lcm_rl_pytorch.rewards.aesthetic import AestheticScorer
import torch
from peft import LoraConfig

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from lcm_rl_pytorch.rewards import rewards
import wandb
import numpy as np

from lcm_rl_pytorch.core.training_loop import training_loop

from diffusers.loaders import AttnProcsLayers
import os
import cv2

import time
from PIL import Image
import io


logger = get_logger(__name__)


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

    set_seed(cfg.training.seed,device_specific=True)

    # load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        training_config.model,
        custom_pipeline="/share/cuvl/ojo2/cm/lcm_rl_pytorch/diffusers_patch",
    )
    pipeline.scheduler = LCMScheduler()
    pipeline.to(torch_device=accelerator.device, torch_dtype=torch.float16)

    def truncate(n):
        return int(n * 1000) / 1000
    
    def incompressibility():
        def _fn(images):
            if isinstance(images, torch.Tensor):
                images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            buffers = [io.BytesIO() for _ in images]
            for image, buffer in zip(images, buffers):
                image.save(buffer, format="JPEG", quality=95)
            sizes = [buffer.tell() / 1000 for buffer in buffers]
            return torch.tensor(sizes)

        return _fn
    
    # Prepare Dataset
    dataset = SimpleAnimals(4)

    # load optimizer, prepare unet

    # prepare dataloader, optimizer, unet
    #aesthetic score
    scorer = AestheticScorer(dtype=torch.float16).cuda()
    #incompressibility
    # jpeg_fn = incompressibility()

    #save inference results

    for time in range(1,8):
        with accelerator.autocast():
            sample_dict = pipeline(
                        prompt=["cow"]*4,
                        num_inference_steps=8,
                        guidance_scale=cfg.training.guidance_scale,
                        lcm_origin_steps=cfg.training.lcm_origin_steps,
                        output_type="pt",
                        inference_time = time,
                    )

        last_images = sample_dict["images"].to(dtype=torch.float32)

        if isinstance(last_images, torch.Tensor):
            images = (last_images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = last_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        # for aesthetic
        scores = scorer(images)
        # scores = jpeg_fn(images)
       
        cv2.imwrite("/share/cuvl/ojo2/cm/scripts/front_page_figure/"+"animals-infer{}-{}.png".format(str(time), truncate(scores[0])), (last_images[0]*255).cpu().numpy().transpose(1,2,0)[:,:,::-1].astype(np.uint8))


if __name__ == "__main__":
    main()