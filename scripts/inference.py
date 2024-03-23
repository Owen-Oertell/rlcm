from diffusers import DiffusionPipeline
from lcm_rl_pytorch.diffusers_patch.lcm_scheduler import LCMScheduler
import torch
from accelerate import Accelerator
from peft import LoraConfig

accelerator = Accelerator(
    mixed_precision="fp16",
)

import torch
from safetensors.torch import load_file

pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        custom_pipeline="/share/cuvl/ojo2/cm/lcm_rl_pytorch/diffusers_patch",
    )
pipe.scheduler = LCMScheduler()
pipe.unet.requires_grad_(False)

pipe.to(torch_device=accelerator.device, torch_dtype=torch.float16)

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.

# set unet state dict
# state_dict = load_file("/home/ojo2/jonathan_checkpoints/logs/horizon_8_aesthetic_lora_jc/checkpoints/checkpoint_19/diffusion_pytorch_model.safetensors")
# state_dict = load_file("/home/ojo2/jonathan_checkpoints/logs/horizon_8_aesthetic_lora_jc/checkpoints/checkpoint_19/diffusion_pytorch_model.safetensors")
# state_dict = load_file("/home/ojo2/aeesthetic_checkpoints/horizon_/checkpoints/checkpoint_1/diffusion_pytorch_model.safetensors")
state_dict = load_file("/share/cuvl/ojo2/cm/scripts/logs/pia_rlcm_2024-02-27_10-45-51/checkpoints/checkpoint_22/diffusion_pytorch_model.safetensors")
# state_dict = load_file("/share/cuvl/ojo2/cm/scripts/logs/compression_hparams_testing_new_lora_2024-02-22_21-02-39/checkpoints/checkpoint_15/diffusion_pytorch_model.safetensors")
# state_dict = load_file("/home/ojo2/incompression_main_result_2024-03-08_21-37-25/checkpoints/checkpoint_0/diffusion_pytorch_model.safetensors")
# state_dict = load_file("/share/cuvl/ojo2/cm/scripts/logs/pia_rlcm_2024-02-28_23-59-35/checkpoints/checkpoint_7/diffusion_pytorch_model.safetensors")
unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
pipe.unet.add_adapter(unet_lora_config)
pipe.unet.load_state_dict(state_dict)

# prompts = ["a dog riding a bike", "a fox playing chess", "a horse washing the dishes", "a hedgehog riding a bike", "a raccoon washing the dishes", "a sheep playing chess", "a tiger playing chess", "a cat washing the dishes"]
# prompts = ["bike", "fridge", "tractor", "waterfall"]

from lcm_rl_pytorch.core.dataloader import AnimalsWithActions
# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 16
# unet = accelerator.prepare(pipe.unet)

import numpy as np
from tqdm import tqdm
import wandb

all_images = None
for i in tqdm(range(8)):
    prompts = AnimalsWithActions(8).sample()

    with accelerator.autocast():
        opt = pipe(prompt=prompts, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pt")
    image_list = []
    if all_images is None:
        all_images = opt["images"]
    else:
        all_images = torch.cat([all_images, opt["images"]], dim=0)

# display all images
from PIL import Image
import matplotlib.pyplot as plt

all_images = torch.tensor(all_images)
if isinstance(all_images, torch.Tensor):
    all_images = (all_images * 255).round().clamp(0, 255).to(torch.uint8)
else:
    all_images = all_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    all_images = torch.tensor(all_images, dtype=torch.uint8)

plt.figure(figsize=(8,8)) # specifying the overall grid size
for i in range(64):
    plt.subplot(8,8,i+1)    # the number of images in the grid is 5*5 (25)
    
    # scores = jpeg_fn(images)
    
    img = (all_images[i]).cpu().numpy().transpose(1,2,0).astype(np.uint8)
    # convert tensor to PIL
    img = Image.fromarray(img) 

    plt.imshow(img)
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("inference_result.pdf", bbox_inches='tight', pad_inches=0, dpi=300)