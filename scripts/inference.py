from diffusers import DiffusionPipeline
from lcm_rl_pytorch.diffusers_patch.lcm_scheduler import LCMScheduler
import torch
from accelerate import Accelerator
from peft import LoraConfig
import torch
from safetensors.torch import load_file
from lcm_rl_pytorch.core.dataloader import AnimalsWithActions
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
import matplotlib.pyplot as plt

accelerator = Accelerator(
    mixed_precision="fp16",
)

pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        custom_pipeline="./../lcm_rl_pytorch/diffusers_patch",
)
pipe.scheduler = LCMScheduler()
pipe.unet.requires_grad_(False)

pipe.to(torch_device=accelerator.device, torch_dtype=torch.float16)

# set unet state dict
state_dict = load_file("PATH_TO_SAFE_TENSOR_FILE")

unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
pipe.unet.add_adapter(unet_lora_config)
pipe.unet.load_state_dict(state_dict)

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 16

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
all_images = torch.tensor(all_images)
if isinstance(all_images, torch.Tensor):
    all_images = (all_images * 255).round().clamp(0, 255).to(torch.uint8)
else:
    all_images = all_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    all_images = torch.tensor(all_images, dtype=torch.uint8)

plt.figure(figsize=(8,8)) # specifying the overall grid size
for i in range(64):
    plt.subplot(8,8,i+1)
        
    img = (all_images[i]).cpu().numpy().transpose(1,2,0).astype(np.uint8)
    # convert tensor to PIL
    img = Image.fromarray(img) 

    plt.imshow(img)
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("inference_result.pdf", bbox_inches='tight', pad_inches=0, dpi=300)