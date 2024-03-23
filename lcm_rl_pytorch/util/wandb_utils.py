import tempfile
from PIL import Image
import wandb
import os
import numpy as np


def log_images_to_wandb(accelerator, prompts, images, rewards, step):
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, image in enumerate(images):
            pil = Image.fromarray(
                (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((256, 256))
            pil.save(os.path.join(tmpdir, f"{i}.jpg"))
        accelerator.log(
            {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{i}.jpg"),
                        caption=f"{prompt:.25} | {reward:.2f}",
                    )
                    for i, (prompt, reward) in enumerate(
                        zip(prompts, rewards)
                    )  # only log rewards from process 0
                ],
            },
            step=step,
        )
