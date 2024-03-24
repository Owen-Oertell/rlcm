from setuptools import setup, find_packages

setup(
    name="lcm-rl-pytorch",
    version="0.0.1",
    packages=["lcm_rl_pytorch"],
    python_requires=">=3.10",
    install_requires=[
        "accelerate==0.24.0",
        "hydra-core==1.3.2",
        "torch==2.1.0",
        "wandb",
        "diffusers==0.23.1",
        "peft==0.6.2",
        "numpy==1.26.1",
        "tqdm",
        "transformers==4.36.1",
        "inflect==7.0.0"
    ],
)
