from random import shuffle
from accelerate.logging import get_logger
from functools import partial
import tqdm
import torch
import pdb
from lcm_rl_pytorch.util.wandb_utils import log_images_to_wandb
from lcm_rl_pytorch.util.model import get_model_prediction
from lcm_rl_pytorch.util.stat_tracking import PerPromptStatTracker
from collections import defaultdict
from concurrent import futures
import numpy as np

logger = get_logger(__name__)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def rebel_training_loop(accelerator, cfg, training_config, pipeline, reward_fn, dataset):
    logging_step = 0
    stat_tracker = PerPromptStatTracker(
        training_config.stat_buffer_size, training_config.stat_min_count
    )
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # load optimizer, prepare unet
    optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=training_config.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=training_config.num_epochs * 3,
    )

    # prepare dataloader, optimizer, unet
    optimizer, pipeline.unet = accelerator.prepare(optimizer, pipeline.unet)
    # training loop
    logger.info("::Starting Training::")

    for epoch in range(training_config.num_epochs):
        pipeline.unet.eval()

        scheduler.step()

        for iter in tqdm(
            range(training_config.batches_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        ):
            samples1 = []
            samples2 = []
            last_images = None
            # =================== SAMPLING ===================
            for _ in range(training_config.num_sample_iters):
                # sample from dataset
                prompts = dataset.sample()

                # SAMPLE IMAGES FIRST TIME ======
                # sample images, first time
                sample_dict_1 = pipeline(
                    prompt=prompts,
                    num_inference_steps=training_config.num_inference_steps,
                    guidance_scale=cfg.training.guidance_scale,
                    lcm_origin_steps=cfg.training.lcm_origin_steps,
                    output_type="pt",
                )
                # obtain reward
                rewards = executor.submit(reward_fn, sample_dict_1["images"], prompts)

                # add rewards to sample dict
                sample_dict_1["rewards"] = rewards
                sample_dict_1["encoded_prompts"] = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.training.max_prompt_length,
                ).input_ids

                last_images = sample_dict_1["images"]

                # SAMPLE IMAGES AGAIN ======
                # sample images, second time
                sample_dict_2 = pipeline(
                    prompt=prompts,
                    num_inference_steps=training_config.num_inference_steps,
                    guidance_scale=cfg.training.guidance_scale,
                    lcm_origin_steps=cfg.training.lcm_origin_steps,
                    output_type="pt",
                )

                # obtain reward
                rewards = executor.submit(reward_fn, sample_dict_2["images"], prompts)

                # add rewards to sample dict
                sample_dict_2["rewards"] = rewards
                sample_dict_2["encoded_prompts"] = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.training.max_prompt_length,
                ).input_ids

                samples1.append(sample_dict_1)
                samples2.append(sample_dict_2)
            # =================== BATCHING (and Logging and advantage computation) ===================
            for sample in tqdm(
                samples1 + samples2,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                rewards = sample["rewards"].result()
                sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

            # mostly copied from kevins repo
            # create super batch
            samples1 = {
                k: torch.cat([s[k] for s in samples1]) for k in samples1[0].keys()
            }
            samples2 = {
                k: torch.cat([s[k] for s in samples2]) for k in samples2[0].keys()
            }

            if not isinstance(samples1["rewards"], torch.Tensor):
                samples1["rewards"] = torch.tensor(samples1["rewards"])
            gathered_rewards1 = (
                accelerator.gather(samples1["rewards"].to(accelerator.device))
                .cpu()
                .numpy()
            )
            if cfg.wandb.log:
                log_images_to_wandb(
                    accelerator=accelerator,
                    prompts=prompts,
                    images=last_images,
                    rewards=rewards,
                    step=logging_step,
                )

                accelerator.log(
                    {
                        "rewards": gathered_rewards1,
                        "mean_reward": gathered_rewards1.mean(),
                        "std_reward": gathered_rewards1.std(),
                        "epoch": epoch,
                    },
                    step=logging_step,
                )

            # compute advantages with per prompt stat tracking
            aggregated_prompts = []
            for samples in [samples1, samples2]:
                if not isinstance(samples["encoded_prompts"], torch.Tensor):
                    samples["encoded_prompts"] = torch.tensor(
                        samples["encoded_prompts"]
                    )
                prompt_ids = (
                    accelerator.gather(
                        samples["encoded_prompts"].to(accelerator.device)
                    )
                    .cpu()
                    .numpy()
                )

                prompts = pipeline.tokenizer.batch_decode(
                    prompt_ids, skip_special_tokens=True
                )
                aggregated_prompts.extend(prompts)

            if not isinstance(samples2["rewards"], torch.Tensor):
                samples2["rewards"] = torch.tensor(samples2["rewards"])
            gathered_rewards2 = (
                accelerator.gather(samples2["rewards"].to(accelerator.device))
                .cpu()
                .numpy()
            )

            aggregated_rewards = np.concatenate(
                [gathered_rewards1, gathered_rewards2]
            )

            advantages = stat_tracker.update(aggregated_prompts, aggregated_rewards)
            # split advantages back into two samples
            samples1["advantages"] = (
                torch.as_tensor(np.split(advantages,2)[0])
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )
            samples2["advantages"] = (
                torch.as_tensor(np.split(advantages,2)[1])
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )

            del samples1["rewards"]
            del samples1["encoded_prompts"]
            del samples2["rewards"]
            del samples2["encoded_prompts"]

            total_batch_size, num_timesteps = samples1["timesteps"].shape

            perm = torch.randperm(total_batch_size)
            samples1 = {k: v[perm] for k, v in samples1.items()}
            samples2 = {k: v[perm] for k, v in samples2.items()}
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps) for _ in range(total_batch_size)]
            )
            for key in [
                "timesteps",
                "states",
                "next_states",
                "log_probs",
                "time_index",
            ]:
                samples1[key] = samples1[key][
                    torch.arange(total_batch_size)[:, None],
                    perms,
                ]
                samples2[key] = samples2[key][
                    torch.arange(total_batch_size)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched1 = {
                k: v.reshape(-1, training_config.train_batch_size_per_gpu, *v.shape[1:])
                for k, v in samples1.items()
            }
            samples_batched2 = {
                k: v.reshape(-1, training_config.train_batch_size_per_gpu, *v.shape[1:])
                for k, v in samples2.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched1 = [
                dict(zip(samples_batched1, x)) for x in zip(*samples_batched1.values())
            ]
            samples_batched2 = [
                dict(zip(samples_batched2, x)) for x in zip(*samples_batched2.values())
            ]

            # # =================== TRAINING ===================
            # # loop through each sample and train
            pipeline.unet.train()
            info = defaultdict(list)

            # compute w_embedding
            # get w_embedding
            w = torch.tensor(cfg.training.guidance_scale - 1).repeat(
                training_config.train_batch_size_per_gpu
            )
            w_embedding = pipeline.get_guidance_scale_embedding(
                w, embedding_dim=256
            ).to(accelerator.device)

            for _ in range(training_config.num_inner_epochs):
                # shuffle samples_batched
                for i, (sample1, sample2) in tqdm(
                    enumerate(zip(samples_batched1, samples_batched2)),
                    desc=f"Training, {epoch}.{iter}",
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(pipeline.unet):

                        pi_theta_t_y = sample1["log_probs"].sum(dim=1)
                        pi_theta_t_y_prime = sample2["log_probs"].sum(dim=1)

                        pi_theta_y = []
                        pi_theta_y_prime = []

                        for j in tqdm(
                            range(num_timesteps),
                            desc="Timestep",
                            disable=not accelerator.is_local_main_process,
                        ):

                            # ln pi_theta (y|x) / pi_theta_t(y | x) - ln (pi_theta(y' | x) / pi_theta_t(y' | x)
                            model_pred = get_model_prediction(
                                pipeline=pipeline,
                                latents=sample1["states"][:, j],
                                timesteps=sample1["timesteps"][:, j],
                                prompt_embeds=sample1["prompt_embeds"],
                                w_embedding=w_embedding,
                            )
                            lps = []
                            for k in range(training_config.train_batch_size_per_gpu):
                                _, _, lp = pipeline.scheduler.step(
                                    model_output=model_pred[k].unsqueeze(0),
                                    step_index=sample1["time_index"][:, j][k],
                                    timestep=sample1["timesteps"][:, j][k],
                                    sample=sample1["states"][:, j][k].unsqueeze(0),
                                    prev_sample=sample1["next_states"][:, j][k].unsqueeze(0),
                                    return_dict=False,
                                )
                                lps.append(lp)
                            pi_theta_y.append(torch.cat(lps))

                            model_pred = get_model_prediction(
                                pipeline=pipeline,
                                latents=sample2["states"][:, j],
                                timesteps=sample2["timesteps"][:, j],
                                prompt_embeds=sample2["prompt_embeds"],
                                w_embedding=w_embedding,
                            )
                            lps = []
                            for k in range(training_config.train_batch_size_per_gpu):
                                _, _, lp = pipeline.scheduler.step(
                                    model_output=model_pred[k].unsqueeze(0),
                                    step_index=sample2["time_index"][:, j][k],
                                    timestep=sample2["timesteps"][:, j][k],
                                    sample=sample2["states"][:, j][k].unsqueeze(0),
                                    prev_sample=sample2["next_states"][:, j][k].unsqueeze(0),
                                    return_dict=False,
                                )
                                lps.append(lp)
                            pi_theta_y_prime.append(torch.cat(lps))
                            
                        pi_theta_y = torch.cat(pi_theta_y, dim=-1)
                        pi_theta_y_prime = torch.cat(pi_theta_y_prime, dim=-1)

                        pi_theta_y = pi_theta_y.sum(dim=-1)
                        pi_theta_y_prime = pi_theta_y_prime.sum(dim=-1)
                        
                        log_difference = pi_theta_y - pi_theta_t_y - (pi_theta_y_prime - pi_theta_t_y_prime)
                        advantage_difference = sample1["advantages"] - sample2["advantages"]

                        loss = training_config.rebel_lr * log_difference - advantage_difference
                        loss = torch.square(loss).mean()
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        if cfg.wandb.log:
                            accelerator.log(info, step=logging_step)
                        logging_step += 1
                        info = defaultdict(list)

        if epoch % cfg.training.save_interval == 0:
            accelerator.save_state()
