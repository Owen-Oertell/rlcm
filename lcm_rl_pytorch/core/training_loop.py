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
logger = get_logger(__name__)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def training_loop(accelerator, cfg, training_config, pipeline, reward_fn, dataset):
    logging_step = 0
    stat_tracker = PerPromptStatTracker(
        training_config.stat_buffer_size, training_config.stat_min_count
    )
    executor = futures.ThreadPoolExecutor(max_workers=2)


    # load optimizer, prepare unet
    optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=training_config.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,end_factor=0.0, total_iters=training_config.num_epochs*3)

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
            samples = []
            last_images = None
            # =================== SAMPLING ===================
            for _ in range(training_config.num_inner_epochs):
                # sample from dataset
                prompts = dataset.sample()

                # sample images
                sample_dict = pipeline(
                    prompt=prompts,
                    num_inference_steps=training_config.num_inference_steps,
                    guidance_scale=cfg.training.guidance_scale,
                    lcm_origin_steps=cfg.training.lcm_origin_steps,
                    output_type="pt",
                )

                # obtain reward
                print("prompts: ", prompts)
                print("submitting reward function")
                rewards = executor.submit(reward_fn, sample_dict["images"], prompts)

                # add rewards to sample dict
                sample_dict["rewards"] = rewards
                sample_dict["encoded_prompts"] = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.training.max_prompt_length,
                ).input_ids

                last_images = sample_dict["images"]

                samples.append(sample_dict)

            # =================== BATCHING (and Logging and advantage computation) ===================
            for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
            ):
                rewards = sample["rewards"].result()
                sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

            # mostly copied from kevins repo
            # create super batch
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            # log reward information
            # exit()

            if not isinstance(samples["rewards"], torch.Tensor):
                samples["rewards"] = torch.tensor(samples["rewards"])
            gathered_rewards = accelerator.gather(samples["rewards"].to(accelerator.device)).cpu().numpy()
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
                        "rewards": gathered_rewards,
                        "mean_reward": gathered_rewards.mean(),
                        "std_reward": gathered_rewards.std(),
                        "epoch": epoch,
                    },
                    step=logging_step,
                )

            # compute advantages with per prompt stat tracking
            if not isinstance(samples["encoded_prompts"], torch.Tensor):
                samples["encoded_prompts"] = torch.tensor(samples["encoded_prompts"])
            prompt_ids = accelerator.gather(samples["encoded_prompts"].to(accelerator.device)).cpu().numpy()

            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards)

            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index].to(accelerator.device)
            )


            del samples["rewards"]
            del samples["encoded_prompts"]

            total_batch_size, num_timesteps = samples["timesteps"].shape

            perm = torch.randperm(total_batch_size)
            samples = {k: v[perm] for k, v in samples.items()}
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
                samples[key] = samples[key][
                    torch.arange(total_batch_size)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, training_config.train_batch_size_per_gpu, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # # =================== TRAINING ===================
            # # loop through each sample and train
            pipeline.unet.train()
            info = defaultdict(list)

            # compute w_embedding
            # get w_embedding
            w = torch.tensor(cfg.training.guidance_scale-1).repeat(
                training_config.train_batch_size_per_gpu
            )
            w_embedding = pipeline.get_guidance_scale_embedding(w, embedding_dim=256).to(
                accelerator.device
            )

            for k in range(training_config.train_num_inner_epochs):
                # shuffle samples_batched
                shuffle(samples_batched)

                for i, sample in tqdm(
                    enumerate(samples_batched),
                    desc=f"Training, {epoch}.{iter}",
                    disable=not accelerator.is_local_main_process,
                ):
                    for j in tqdm(
                        range(num_timesteps),
                        desc="Timestep",
                        disable=not accelerator.is_local_main_process,
                    ):
                        with accelerator.accumulate(pipeline.unet):
                            model_prediction = get_model_prediction(
                                pipeline=pipeline,
                                latents=sample["states"][:, j],
                                timesteps=sample["timesteps"][:, j],
                                prompt_embeds=sample["prompt_embeds"],
                                w_embedding=w_embedding,
                            )

                            # compute log prob of next_latents given latents under current model
                            lps = []
                            for k in range(training_config.train_batch_size_per_gpu):
                                _, _, lp = pipeline.scheduler.step(
                                    model_output=model_prediction[k].unsqueeze(0),
                                    step_index=sample["time_index"][:, j][k],
                                    timestep=sample["timesteps"][:, j][k],
                                    sample=sample["states"][:, j][k].unsqueeze(0),
                                    prev_sample=sample["next_states"][:, j][k].unsqueeze(0),
                                    return_dict=False,
                                )
                                lps.append(lp)

                            log_prob = torch.cat(lps)

                            advantages = torch.clamp(
                                sample["advantages"],
                                -training_config.adv_clip_max,
                                training_config.adv_clip_max,
                            )
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - training_config.clip_range,
                                1.0 + training_config.clip_range,
                            )
                            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            # debugging values
                            # John Schulman says that (ratio - 1) - log(ratio) is a better
                            # estimator, but most existing code uses this so...
                            # http://joschu.net/blog/kl-approx.html
                            info["approx_kl"].append(
                                0.5
                                * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                            )
                            info["clipfrac"].append(
                                torch.mean(
                                    (
                                        torch.abs(ratio - 1.0) > training_config.clip_range
                                    ).float()
                                )
                            )
                            # info["adv_mean"].append(advantages_mean)
                            info["ratio"].append(torch.mean(ratio))
                            info["loss"].append(loss)

                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    pipeline.unet.parameters(), training_config.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                        if accelerator.sync_gradients:
                            assert (j == num_timesteps - 1) and (
                                i + 1
                            ) % training_config.gradient_accumulation_steps == 0
                            # log training-related stuff
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            if cfg.wandb.log:
                                accelerator.log(info, step=logging_step)
                            logging_step += 1
                            info = defaultdict(list)

        if epoch % cfg.training.save_interval == 0:
            accelerator.save_state()