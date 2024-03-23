import torch


def get_model_prediction(
    pipeline,
    latents,
    timesteps,
    w_embedding,
    prompt_embeds,
):
    return pipeline.unet(
        latents,
        timesteps,
        timestep_cond=w_embedding,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=None,
        return_dict=False,
    )[0]
