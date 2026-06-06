# Copyright (c) 2026 the InvokeAI Development Team
"""Ideogram 4 denoising loop.

Ideogram 4 uses asymmetric classifier-free guidance with two distinct transformers: a conditional
transformer that sees the full ``[text][image]`` sequence, and an unconditional transformer that sees
the image tokens only with zeroed text conditioning. Each Euler step blends their velocity
predictions with a per-step guidance weight, mirroring the reference implementation.
"""

from typing import Callable, Optional

import torch
from tqdm import tqdm

from invokeai.backend.ideogram4.sampling_utils import LogitNormalSchedule
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    conditional_transformer: torch.nn.Module,
    unconditional_transformer: torch.nn.Module,
    # conditioning
    text_features: torch.Tensor,
    position_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    indicator: torch.Tensor,
    max_text_tokens: int,
    # initial latents
    z: torch.Tensor,
    # schedule
    schedule: LogitNormalSchedule,
    step_intervals: torch.Tensor,
    guidance_weights: torch.Tensor,
    num_steps: int,
    step_callback: Callable[[PipelineIntermediateState], None],
    start_step: int = 0,
    end_step: Optional[int] = None,
    inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None,
) -> torch.Tensor:
    """Denoise packed Ideogram 4 latents with asymmetric CFG over two transformers.

    Args:
        conditional_transformer: Ideogram4Transformer2DModel for the conditional pass.
        unconditional_transformer: Ideogram4Transformer2DModel for the image-only unconditional pass.
        text_features: (B, max_text_tokens, 53248) float text conditioning from Qwen3-VL.
        position_ids: (B, total_seq_len, 3) int64 MRoPE coordinates for the full packed sequence.
        segment_ids: (B, total_seq_len) int64 attention block ids.
        indicator: (B, total_seq_len) int64 per-token role markers.
        max_text_tokens: Width of the text region; image tokens follow it.
        z: (B, num_image_tokens, 128) float32 starting latents. For txt2img this is pure noise; for
            img2img/inpaint it is the init latents already blended with noise at ``start_step``.
        schedule: Logit-normal flow-matching schedule mapping intervals to model time.
        step_intervals: (num_steps + 1,) linear interval grid.
        guidance_weights: (num_steps,) per-step guidance weight in step order (index 0 = first step
            at the highest noise level, index num_steps - 1 = final step), matching the diffusers
            Ideogram4Pipeline `guidance_schedule` convention.
        num_steps: Number of denoising steps.
        step_callback: Progress callback receiving packed clean-latent estimates.
        start_step: First step index to run (>0 for img2img, skipping the highest-noise steps).
        end_step: One past the last step index to run (defaults to ``num_steps``; <num_steps stops early).
        inpaint_extension: When set, after each step the non-masked latents are re-pinned to the noised
            init image so only the masked region is denoised (inpaint/outpaint).

    Returns:
        (B, num_image_tokens, 128) float32 denoised packed latents.
    """
    batch_size = z.shape[0]
    device = z.device
    dtype = next(conditional_transformer.parameters()).dtype
    if end_step is None:
        end_step = num_steps

    # The unconditional branch is image-only: slice the bookkeeping tensors past the text region and
    # zero the text conditioning.
    neg_position_ids = position_ids[:, max_text_tokens:]
    neg_segment_ids = segment_ids[:, max_text_tokens:]
    neg_indicator = indicator[:, max_text_tokens:]
    num_image_tokens = z.shape[1]
    neg_llm_features = torch.zeros(
        batch_size, num_image_tokens, text_features.shape[-1], dtype=text_features.dtype, device=device
    )

    # The conditional branch pads the text region of the latent stream with zeros and pads the image
    # region of the text features with zeros, so both streams span the full packed sequence.
    text_z_padding = torch.zeros(batch_size, max_text_tokens, z.shape[-1], dtype=torch.float32, device=device)
    image_feature_padding = torch.zeros(
        batch_size, num_image_tokens, text_features.shape[-1], dtype=text_features.dtype, device=device
    )
    llm_features = torch.cat([text_features, image_feature_padding], dim=1)

    # Steps run in step order (index 0 = highest noise). img2img/inpaint start partway through.
    step_pairs = list(enumerate(range(num_steps - 1, -1, -1)))[start_step:end_step]
    pbar = tqdm(total=len(step_pairs), desc="Denoising")
    # Integrate from noise (t near 0) toward clean data (t near 1).
    for callback_step, (step_index, i) in enumerate(step_pairs):
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        # The final step integrates to clean data (model time 1.0). The diffusers pipeline does this by
        # appending a terminal zero sigma rather than using the schedule's clamped endpoint (~0.9994), so
        # match that here and step the last interval all the way to 1.0.
        s_val = 1.0 if i == 0 else float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((batch_size,), t_val, dtype=dtype, device=device)

        pos_z = torch.cat([text_z_padding, z], dim=1).to(dtype)
        pos_out = conditional_transformer(
            hidden_states=pos_z,
            timestep=t,
            encoder_hidden_states=llm_features.to(dtype),
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
            return_dict=False,
        )[0]
        pos_v = pos_out[:, max_text_tokens:].to(torch.float32)

        neg_out = unconditional_transformer(
            hidden_states=z.to(dtype),
            timestep=t,
            encoder_hidden_states=neg_llm_features.to(dtype),
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
            return_dict=False,
        )[0]
        neg_v = neg_out.to(torch.float32)

        gw_i = float(guidance_weights[step_index].item())
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)

        # Inpaint/outpaint: re-pin the non-masked region to the init image noised to the next model
        # time. The extension works in sigma convention (1 = noise, 0 = clean), so pass 1 - s_val.
        if inpaint_extension is not None:
            z = inpaint_extension.merge_intermediate_latents_with_init_latents(z, 1.0 - s_val)

        pbar.update(1)
        # Euler extrapolation to t = 1 gives the current clean-data estimate.
        preview = z + (1.0 - s_val) * v
        step_callback(
            PipelineIntermediateState(
                step=callback_step + 1,
                order=1,
                total_steps=len(step_pairs),
                timestep=int(t_val * 1000),
                latents=preview,
            ),
        )

    pbar.close()
    return z
