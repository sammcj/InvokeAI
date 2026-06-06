# Copyright (c) 2026 the InvokeAI Development Team
"""Ideogram 4 sampling utilities.

Ideogram 4 uses a logit-normal flow-matching schedule with a resolution-aware mean shift, and a
single packed sequence that interleaves left-padded text tokens with image (latent) tokens. These
helpers build the schedule, the per-token bookkeeping tensors (position_ids / segment_ids /
indicator) and pack/unpack the latents, mirroring the reference implementation.
"""

import math
from dataclasses import dataclass

import torch
from einops import rearrange

from invokeai.backend.ideogram4.constants import (
    AE_CHANNELS,
    AE_SCALE_FACTOR,
    IMAGE_POSITION_OFFSET,
    LATENT_DIM,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    PATCH_SIZE,
    SEQUENCE_PADDING_INDICATOR,
)


@dataclass(frozen=True)
class LogitNormalSchedule:
    """Logit-normal flow-matching schedule.

    Maps a linear interval in ``[0, 1]`` to a model time ``t`` in ``[0, 1]`` where ``t`` near 0 is
    pure noise and ``t`` near 1 is clean data. The float64 warp (``ndtri``/``expit``) needs precision
    at the tails; MPS supports neither float64 nor these special functions, so the warp runs on CPU
    there and the result is cast back.
    """

    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        if device.type == "mps":
            t = t.cpu().to(torch.float64)
        else:
            t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = 1 - torch.special.expit(y)
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        warped: torch.Tensor = t_.clamp(t_min, t_max)
        return warped.to(device=device, dtype=torch.float32)


def get_schedule_for_resolution(
    height: int,
    width: int,
    known_mean: float = 1.0,
    std: float = 1.0,
    known_resolution: tuple[int, int] = (512, 512),
) -> LogitNormalSchedule:
    """Build the resolution-aware logit-normal schedule.

    The schedule mean is shifted by half the log pixel-count ratio relative to a 512x512 reference,
    so higher resolutions spend more steps at high noise levels.
    """
    num_pixels = height * width
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int, device: torch.device) -> torch.Tensor:
    """Linear step grid in ``[0, 1]`` with ``num_steps + 1`` points."""
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32, device=device)


def build_guidance_schedule(
    num_steps: int,
    guidance_scale: float,
    polish_guidance_scale: float,
    polish_steps: int,
) -> list[float]:
    """Build the per-step guidance schedule in step order (index 0 = first, highest-noise step).

    Ideogram 4's recommended presets run the high-noise steps at ``guidance_scale`` and the final
    ``polish_steps`` at the lower ``polish_guidance_scale`` to refine detail. ``polish_steps`` is
    clamped to ``[0, num_steps]``. This ordering matches the diffusers Ideogram4Pipeline
    ``guidance_schedule`` (first entry applies to the first step).
    """
    polish = min(max(polish_steps, 0), num_steps)
    main = num_steps - polish
    return [guidance_scale] * main + [polish_guidance_scale] * polish


def resolve_step_range(num_steps: int, denoising_start: float, denoising_end: float) -> tuple[int, int]:
    """Map ``denoising_start``/``denoising_end`` fractions in ``[0, 1]`` to a ``[start, end)`` step range.

    ``denoising_start`` is the fraction of the schedule to skip (0 = start from pure noise for txt2img,
    higher preserves more of the init image for img2img). The returned ``start_step`` is the first step
    index to run and ``end_step`` is one past the last.
    """
    start_step = round(denoising_start * num_steps)
    end_step = round(denoising_end * num_steps)
    start_step = max(0, min(start_step, num_steps))
    end_step = max(start_step, min(end_step, num_steps))
    return start_step, end_step


def latent_grid(height: int, width: int) -> tuple[int, int]:
    """Return the (grid_h, grid_w) packed-token grid for a pixel resolution.

    Each packed image token covers ``ae_scale_factor * patch_size`` pixels per axis.
    """
    patch = PATCH_SIZE * AE_SCALE_FACTOR
    if height % patch != 0 or width % patch != 0:
        raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
    return height // patch, width // patch


def build_packed_ids(
    text_lengths: list[int],
    grid_h: int,
    grid_w: int,
    max_text_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build position_ids / segment_ids / indicator for the packed ``[pad][text][image]`` sequence.

    Per sample the layout is ``[left-pad][text tokens][image tokens]`` so all prompts in a batch share
    the same ``max_text_tokens`` text region width.

    Returns:
        position_ids: (B, total_seq_len, 3) int64 MRoPE coordinates (t, h, w).
        segment_ids:  (B, total_seq_len) int64 block-diagonal attention id (1 = real sample, -1 = pad).
        indicator:    (B, total_seq_len) int64 per-token role marker.
    """
    batch_size = len(text_lengths)
    num_image_tokens = grid_h * grid_w
    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, num_text in enumerate(text_lengths):
        offset = max_text_tokens - num_text
        total_unpadded = num_text + num_image_tokens

        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        position_ids[b, offset : offset + num_text] = text_pos_3d
        position_ids[b, offset + num_text :] = image_pos

        indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
        indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

        segment_ids[b, offset : offset + total_unpadded] = 1

    return position_ids.to(device), segment_ids.to(device), indicator.to(device)


def get_noise(
    batch_size: int,
    num_image_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample the initial packed image latents (B, num_image_tokens, 128) in float32.

    Noise is generated on the CPU for cross-device reproducibility, then moved to ``device``.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        batch_size,
        num_image_tokens,
        LATENT_DIM,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device)


def vae_latent_norm(vae: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the (shift, scale) latent denormalisation from the VAE's BatchNorm buffers.

    Ideogram 4's AutoencoderKLFlux2 carries a ``bn`` BatchNorm over the 128-dim packed latent space.
    Decoding denormalises with ``z * scale + shift`` where ``shift = running_mean`` and
    ``scale = sqrt(running_var + eps)``.
    """
    bn: torch.nn.BatchNorm2d = vae.get_submodule("bn")  # type: ignore[assignment]
    eps = bn.eps
    assert bn.running_mean is not None and bn.running_var is not None
    shift = bn.running_mean.detach().to(torch.float32)
    scale = torch.sqrt(bn.running_var.detach().to(torch.float32) + eps)
    return shift, scale


def unpack_latents(z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Unpack denormalised packed latents (B, grid_h*grid_w, 128) to (B, 32, grid_h*2, grid_w*2).

    The 128-dim latent splits as ``(patch, patch, ae_channels)`` with channels innermost, matching the
    reference ``view(B, grid_h, grid_w, patch, patch, ae_channels).permute(0, 5, 1, 3, 2, 4)``.
    """
    return rearrange(
        z,
        "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
        gh=grid_h,
        gw=grid_w,
        ph=PATCH_SIZE,
        pw=PATCH_SIZE,
        c=AE_CHANNELS,
    )


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    """Pack VAE latents (B, 32, H, W) into the packed token layout (B, (H/2)*(W/2), 128).

    This is the exact inverse of :func:`unpack_latents`, used by the image-to-latents path so an
    encoded init image lands in the same packed space the denoise loop operates on.
    """
    return rearrange(
        z,
        "b c (gh ph) (gw pw) -> b (gh gw) (ph pw c)",
        ph=PATCH_SIZE,
        pw=PATCH_SIZE,
        c=AE_CHANNELS,
    )
