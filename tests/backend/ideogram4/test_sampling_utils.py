import torch

from invokeai.backend.ideogram4.constants import (
    AE_CHANNELS,
    IMAGE_POSITION_OFFSET,
    LATENT_DIM,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from invokeai.backend.ideogram4.sampling_utils import (
    build_guidance_schedule,
    build_packed_ids,
    get_noise,
    get_schedule_for_resolution,
    latent_grid,
    make_step_intervals,
    pack_latents,
    resolve_step_range,
    unpack_latents,
)

CPU = torch.device("cpu")


def test_schedule_runs_from_noise_to_clean() -> None:
    # The schedule maps interval 0 -> model time ~1 (noise) and interval 1 -> ~0 (clean), and is
    # monotonically decreasing in between.
    schedule = get_schedule_for_resolution(1024, 1024, known_mean=0.0, std=1.75)
    intervals = make_step_intervals(20, CPU)
    values = [float(schedule(intervals[i].unsqueeze(0)).item()) for i in range(len(intervals))]

    assert values[0] > 0.9
    assert values[-1] < 0.1
    assert all(a >= b for a, b in zip(values[:-1], values[1:], strict=True))


def test_resolution_shift_increases_schedule_mean() -> None:
    low = get_schedule_for_resolution(512, 512, known_mean=0.0)
    high = get_schedule_for_resolution(1024, 1024, known_mean=0.0)
    assert high.mean > low.mean


def test_latent_grid_matches_patch_geometry() -> None:
    # Each packed token covers ae_scale_factor (8) * patch_size (2) = 16 pixels per axis.
    assert latent_grid(512, 512) == (32, 32)
    assert latent_grid(1024, 768) == (64, 48)


def test_build_packed_ids_layout() -> None:
    grid_h, grid_w = 4, 4
    num_image = grid_h * grid_w
    num_text = 5
    position_ids, segment_ids, indicator = build_packed_ids([num_text], grid_h, grid_w, num_text, CPU)

    total = num_text + num_image
    assert position_ids.shape == (1, total, 3)
    assert segment_ids.shape == (1, total)
    assert indicator.shape == (1, total)

    # Single prompt: no left padding, so the first num_text tokens are text, the rest are image.
    assert torch.equal(indicator[0, :num_text], torch.full((num_text,), LLM_TOKEN_INDICATOR))
    assert torch.equal(indicator[0, num_text:], torch.full((num_image,), OUTPUT_IMAGE_INDICATOR))
    assert SEQUENCE_PADDING_INDICATOR not in indicator.tolist()[0]

    # Text positions are sequential and identical on all three axes.
    assert torch.equal(position_ids[0, 0], torch.tensor([0, 0, 0]))
    assert torch.equal(position_ids[0, num_text - 1], torch.tensor([num_text - 1] * 3))

    # Image positions are offset and carry the (h, w) grid coordinates.
    assert torch.equal(position_ids[0, num_text], torch.tensor([0, 0, 0]) + IMAGE_POSITION_OFFSET)
    last = position_ids[0, -1]
    assert int(last[1]) == IMAGE_POSITION_OFFSET + grid_h - 1
    assert int(last[2]) == IMAGE_POSITION_OFFSET + grid_w - 1

    # The whole single-prompt sequence shares one attention segment.
    assert torch.equal(segment_ids[0], torch.ones(total, dtype=torch.long))


def test_left_padding_for_short_prompt_in_batch() -> None:
    # With a batch the shorter prompt is left-padded to max_text_tokens.
    grid_h, grid_w = 2, 2
    max_text = 5
    _, segment_ids, indicator = build_packed_ids([3, 5], grid_h, grid_w, max_text, CPU)

    pad_len = max_text - 3
    # Padding region of the short prompt has the padding segment id and zero indicator.
    assert torch.equal(segment_ids[0, :pad_len], torch.full((pad_len,), SEQUENCE_PADDING_INDICATOR))
    assert torch.equal(indicator[0, :pad_len], torch.zeros(pad_len, dtype=torch.long))
    # The long prompt fills the whole text region.
    assert torch.equal(indicator[1, :max_text], torch.full((max_text,), LLM_TOKEN_INDICATOR))


def test_get_noise_shape_and_determinism() -> None:
    a = get_noise(1, 16, seed=42, device=CPU)
    b = get_noise(1, 16, seed=42, device=CPU)
    c = get_noise(1, 16, seed=43, device=CPU)
    assert a.shape == (1, 16, LATENT_DIM)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_build_guidance_schedule_polish_steps_are_last() -> None:
    # The high-noise steps come first at the main weight; the final polish_steps use the lower weight.
    # This is the V4_DEFAULT_20 preset: 18 steps at 7.0, then 2 polish steps at 3.0.
    schedule = build_guidance_schedule(num_steps=20, guidance_scale=7.0, polish_guidance_scale=3.0, polish_steps=2)
    assert schedule == [7.0] * 18 + [3.0] * 2


def test_build_guidance_schedule_clamps_polish_steps() -> None:
    # polish_steps is clamped to [0, num_steps].
    assert build_guidance_schedule(4, 7.0, 3.0, polish_steps=10) == [3.0] * 4
    assert build_guidance_schedule(4, 7.0, 3.0, polish_steps=0) == [7.0] * 4
    assert build_guidance_schedule(4, 7.0, 3.0, polish_steps=-1) == [7.0] * 4


def test_unpack_latents_shape() -> None:
    grid_h, grid_w = 6, 4
    z = torch.randn(1, grid_h * grid_w, LATENT_DIM)
    img = unpack_latents(z, grid_h, grid_w)
    assert img.shape == (1, AE_CHANNELS, grid_h * 2, grid_w * 2)


def test_resolve_step_range() -> None:
    # denoising_start=0 -> txt2img full range; fractions map to step indices and clamp.
    assert resolve_step_range(20, 0.0, 1.0) == (0, 20)
    assert resolve_step_range(20, 0.6, 1.0) == (12, 20)
    assert resolve_step_range(20, 0.5, 0.75) == (10, 15)
    # end is clamped to be >= start, and both clamp to [0, num_steps].
    assert resolve_step_range(20, 1.0, 1.0) == (20, 20)
    assert resolve_step_range(20, 0.8, 0.2) == (16, 16)


def test_pack_unpack_latents_round_trip() -> None:
    # pack_latents is the exact inverse of unpack_latents, so a round trip is the identity.
    grid_h, grid_w = 6, 4
    packed = torch.randn(1, grid_h * grid_w, LATENT_DIM)
    unpacked = unpack_latents(packed, grid_h, grid_w)
    assert unpacked.shape == (1, AE_CHANNELS, grid_h * 2, grid_w * 2)
    repacked = pack_latents(unpacked)
    assert repacked.shape == packed.shape
    assert torch.equal(repacked, packed)
