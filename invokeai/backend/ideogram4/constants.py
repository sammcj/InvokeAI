# Copyright (c) 2026 the InvokeAI Development Team
"""Constants for the Ideogram 4 packed text+image sequence layout.

Ideogram 4 is a single-stream DiT: text tokens and image (latent) tokens share one sequence.
Each token carries an ``indicator`` marking its role and a 3-axis ``position_ids`` entry for the
multi-axis rotary embedding (MRoPE). These values mirror the reference implementation.
"""

# Per-token role markers used in the ``indicator`` tensor.
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image position ids are offset so they never collide with text positions in the shared MRoPE space.
IMAGE_POSITION_OFFSET = 65536

# Qwen3-VL decoder layers whose hidden states are tapped and concatenated to form the text conditioning.
# 13 layers x 4096 hidden size = 53248-dim per-token feature.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Latent geometry: AutoencoderKLFlux2 downsamples by 8, the transformer patchifies 2x2.
# Latent dim = ae_channels (32) * patch_size**2 (4) = 128.
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8
AE_CHANNELS = 32
LATENT_DIM = AE_CHANNELS * PATCH_SIZE * PATCH_SIZE

# Maximum number of text tokens the text encoder will accept per prompt.
MAX_TEXT_TOKENS = 2048
