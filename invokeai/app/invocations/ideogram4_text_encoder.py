"""Ideogram 4 Text Encoder Invocation.

Ideogram 4 conditions on the Qwen3-VL-8B text encoder run in text-only mode. The hidden states from
13 decoder layers are concatenated into a 53248-dim per-token feature, matching the reference
implementation. The image-token region of the packed sequence is assembled later, in the denoise node.
"""

from contextlib import ExitStack
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.masking_utils import create_causal_mask

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Ideogram4ConditioningField,
    Input,
    InputField,
    OutputField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.constants import MAX_TEXT_TOKENS, QWEN3_VL_ACTIVATION_LAYERS
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device


@invocation_output("ideogram4_text_encoder_output")
class Ideogram4TextEncoderOutput(BaseInvocationOutput):
    """Ideogram 4 text encoder output."""

    conditioning: Ideogram4ConditioningField = OutputField(
        description="Stacked Qwen3-VL text features for the prompt.", title="Conditioning"
    )


@invocation(
    "ideogram4_text_encoder",
    title="Prompt - Ideogram 4",
    tags=["prompt", "conditioning", "ideogram4", "ideogram", "qwen3"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for Ideogram 4 using the Qwen3-VL text encoder.

    The Qwen3-VL hidden states from 13 decoder layers are concatenated per token into a 53248-dim
    feature vector. Only the real text tokens are encoded here; the denoise node assembles the full
    packed text+image sequence.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Ideogram4TextEncoderOutput:
        with ExitStack() as exit_stack:
            features = self._encode_prompt(context, exit_stack)
        conditioning_name = context.tensors.save(features)
        return Ideogram4TextEncoderOutput(conditioning=Ideogram4ConditioningField(conditioning_name=conditioning_name))

    def _encode_prompt(self, context: InvocationContext, exit_stack: ExitStack) -> torch.Tensor:
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)
        (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

        if not isinstance(text_encoder, PreTrainedModel):
            raise TypeError(f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}.")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}.")

        device = get_effective_device(text_encoder)

        # Ideogram 4 tokenises the prompt with the Qwen chat template, text-only.
        messages: Any = [{"role": "user", "content": [{"type": "text", "text": self.prompt}]}]
        text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        num_prompt_tokens = input_ids.shape[1]
        if num_prompt_tokens > MAX_TEXT_TOKENS:
            raise ValueError(
                f"Prompt tokenises to {num_prompt_tokens} tokens, exceeding the Ideogram 4 limit of "
                f"{MAX_TEXT_TOKENS}. Shorten the prompt."
            )

        context.util.signal_progress("Running Qwen3-VL text encoder (Ideogram 4)")

        # Qwen3VLTextModel.forward returns only the final normed state, so it can't surface the 13
        # intermediate activation layers. Run the decoder stack manually to tap them, matching the
        # reference encoder. For text the multi-axis RoPE degenerates to standard sequential positions.
        language_model: Any = text_encoder.language_model
        inputs_embeds = language_model.embed_tokens(input_ids)

        num_tokens = input_ids.shape[1]
        pos_2d = torch.arange(num_tokens, device=device).unsqueeze(0)  # (1, L)
        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)  # (4, 1, L)
        text_position_ids = position_ids_4d[0]  # (1, L)
        mrope_position_ids = position_ids_4d[1:]  # (3, 1, L)

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
        selected = [captured[layer_idx] for layer_idx in QWEN3_VL_ACTIVATION_LAYERS]

        # Stack layers into a per-token feature with layer index fastest: (B, L, hidden * num_layers).
        stacked = torch.stack(selected, dim=0)  # (num_layers, B, L, hidden)
        stacked = stacked.permute(1, 2, 3, 0)  # (B, L, hidden, num_layers)
        batch_size, seq_len = input_ids.shape
        stacked = stacked.reshape(batch_size, seq_len, -1)

        # Zero any non-attended tokens, then keep features in float32 (the schedule/denoise math is f32).
        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        stacked = stacked * text_mask
        features: torch.Tensor = stacked.to(torch.float32).cpu()
        return features
