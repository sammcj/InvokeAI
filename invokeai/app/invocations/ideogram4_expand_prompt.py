"""Ideogram 4 magic-prompt expansion invocation.

Expands a plain prompt into Ideogram 4's structured JSON caption with a local text LLM, so the linear
UI can auto-expand at generation time. Mirrors the local-LLM path of the ``expand-prompt`` API route.
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    Classification,
    invocation,
)
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.magic_prompt import IDEOGRAM4_V1, get_preset
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ideogram4_expand_prompt",
    title="Magic Prompt - Ideogram 4",
    tags=["prompt", "ideogram4", "ideogram", "magic", "llm"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4ExpandPromptInvocation(BaseInvocation):
    """Expands a plain prompt into an Ideogram 4 structured JSON caption using a text LLM.

    Ideogram 4 is trained on structured JSON captions; a plain prompt samples out-of-distribution and
    the model tends to return its baked-in safety-filter card. This node rewrites the prompt with the
    Ideogram magic-prompt system prompt. If the LLM returns something that isn't valid JSON the text is
    passed through unchanged, so generation still proceeds.
    """

    prompt: str = InputField(description="Plain text prompt to expand.", ui_component=UIComponent.Textarea)
    text_llm_model: ModelIdentifierField = InputField(
        description="Text LLM used to expand the prompt into a caption.",
        ui_model_type=[ModelType.TextLLM],
    )
    max_tokens: int = InputField(
        default=2048,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate for the expanded caption.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        preset = get_preset(IDEOGRAM4_V1)
        assert preset is not None, "Ideogram 4 magic-prompt preset is missing."

        model_config = context.models.get_config(self.text_llm_model)
        if model_config.type != ModelType.TextLLM:
            raise ValueError(f"Model '{self.text_llm_model.key}' is not a TextLLM model (got {model_config.type}).")

        model_path = Path(model_config.path)
        if not model_path.is_absolute():
            model_path = context.config.get().models_path / model_path

        loaded_model = context.models.load(self.text_llm_model)
        with loaded_model.model_on_device() as (_, model):
            assert isinstance(model, PreTrainedModel)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            pipeline = TextLLMPipeline(model, tokenizer)
            output = pipeline.run(
                prompt=self.prompt,
                system_prompt=preset.system_prompt,
                max_new_tokens=self.max_tokens,
                device=next(model.parameters()).device,
                dtype=TorchDevice.choose_torch_dtype(),
            )

        return StringOutput(value=preset.postprocess(output))
