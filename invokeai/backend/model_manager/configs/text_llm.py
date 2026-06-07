from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_class_name_from_config_dict_or_raise,
    get_config_dict_or_raise,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)


def _loads_as_causal_lm(config_dict: dict[str, Any]) -> bool:
    """True if transformers' ``AutoModelForCausalLM`` can load this config by ``model_type``.

    Covers models whose config lists a ``*ForConditionalGeneration`` architecture (e.g. Qwen3.5's
    ``Qwen3_5ForConditionalGeneration``) but still resolve to a ``*ForCausalLM`` class under
    ``AutoModelForCausalLM``.
    """
    model_type = config_dict.get("model_type")
    if not isinstance(model_type, str):
        return False
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    return model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class TextLLM_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for text-only causal language models (e.g. Llama, Phi, Qwen, Mistral)."""

    type: Literal[ModelType.TextLLM] = Field(default=ModelType.TextLLM)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check that the model's architecture is a causal language model.
        # This covers LlamaForCausalLM, PhiForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM,
        # MistralForCausalLM, GemmaForCausalLM, GPTNeoXForCausalLM, etc.
        #
        # Some newer causal LMs (e.g. Qwen3.5) declare a *ForConditionalGeneration architecture in their
        # config but still load under AutoModelForCausalLM. For those, fall back to the transformers
        # causal-LM registry keyed by model_type, so they are recognised as TextLLMs rather than unknown.
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))
        class_name = get_class_name_from_config_dict_or_raise(config_dict)
        if not class_name.endswith("ForCausalLM") and not _loads_as_causal_lm(config_dict):
            raise NotAMatchError(f"model architecture '{class_name}' is not a causal language model")

        # Verify tokenizer files exist to avoid runtime failures
        tokenizer_files = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
        if not any((mod.path / f).exists() for f in tokenizer_files):
            raise NotAMatchError(
                f"no tokenizer files found in '{mod.path}' "
                f"(expected at least one of: {', '.join(sorted(tokenizer_files))})"
            )

        return cls(**override_fields)
