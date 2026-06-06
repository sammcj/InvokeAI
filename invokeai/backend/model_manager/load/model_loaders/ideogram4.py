# Copyright (c) 2026 the InvokeAI Development Team
"""Model loaders for Ideogram 4 models."""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings


@ModelLoaderRegistry.register(base=BaseModelType.Ideogram4, type=ModelType.Main, format=ModelFormat.Diffusers)
class Ideogram4DiffusersModel(GenericDiffusersLoader):
    """Load Ideogram 4 main models in diffusers format.

    Ideogram 4 ships as a diffusers pipeline (Ideogram4Pipeline) with these submodels: transformer
    and unconditional_transformer (both Ideogram4Transformer2DModel, but distinct weights), vae
    (AutoencoderKLFlux2 - the same VAE as FLUX.2), text_encoder (Qwen3VLModel) and tokenizer
    (Qwen2Tokenizer).

    Ideogram 4 uses asymmetric classifier-free guidance: the conditional pass runs the `transformer`
    over the full text+image sequence, and the unconditional pass runs the separately-trained
    `unconditional_transformer` over the image tokens only. Both are loaded as submodels.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("Checkpoint format is not implemented for Ideogram 4 diffusers models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        submodel_path = model_path / submodel_type.value

        # Ideogram 4 requires bfloat16 for correct inference. low_cpu_mem_usage=False avoids meta tensors
        # for any weights the model class creates that are not present in the checkpoint.
        #
        # Exception: the Qwen3-VL text encoder produces NaN hidden states in bfloat16 on MPS, so it is
        # loaded in float32 there. It runs once per prompt, so the extra memory/time is acceptable.
        dtype = torch.bfloat16
        if submodel_type == SubModelType.TextEncoder and TorchDevice.choose_torch_device().type == "mps":
            dtype = torch.float32
        with SilenceWarnings():
            try:
                result: AnyModel = load_class.from_pretrained(
                    submodel_path,
                    torch_dtype=dtype,
                    variant=variant,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                )
            except OSError as e:
                if variant and "no file named" in str(e):
                    # Retry without the variant in case the user's preferences changed.
                    result = load_class.from_pretrained(
                        submodel_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                        low_cpu_mem_usage=False,
                    )
                else:
                    raise e

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result
