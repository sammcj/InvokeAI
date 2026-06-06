"""Ideogram 4 Model Loader Invocation.

Loads an Ideogram 4 diffusers pipeline, outputting its submodels: the conditional and unconditional
transformers (asymmetric CFG), the Qwen3-VL text encoder + tokenizer, and the AutoencoderKLFlux2 VAE.
"""

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    ModelIdentifierField,
    Qwen3EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
    SubModelType,
)


@invocation_output("ideogram4_model_loader_output")
class Ideogram4ModelLoaderOutput(BaseInvocationOutput):
    """Ideogram 4 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    unconditional_transformer: TransformerField = OutputField(
        description="The unconditional transformer used for the asymmetric-CFG negative pass.",
        title="Unconditional Transformer",
    )
    qwen3_encoder: Qwen3EncoderField = OutputField(
        description=FieldDescriptions.qwen3_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "ideogram4_model_loader",
    title="Main Model - Ideogram 4",
    tags=["model", "ideogram4", "ideogram", "qwen3"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4ModelLoaderInvocation(BaseInvocation):
    """Loads an Ideogram 4 model, outputting its submodels.

    Ideogram 4 is a single-stream DiT with a Qwen3-VL text encoder and the FLUX.2 32-channel VAE.
    It uses asymmetric classifier-free guidance with two distinct transformers, both extracted here
    from the diffusers pipeline.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.main_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        title="Ideogram 4 Model",
    )

    def invoke(self, context: InvocationContext) -> Ideogram4ModelLoaderOutput:
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        unconditional_transformer = self.model.model_copy(
            update={"submodel_type": SubModelType.TransformerUnconditional}
        )
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return Ideogram4ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            unconditional_transformer=TransformerField(transformer=unconditional_transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )
