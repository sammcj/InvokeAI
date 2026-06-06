"""Ideogram 4 VAE Decode Invocation.

Decodes packed Ideogram 4 latents to an image. The packed 128-dim latents are denormalised with the
VAE's BatchNorm statistics, unpacked to the 32-channel latent grid, then decoded with the FLUX.2 VAE.
"""

import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.sampling_utils import latent_grid, unpack_latents, vae_latent_norm
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ideogram4_vae_decode",
    title="Latents to Image - Ideogram 4",
    tags=["latents", "image", "vae", "l2i", "ideogram4", "ideogram"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from packed Ideogram 4 latents using the FLUX.2 32-channel VAE."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    width: int = InputField(default=1024, multiple_of=16, description="Image width in pixels.")
    height: int = InputField(default=1024, multiple_of=16, description="Image height in pixels.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        grid_h, grid_w = latent_grid(self.height, self.width)

        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE")
        with vae_info.model_on_device() as (_, vae):
            device = TorchDevice.choose_torch_device()
            shift, scale = vae_latent_norm(vae)

            # Denormalise the packed latents into the VAE's latent space, then unpack to (B, 32, H, W).
            z = latents.to(device=device, dtype=torch.float32)
            z = z * scale.to(device) + shift.to(device)
            z = unpack_latents(z, grid_h, grid_w)

            vae_dtype = next(iter(vae.parameters())).dtype
            decoded = vae.decode(z.to(dtype=vae_dtype), return_dict=False)[0]

        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_np = (img * 255).byte().cpu().numpy()
        image = Image.fromarray(img_np, mode="RGB")

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
