"""Ideogram 4 Image-to-Latents Invocation.

Encodes an image into the packed, normalised latent space the Ideogram 4 denoise loop operates in:
the FLUX.2 32-channel VAE encodes to (B, 32, H/8, W/8), the latents are packed to (B, N, 128) and
then normalised with the VAE's BatchNorm statistics. This is the exact inverse of the VAE decode node.
"""

import einops
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.sampling_utils import pack_latents, vae_latent_norm
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ideogram4_i2l",
    title="Image to Latents - Ideogram 4",
    tags=["latents", "image", "vae", "i2l", "ideogram4", "ideogram"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4ImageToLatentsInvocation(BaseInvocation):
    """Encodes an image into packed, normalised Ideogram 4 latents for image-to-image and inpainting."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE Encode")
        with vae_info.model_on_device() as (_, vae):
            device = TorchDevice.choose_torch_device()
            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=device, dtype=vae_dtype)

            latent_dist = vae.encode(image_tensor, return_dict=False)[0]
            latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()

            # Pack into the (B, N, 128) token layout, then normalise into the denoise space using the
            # VAE BatchNorm stats (inverse of the decode node's z * scale + shift).
            shift, scale = vae_latent_norm(vae)
            packed = pack_latents(latents.to(torch.float32))
            packed = (packed - shift.to(packed.device)) / scale.to(packed.device)

        packed = packed.to("cpu")
        name = context.tensors.save(tensor=packed)
        return LatentsOutput.build(latents_name=name, latents=packed, seed=None)
