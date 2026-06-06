"""Ideogram 4 Denoise Invocation.

Runs the Ideogram 4 flow-matching denoise loop with asymmetric classifier-free guidance over the
conditional and unconditional transformers, producing packed latents for the VAE decode node.
"""

from contextlib import ExitStack
from typing import Callable, Optional

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Ideogram4ConditioningField,
    Input,
    InputField,
    LatentsField,
    OutputField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.constants import AE_CHANNELS
from invokeai.backend.ideogram4.denoise import denoise
from invokeai.backend.ideogram4.sampling_utils import (
    LogitNormalSchedule,
    build_guidance_schedule,
    build_packed_ids,
    get_noise,
    get_schedule_for_resolution,
    latent_grid,
    make_step_intervals,
    pack_latents,
    resolve_step_range,
)
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


@invocation_output("ideogram4_denoise_output")
class Ideogram4DenoiseOutput(BaseInvocationOutput):
    """Ideogram 4 denoise output: packed latents for the Ideogram 4 VAE decode node."""

    latents: LatentsField = OutputField(description="Packed Ideogram 4 latents.", title="Latents")
    width: int = OutputField(description="The image width in pixels.", title="Width")
    height: int = OutputField(description="The image height in pixels.", title="Height")


@invocation(
    "ideogram4_denoise",
    title="Ideogram 4 Denoise",
    tags=["image", "ideogram4", "ideogram", "denoise"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4DenoiseInvocation(BaseInvocation):
    """Run the Ideogram 4 denoising loop.

    Ideogram 4 uses asymmetric CFG: the conditional transformer sees the full text+image sequence and
    the unconditional transformer sees the image tokens only with zeroed text conditioning. Their
    velocity predictions are blended per step.
    """

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    unconditional_transformer: TransformerField = InputField(
        description="The unconditional transformer for the asymmetric-CFG negative pass.",
        input=Input.Connection,
        title="Unconditional Transformer",
    )
    conditioning: Ideogram4ConditioningField = InputField(
        description="Positive conditioning from the Ideogram 4 text encoder.",
        input=Input.Connection,
        title="Conditioning",
    )
    latents: Optional[LatentsField] = InputField(
        default=None,
        description="Packed init latents for image-to-image (from the Ideogram 4 image-to-latents node).",
        input=Input.Connection,
        title="Init Latents",
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
        title="Denoise Mask",
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise to the init latents based on denoising start.")
    width: int = InputField(default=1024, multiple_of=16, description="Image width in pixels.")
    height: int = InputField(default=1024, multiple_of=16, description="Image height in pixels.")
    num_steps: int = InputField(default=20, ge=1, le=200, description="Number of denoising steps.")
    guidance_scale: float = InputField(
        default=7.0, ge=0.0, description="Classifier-free guidance weight for the high-noise sampling steps."
    )
    polish_guidance_scale: float = InputField(
        default=3.0,
        ge=0.0,
        description="Lower guidance weight applied during the final polish steps to refine detail.",
    )
    polish_steps: int = InputField(
        default=2, ge=0, description="Number of final steps to run at the polish guidance weight."
    )
    schedule_mean: float = InputField(
        default=0.0, description="Mean of the logit-normal schedule (before resolution shift)."
    )
    schedule_std: float = InputField(default=1.75, gt=0.0, description="Std of the logit-normal schedule.")
    seed: int = InputField(default=0, description="Seed for the initial noise.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Ideogram4DenoiseOutput:
        device = TorchDevice.choose_torch_device()

        text_features = context.tensors.load(self.conditioning.conditioning_name).to(device)
        num_text_tokens = text_features.shape[1]

        grid_h, grid_w = latent_grid(self.height, self.width)
        num_image_tokens = grid_h * grid_w

        position_ids, segment_ids, indicator = build_packed_ids(
            text_lengths=[num_text_tokens],
            grid_h=grid_h,
            grid_w=grid_w,
            max_text_tokens=num_text_tokens,
            device=device,
        )

        noise = get_noise(batch_size=1, num_image_tokens=num_image_tokens, seed=self.seed, device=device)

        schedule = get_schedule_for_resolution(
            self.height, self.width, known_mean=self.schedule_mean, std=self.schedule_std
        )
        step_intervals = make_step_intervals(self.num_steps, device)
        start_step, end_step = resolve_step_range(self.num_steps, self.denoising_start, self.denoising_end)

        # Init latents (packed, already in the normalised denoise space) for image-to-image / inpaint.
        init_latents: Optional[torch.Tensor] = None
        if self.latents is not None:
            init_latents = context.tensors.load(self.latents.latents_name).to(device=device, dtype=torch.float32)

        z = self._prepare_start_latents(noise, init_latents, schedule, step_intervals, start_step)

        # Masked denoising (inpaint / outpaint): re-pin the unmasked region to the init image each step.
        inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None
        if self.denoise_mask is not None:
            assert init_latents is not None, "Ideogram 4 inpainting requires init latents."
            mask = self._prep_inpaint_mask(context, grid_h, grid_w, device)
            inpaint_extension = RectifiedFlowInpaintExtension(init_latents=init_latents, inpaint_mask=mask, noise=noise)

        guidance_schedule = build_guidance_schedule(
            num_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            polish_guidance_scale=self.polish_guidance_scale,
            polish_steps=self.polish_steps,
        )
        guidance_weights = torch.tensor(guidance_schedule, dtype=torch.float32, device=device)

        if start_step >= end_step:
            # Nothing to denoise (e.g. denoising_start == denoising_end): return the prepared latents.
            latents = z
        else:
            with ExitStack() as exit_stack:
                (_, conditional_transformer) = exit_stack.enter_context(
                    context.models.load(self.transformer.transformer).model_on_device()
                )
                (_, unconditional_transformer) = exit_stack.enter_context(
                    context.models.load(self.unconditional_transformer.transformer).model_on_device()
                )

                latents = denoise(
                    conditional_transformer=conditional_transformer,
                    unconditional_transformer=unconditional_transformer,
                    text_features=text_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    max_text_tokens=num_text_tokens,
                    z=z,
                    schedule=schedule,
                    step_intervals=step_intervals,
                    guidance_weights=guidance_weights,
                    num_steps=self.num_steps,
                    step_callback=self._build_step_callback(context),
                    start_step=start_step,
                    end_step=end_step,
                    inpaint_extension=inpaint_extension,
                )

        TorchDevice.empty_cache()
        name = context.tensors.save(latents.cpu())
        return Ideogram4DenoiseOutput(
            latents=LatentsField(latents_name=name),
            width=self.width,
            height=self.height,
        )

    def _prepare_start_latents(
        self,
        noise: torch.Tensor,
        init_latents: Optional[torch.Tensor],
        schedule: LogitNormalSchedule,
        step_intervals: torch.Tensor,
        start_step: int,
    ) -> torch.Tensor:
        """Build the starting latents: pure noise for txt2img, or noised init latents for img2img."""
        if init_latents is None:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when no init latents are provided.")
            return noise

        if not self.add_noise:
            return init_latents

        # Noise the init latents to the model time at the first run step (the inverse of the txt2img
        # trajectory z(t) = (1 - t) * noise + t * data). start_step's model time is the interval at
        # index num_steps - start_step.
        if start_step >= self.num_steps:
            return init_latents
        i_begin = self.num_steps - start_step
        t_begin = float(schedule(step_intervals[i_begin].unsqueeze(0)).item())
        return (1.0 - t_begin) * noise + t_begin * init_latents

    def _prep_inpaint_mask(
        self, context: InvocationContext, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        """Load the denoise mask and pack it into the (B, num_image_tokens, 128) latent token layout."""
        assert self.denoise_mask is not None
        mask = context.tensors.load(self.denoise_mask.mask_name)
        # The mask marks regions to KEEP; the inpaint extension expects 1 = inpaint, so invert.
        mask = 1.0 - mask
        latent_h, latent_w = grid_h * 2, grid_w * 2
        mask = tv_resize(
            img=mask,
            size=[latent_h, latent_w],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        mask = mask.to(device=device, dtype=torch.float32)
        # Broadcast across the AE channels, then pack to match the latent token layout.
        mask = mask.expand(mask.shape[0], AE_CHANNELS, latent_h, latent_w)
        return pack_latents(mask)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.signal_progress(
                f"Denoising step {state.step}/{state.total_steps}",
                state.step / max(state.total_steps, 1),
            )

        return step_callback
