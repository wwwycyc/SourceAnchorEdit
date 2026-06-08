from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sourceanchor.config import MethodConfig, ModelConfig
from sourceanchor.runtime.model_loader import configure_inversion_module, create_native_inversion_module


@dataclass
class InversionOutput:
    zt_src: torch.Tensor
    src_latents: list[torch.Tensor]
    reconstruction_image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class DDIMInversionBackend:
    def __init__(self, pipe, models: ModelConfig, method: MethodConfig) -> None:
        self.pipe = pipe
        self.models = models
        self.method = method
        self.ntip2p = create_native_inversion_module()
        configure_inversion_module(self.ntip2p, pipe, method)

    @torch.no_grad()
    def invert(self, source_image: np.ndarray, source_prompt: str | None = None) -> InversionOutput:
        prompt = (source_prompt or "").strip()
        self.ntip2p.ptp_utils.register_attention_control(self.pipe, None)
        inversion = self.ntip2p.NullInversion(self.pipe)
        inversion.init_prompt(prompt)
        reconstruction_image, ddim_latents = inversion.ddim_inversion(source_image)
        src_latents = [latent.detach().clone() for latent in reversed(ddim_latents[1:])]
        inversion_timesteps = [
            int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            for timestep in self.pipe.scheduler.timesteps
        ]
        return InversionOutput(
            zt_src=ddim_latents[-1].detach().clone(),
            src_latents=src_latents,
            reconstruction_image=reconstruction_image,
            metadata={
                "backend": "ddim",
                "source_prompt": prompt,
                "source_prompt_used_for_inversion": bool(prompt),
                "inversion_prompt_mode": "source_prompt" if prompt else "empty",
                "num_inversion_steps": self.method.num_inversion_steps,
                "inversion_timesteps": inversion_timesteps,
            },
        )
