from __future__ import annotations

import os
from pathlib import Path

import torch

from sourceanchor.config import MethodConfig, ModelConfig
from sourceanchor.runtime.device import resolve_device


def resolve_model_source(value: str | Path | None) -> str:
    if value is None:
        raise ValueError("Model source is not configured.")
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return str(value)


class NativeInversionModule:
    """Native implementation module providing NTIP2P-compatible interface.

    This module replaces the external NTIP2P dependency with native implementations
    using diffusers APIs.
    """

    def __init__(self):
        # Module-level state (configured externally)
        self.device = None
        self.tokenizer = None
        self.prompts = []
        self.NUM_DDIM_STEPS = 50
        self.GUIDANCE_SCALE = 7.5
        self.LOW_RESOURCE = False

        # Import classes lazily to avoid circular imports
        from sourceanchor.runtime.attention import AttentionStore, ptp_utils
        from sourceanchor.inversion.native_inversion import NativeInversion

        # Export classes
        self.AttentionStore = AttentionStore
        self.NullInversion = NativeInversion

        # Export utilities
        self.ptp_utils = ptp_utils


def create_native_inversion_module() -> NativeInversionModule:
    """Factory function to create native inversion module.

    Returns:
        NativeInversionModule instance with NTIP2P-compatible interface
    """
    return NativeInversionModule()


def configure_inversion_module(module: NativeInversionModule, pipe, method: MethodConfig) -> None:
    """Configure module with pipeline and method settings.

    Args:
        module: NativeInversionModule instance to configure
        pipe: StableDiffusionPipeline with components
        method: Method configuration with inversion/guidance parameters
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    pipe_device = getattr(pipe, "_execution_device", pipe.device)
    module.device = torch.device(pipe_device)
    module.tokenizer = pipe.tokenizer
    module.prompts = []
    module.NUM_DDIM_STEPS = method.num_inversion_steps
    module.GUIDANCE_SCALE = method.guidance_scale
    module.LOW_RESOURCE = False

    # Store configuration on pipeline for access in NativeInversion
    pipe._num_ddim_steps = method.num_inversion_steps
    pipe._guidance_scale = method.guidance_scale


def execution_device_for_pipe(pipe, requested_device: str) -> str:
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is None:
        return resolve_device(requested_device)
    return str(execution_device)
