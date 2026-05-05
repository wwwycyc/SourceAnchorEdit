from __future__ import annotations

import torch
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionDiffEditPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from sourceanchor.config import MethodConfig, ModelConfig, RuntimeConfig
from sourceanchor.runtime.device import resolve_device, resolve_torch_dtype
from sourceanchor.runtime.model_loader import resolve_model_source


def _configure_torch_backends(runtime: RuntimeConfig, device: str) -> None:
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = runtime.enable_tf32
        torch.backends.cudnn.allow_tf32 = runtime.enable_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if runtime.enable_tf32 else "highest")


def build_main_pipeline(models: ModelConfig, runtime: RuntimeConfig) -> StableDiffusionPipeline:
    model_source = resolve_model_source(models.sd_model)
    device = resolve_device(runtime.device)
    dtype = resolve_torch_dtype(runtime.dtype, device=device)
    _configure_torch_backends(runtime, device)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_source,
        subfolder="tokenizer",
        local_files_only=runtime.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_source,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=runtime.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        model_source,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=runtime.local_files_only,
        use_safetensors=False,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_source,
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=runtime.local_files_only,
        use_safetensors=False,
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_source,
        subfolder="scheduler",
        local_files_only=runtime.local_files_only,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    if runtime.enable_cpu_offload and torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception as exc:
            print(f"[warn] cpu offload unavailable, fallback to device placement: {exc}")
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    if runtime.channels_last and device.startswith("cuda"):
        pipe.unet.to(memory_format=torch.channels_last)
        try:
            pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if runtime.attention_slicing:
        pipe.enable_attention_slicing()
    if runtime.vae_slicing:
        try:
            pipe.vae.enable_slicing()
        except AttributeError:
            pass
    if runtime.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def build_shared_diffedit_pipeline(
    pipe: StableDiffusionPipeline,
    runtime: RuntimeConfig,
    method: MethodConfig,
) -> StableDiffusionDiffEditPipeline:
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, set_alpha_to_one=False)
    inverse_scheduler = DDIMInverseScheduler.from_config(
        scheduler.config,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    diffedit_pipe = StableDiffusionDiffEditPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        inverse_scheduler=inverse_scheduler,
        requires_safety_checker=False,
    )
    execution_device = getattr(pipe, "_execution_device", pipe.device)
    diffedit_pipe = diffedit_pipe.to(execution_device)
    if runtime.attention_slicing:
        diffedit_pipe.enable_attention_slicing()
    if runtime.vae_slicing:
        try:
            diffedit_pipe.vae.enable_slicing()
        except AttributeError:
            pass
    try:
        diffedit_pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return diffedit_pipe
