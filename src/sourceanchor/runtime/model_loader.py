from __future__ import annotations

import importlib
import os
import sys
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


def resolve_ntip2p_root(models: ModelConfig) -> Path:
    if models.ntip2p_root is None:
        raise ValueError("models.ntip2p_root must be configured for the standalone release.")
    root = Path(models.ntip2p_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"NTIP2P root not found: {root}")
    return root


def load_ntip2p_module(models: ModelConfig):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ntip2p_root = resolve_ntip2p_root(models)
    if str(ntip2p_root) not in sys.path:
        sys.path.insert(0, str(ntip2p_root))
    return importlib.import_module("null_text_w_ptp")


def configure_ntip2p_module(module, pipe, method: MethodConfig) -> None:
    pipe_device = getattr(pipe, "_execution_device", pipe.device)
    module.device = torch.device(pipe_device)
    module.tokenizer = pipe.tokenizer
    module.prompts = []
    module.NUM_DDIM_STEPS = method.num_inversion_steps
    module.GUIDANCE_SCALE = method.guidance_scale
    module.LOW_RESOURCE = False


def execution_device_for_pipe(pipe, requested_device: str) -> str:
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is None:
        return resolve_device(requested_device)
    return str(execution_device)
