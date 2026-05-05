from __future__ import annotations

import torch


def resolve_device(device_name: str) -> str:
    requested = str(device_name).strip() or "cuda"
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    if requested.startswith("cuda"):
        return "cpu"
    return requested


def resolve_torch_dtype(dtype_name: str, *, device: str) -> torch.dtype:
    if device.startswith("cuda") and str(dtype_name).lower() == "float16":
        return torch.float16
    return torch.float32


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
