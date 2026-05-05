from __future__ import annotations

import math

import torch

from sourceanchor.config import MethodConfig


def schedule_progress(step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return max(0.0, min(float(step_idx) / float(total_steps - 1), 1.0))


def cosine_schedule(start: float, end: float, progress: float) -> float:
    clipped = max(0.0, min(progress, 1.0))
    eased = 0.5 - 0.5 * math.cos(math.pi * clipped)
    return start + (end - start) * eased


def cosine_gate(step_idx: int, total_steps: int, *, start_ratio: float, full_ratio: float) -> float:
    if full_ratio < start_ratio:
        raise ValueError("full_ratio must be >= start_ratio")
    progress = schedule_progress(step_idx, total_steps)
    if progress <= start_ratio:
        return 0.0
    if progress >= full_ratio:
        return 1.0
    span = max(full_ratio - start_ratio, 1e-6)
    local_progress = (progress - start_ratio) / span
    return cosine_schedule(0.0, 1.0, local_progress)


def soft_roi_weight(method: MethodConfig, step_idx: int, total_steps: int) -> float:
    return cosine_schedule(
        method.soft_roi_start_weight,
        method.soft_roi_end_weight,
        schedule_progress(step_idx, total_steps),
    )


def anchor_hardness(method: MethodConfig, step_idx: int, total_steps: int) -> float:
    return cosine_schedule(
        method.anchor_hardness_start,
        method.anchor_hardness_end,
        schedule_progress(step_idx, total_steps),
    )


def anchor_enable_gate(method: MethodConfig, step_idx: int, total_steps: int) -> float:
    if method.source_anchor_start <= 0.0:
        return 1.0
    return cosine_gate(step_idx, total_steps, start_ratio=method.source_anchor_start, full_ratio=1.0)


def build_effective_mask(
    dynamic_mask: torch.Tensor,
    hard_roi_mask: torch.Tensor,
    soft_roi_mask: torch.Tensor,
    method: MethodConfig,
    step_idx: int,
    total_steps: int,
) -> torch.Tensor:
    support_evidence = (hard_roi_mask * dynamic_mask).clamp(0.0, 1.0)
    return torch.lerp(support_evidence, soft_roi_mask, soft_roi_weight(method, step_idx, total_steps)).clamp(0.0, 1.0)


def build_anchor_mask(
    hard_roi_mask: torch.Tensor,
    soft_roi_mask: torch.Tensor,
    method: MethodConfig,
    step_idx: int,
    total_steps: int,
) -> torch.Tensor:
    adaptive_mask = torch.lerp(soft_roi_mask, hard_roi_mask, anchor_hardness(method, step_idx, total_steps)).clamp(0.0, 1.0)
    gate = anchor_enable_gate(method, step_idx, total_steps)
    if gate >= 1.0:
        return adaptive_mask
    return torch.lerp(torch.ones_like(adaptive_mask), adaptive_mask, gate).clamp(0.0, 1.0)
