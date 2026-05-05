from __future__ import annotations

import torch
import torch.nn.functional as F

from sourceanchor.config import MethodConfig


def _normalize_tensor_map(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_value) / (max_value - min_value).clamp(min=eps)


class DynamicMaskBuilder:
    def __init__(self, method: MethodConfig) -> None:
        self.method = method

    def _compute_aux_maps(
        self,
        reference_noise: torch.Tensor,
        target_noise: torch.Tensor,
        latents: torch.Tensor,
        source_latent: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        discrepancy = torch.abs(target_noise - reference_noise).mean(dim=1, keepdim=True)
        discrepancy = _normalize_tensor_map(discrepancy)

        if attention_map.ndim == 3:
            attention_map = attention_map.unsqueeze(0)
        attention_map = attention_map.to(device=reference_noise.device, dtype=reference_noise.dtype)
        attention_map = _normalize_tensor_map(attention_map)

        latent_drift = torch.abs(latents - source_latent).mean(dim=1, keepdim=True)
        latent_drift = _normalize_tensor_map(latent_drift)

        return {
            "discrepancy": discrepancy,
            "attention": attention_map,
            "latent_drift": latent_drift,
        }

    def build(
        self,
        reference_noise: torch.Tensor,
        target_noise: torch.Tensor,
        latents: torch.Tensor,
        source_latent: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux = self._compute_aux_maps(reference_noise, target_noise, latents, source_latent, attention_map)
        discrepancy = aux["discrepancy"]
        attention_map = aux["attention"]
        latent_drift = aux["latent_drift"]

        raw_mask = (
            self.method.discrepancy_weight * discrepancy
            + self.method.attention_weight * attention_map
            - self.method.latent_weight * latent_drift
        )
        if self.method.smoothing_kernel > 1:
            kernel = self.method.smoothing_kernel
            padding = kernel // 2
            raw_mask = F.avg_pool2d(raw_mask, kernel_size=kernel, stride=1, padding=padding)

        mask = torch.sigmoid((raw_mask - self.method.threshold) * self.method.temperature)
        mask = torch.clamp(mask, min=self.method.min_value, max=self.method.max_value)
        mask = mask.to(device=reference_noise.device, dtype=reference_noise.dtype)
        aux["mask"] = mask
        return mask, aux
