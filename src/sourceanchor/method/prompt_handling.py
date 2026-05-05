from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TextCondition:
    prompt: str
    embeddings: torch.Tensor
    input_ids: torch.Tensor
    token_mask: torch.Tensor


class PromptEncoder:
    def __init__(self, pipe) -> None:
        self.pipe = pipe
        self.cache: dict[str, TextCondition] = {}

    @torch.no_grad()
    def encode(self, prompt: str) -> TextCondition:
        if prompt in self.cache:
            return self.cache[prompt]
        tokenizer = self.pipe.tokenizer
        text_input = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input.input_ids.to(self.pipe.device)
        embeddings = self.pipe.text_encoder(input_ids)[0]
        token_mask = input_ids != tokenizer.pad_token_id
        if token_mask.shape[-1] > 0:
            token_mask[:, 0] = False
        if tokenizer.eos_token_id is not None:
            token_mask = token_mask & (input_ids != tokenizer.eos_token_id)
        condition = TextCondition(
            prompt=prompt,
            embeddings=embeddings.detach(),
            input_ids=input_ids.detach(),
            token_mask=token_mask.detach(),
        )
        self.cache[prompt] = condition
        return condition


class PromptConditionSourcePredictor:
    def __init__(self, pipe) -> None:
        self.pipe = pipe

    @torch.no_grad()
    def predict(self, latents: torch.Tensor, timestep: torch.Tensor, source_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = latents.shape[0]
        if source_embeddings.shape[0] != batch_size:
            raise ValueError(f"source_embeddings batch mismatch: expected {batch_size}, got {source_embeddings.shape[0]}")
        return self.pipe.unet(latents, timestep, encoder_hidden_states=source_embeddings).sample


class TargetPromptPredictor:
    def __init__(self, pipe, prompt_encoder: PromptEncoder, guidance_scale: float) -> None:
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.uncond_condition = prompt_encoder.encode("")

    @torch.no_grad()
    def predict(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | list[float]]]:
        batch_size = latents.shape[0]
        if target_embeddings.shape[0] != batch_size:
            raise ValueError(f"target_embeddings batch mismatch: expected {batch_size}, got {target_embeddings.shape[0]}")
        latents_input = torch.cat([latents, latents], dim=0)
        uncond_embeddings = self.uncond_condition.embeddings.expand(batch_size, -1, -1)
        context = torch.cat([uncond_embeddings, target_embeddings], dim=0)
        noise = self.pipe.unet(latents_input, timestep, encoder_hidden_states=context).sample
        noise_uncond, noise_cond = noise.chunk(2, dim=0)
        delta_per_sample = (noise_cond - noise_uncond).abs().flatten(1).mean(dim=1)
        guided_noise = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
        return guided_noise, noise_cond, noise_uncond, {
            "delta": float(delta_per_sample.mean().item()),
            "delta_per_sample": [float(value) for value in delta_per_sample.detach().cpu().tolist()],
        }


def build_full_target_token_mask(target_condition: TextCondition) -> torch.Tensor:
    return target_condition.token_mask


def aggregate_step_cross_attention(attention_store, token_mask: torch.Tensor, target_hw: tuple[int, int], locations: tuple[str, ...]) -> torch.Tensor:
    averaged = attention_store.get_average_attention()
    maps: list[torch.Tensor] = []
    device = token_mask.device
    if token_mask.ndim == 1:
        token_mask = token_mask.unsqueeze(0)
    token_mask = token_mask.to(device=device, dtype=torch.float32)
    empty_rows = token_mask.sum(dim=-1, keepdim=True) <= 0
    if empty_rows.any():
        token_mask = token_mask.clone()
        token_mask[empty_rows.expand_as(token_mask)] = 1.0
    batch_size = int(token_mask.shape[0])
    for location in locations:
        for item in averaged.get(f"{location}_cross", []):
            pixel_count = item.shape[1]
            resolution = int(math.sqrt(pixel_count))
            if resolution * resolution != pixel_count:
                continue
            if item.shape[0] % batch_size != 0:
                continue
            reshaped = item.reshape(batch_size, -1, resolution, resolution, item.shape[-1])
            token_weights = token_mask.to(reshaped.device)[:, None, None, None, :]
            token_map = (reshaped * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp(min=1.0)
            pooled = token_map.mean(dim=1, keepdim=True)
            pooled = F.interpolate(
                pooled,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            maps.append(pooled)
    if not maps:
        return torch.zeros((batch_size, 1, *target_hw), device=device, dtype=torch.float32)
    return torch.stack(maps, dim=0).mean(dim=0).to(device=device, dtype=torch.float32)
