from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sourceanchor.config import ExperimentConfig, MethodConfig, RoiConfig
from sourceanchor.inversion import DDIMInversionBackend, InversionOutput
from sourceanchor.method.dynamic_mask import DynamicMaskBuilder
from sourceanchor.method.prompt_handling import (
    PromptConditionSourcePredictor,
    PromptEncoder,
    TargetPromptPredictor,
    aggregate_step_cross_attention,
    build_full_target_token_mask,
)
from sourceanchor.method.source_anchor import build_anchor_mask, build_effective_mask
from sourceanchor.outputs import make_run_dir, roi_cache_dir, sample_dir, should_save_overview, should_save_roi_cache, write_json
from sourceanchor.outputs.writer import compose_simple_overview, save_image
from sourceanchor.roi.cache import load_cached_roi
from sourceanchor.roi.live import generate_live_roi
from sourceanchor.roi.serialization import save_roi_payload
from sourceanchor.runtime.device import clear_cuda_memory
from sourceanchor.runtime.model_loader import configure_inversion_module, create_native_inversion_module
from sourceanchor.runtime.pipeline import build_main_pipeline, build_shared_diffedit_pipeline
from sourceanchor.schemas import StandardSample
from sourceanchor.utils.image import load_rgb_image


@dataclass
class EditArtifacts:
    sample_id: str
    sample_output_dir: Path
    edited_image_path: Path
    source_reconstruction_path: Path
    overview_path: Path | None
    roi_soft_path: Path | None
    roi_hard_path: Path | None
    debug_json_path: Path


def build_run_manifest_payload(config: ExperimentConfig, artifacts: list[EditArtifacts]) -> dict:
    return {
        "method": config.method.name,
        "roi_source": config.roi.source,
        "roi_cache_root": str(config.roi.cache_root) if config.roi.cache_root is not None else None,
        "sample_count": len(artifacts),
        "samples": [
            {
                "sample_id": item.sample_id,
                "sample_output_dir": str(item.sample_output_dir),
                "sample_json_path": str(item.sample_output_dir / "sample.json"),
                "edited_image_path": str(item.edited_image_path),
                "source_reconstruction_path": str(item.source_reconstruction_path),
                "overview_path": str(item.overview_path) if item.overview_path is not None else None,
                "roi_soft_path": str(item.roi_soft_path) if item.roi_soft_path is not None else None,
                "roi_hard_path": str(item.roi_hard_path) if item.roi_hard_path is not None else None,
                "debug_json_path": str(item.debug_json_path),
            }
            for item in artifacts
        ],
    }


class SourceAnchorEditor:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.pipe = build_main_pipeline(config.models, config.runtime)
        self._default_attn_processors = dict(self.pipe.unet.attn_processors)
        # Store default processors on pipe for access by register_attention_control
        self.pipe._default_attn_processors = self._default_attn_processors
        self.diffedit_pipe = build_shared_diffedit_pipeline(self.pipe, config.runtime, config.method)
        self.prompt_encoder = PromptEncoder(self.pipe)
        self.source_predictor = PromptConditionSourcePredictor(self.pipe)
        self.target_predictor = TargetPromptPredictor(self.pipe, self.prompt_encoder, config.method.guidance_scale)
        self.inversion_backend = DDIMInversionBackend(self.pipe, config.models, config.method)
        self.dynamic_mask_builder = DynamicMaskBuilder(config.method)
        self.ntip2p = create_native_inversion_module()
        configure_inversion_module(self.ntip2p, self.pipe, config.method)
        self.attention_store = self.ntip2p.AttentionStore()
        self._register_attention_store()

    def _restore_default_attention_processors(self) -> None:
        self.pipe.unet.set_attn_processor(dict(self._default_attn_processors))

    def _register_attention_store(self) -> None:
        self.ntip2p.ptp_utils.register_attention_control(self.pipe, self.attention_store)

    @staticmethod
    def _resample_source_latents(source_latents: list[torch.Tensor], target_count: int) -> list[torch.Tensor]:
        if target_count <= 0:
            return []
        if not source_latents:
            raise ValueError("source_latents must not be empty")
        if len(source_latents) == target_count:
            return source_latents
        if len(source_latents) == 1:
            return [source_latents[0].clone() for _ in range(target_count)]
        aligned: list[torch.Tensor] = []
        for target_index in range(target_count):
            position = target_index * (len(source_latents) - 1) / max(target_count - 1, 1)
            lower = int(np.floor(position))
            upper = int(np.ceil(position))
            if lower == upper:
                aligned.append(source_latents[lower].clone())
                continue
            weight = float(position - lower)
            aligned.append(source_latents[lower] * (1.0 - weight) + source_latents[upper] * weight)
        return aligned

    def _set_timesteps(self) -> None:
        try:
            self.pipe.scheduler.set_timesteps(self.config.method.num_edit_steps, device=self.pipe.device)
        except TypeError:
            self.pipe.scheduler.set_timesteps(self.config.method.num_edit_steps)

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        scaled = latents / 0.18215
        image = self.pipe.vae.decode(scaled).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image[0]

    def _load_roi(self, sample: StandardSample) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self.config.roi.source == "cache":
            if self.config.roi.cache_root is None:
                raise ValueError("roi.cache_root must be configured when roi.source=cache")
            roi_payload = load_cached_roi(self.config.roi.cache_root, sample.sample_id)
        else:
            roi_payload = self._generate_live_roi_payload(sample)
            if should_save_roi_cache(self.config.save) and self.config.roi.cache_root is not None:
                save_roi_payload(roi_cache_dir(self.config.roi.cache_root, sample.sample_id), roi_payload)

        soft_roi = np.clip(np.asarray(roi_payload.mask, dtype=np.float32), 0.0, 1.0)
        hard_roi = (soft_roi > float(self.config.roi.threshold)).astype(np.float32)
        dtype = self.pipe.unet.dtype
        device = self.pipe.device
        soft_roi_tensor = torch.from_numpy(soft_roi).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        hard_roi_tensor = torch.from_numpy(hard_roi).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        return soft_roi_tensor, hard_roi_tensor, dict(roi_payload.metadata)

    def _generate_live_roi_payload(self, sample: StandardSample):
        self._restore_default_attention_processors()
        roi_payload = generate_live_roi(self.diffedit_pipe, sample, self.config.method, self.config.roi)
        self._register_attention_store()
        return roi_payload

    def _save_roi_visuals(self, sample_output_dir: Path, soft_roi: torch.Tensor, hard_roi: torch.Tensor) -> tuple[Path, Path]:
        soft_path = sample_output_dir / "roi_soft.png"
        hard_path = sample_output_dir / "roi_hard.png"
        save_image(soft_path, np.uint8(np.clip(soft_roi[0, 0].detach().cpu().numpy(), 0.0, 1.0) * 255.0))
        save_image(hard_path, np.uint8(np.clip(hard_roi[0, 0].detach().cpu().numpy(), 0.0, 1.0) * 255.0))
        return soft_path, hard_path

    def run_sample(self, sample: StandardSample, run_dir: Path) -> EditArtifacts:
        sample_output_dir = sample_dir(run_dir, sample.sample_id)
        write_json(sample_output_dir / "sample.json", sample.to_dict())
        source_image = load_rgb_image(sample.source_image_path)
        inversion = self.inversion_backend.invert(source_image, source_prompt=sample.source_prompt)
        source_reconstruction_path = save_image(sample_output_dir / "source_reconstruction.png", inversion.reconstruction_image)
        save_image(sample_output_dir / "source.png", source_image)

        soft_roi, hard_roi, roi_metadata = self._load_roi(sample)
        roi_soft_path, roi_hard_path = self._save_roi_visuals(sample_output_dir, soft_roi, hard_roi)

        target_condition = self.prompt_encoder.encode(sample.target_prompt)
        source_condition = self.prompt_encoder.encode(sample.source_prompt)
        focus_mask = build_full_target_token_mask(target_condition).to(self.pipe.device)

        self._set_timesteps()
        latents = inversion.zt_src.detach().clone().to(self.pipe.device, dtype=self.pipe.unet.dtype)
        source_latents = self._resample_source_latents(
            [latent.to(self.pipe.device, dtype=self.pipe.unet.dtype) for latent in inversion.src_latents],
            len(self.pipe.scheduler.timesteps),
        )
        target_embeddings = target_condition.embeddings.to(self.pipe.device, dtype=self.pipe.unet.dtype)
        source_embeddings = source_condition.embeddings.to(self.pipe.device, dtype=self.pipe.unet.dtype)

        step_debug: list[dict[str, float]] = []
        for step_idx, timestep in enumerate(self.pipe.scheduler.timesteps):
            source_latent = source_latents[step_idx]
            self.attention_store.reset()
            eps_src = self.source_predictor.predict(latents, timestep, source_embeddings)
            self.attention_store.reset()
            eps_tar, target_noise, _noise_uncond, target_stats = self.target_predictor.predict(
                latents,
                timestep,
                target_embeddings,
            )
            attention_map = aggregate_step_cross_attention(
                self.attention_store,
                focus_mask,
                target_hw=(latents.shape[-2], latents.shape[-1]),
                locations=self.config.method.attention_locations,
            )
            dynamic_mask, aux = self.dynamic_mask_builder.build(
                eps_src,
                target_noise,
                latents,
                source_latent,
                attention_map,
            )
            effective_mask = build_effective_mask(
                dynamic_mask,
                hard_roi,
                soft_roi,
                self.config.method,
                step_idx,
                len(self.pipe.scheduler.timesteps),
            )
            eps = eps_src + effective_mask * (eps_tar - eps_src)
            prev_latents = self.pipe.scheduler.step(eps, timestep, latents).prev_sample
            next_source_idx = min(step_idx + 1, len(source_latents) - 1)
            anchor_mask = build_anchor_mask(
                hard_roi,
                soft_roi,
                self.config.method,
                step_idx,
                len(self.pipe.scheduler.timesteps),
            )
            latents = anchor_mask * prev_latents + (1.0 - anchor_mask) * source_latents[next_source_idx]
            step_debug.append(
                {
                    "step_idx": float(step_idx),
                    "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
                    "delta": float(target_stats["delta"]),
                    "dynamic_mask_mean": float(dynamic_mask.mean().item()),
                    "effective_mask_mean": float(effective_mask.mean().item()),
                    "soft_roi_mean": float(soft_roi.mean().item()),
                    "hard_roi_mean": float(hard_roi.mean().item()),
                }
            )

        edited_image = self._decode_latents(latents)
        edited_image_path = save_image(sample_output_dir / "edited.png", edited_image)

        overview_path: Path | None = None
        if should_save_overview(self.config.save):
            overview = compose_simple_overview(
                [
                    ("source", source_image),
                    ("reconstruction", inversion.reconstruction_image),
                    ("edited", edited_image),
                ],
                columns=3,
            )
            overview_path = save_image(sample_output_dir / "overview.png", overview)

        debug_json_path = write_json(
            sample_output_dir / "debug.json",
            {
                "sample_id": sample.sample_id,
                "source_prompt": sample.source_prompt,
                "target_prompt": sample.target_prompt,
                "roi_source": self.config.roi.source,
                "roi_metadata": roi_metadata,
                "inversion": inversion.metadata,
                "steps": step_debug,
            },
        )
        clear_cuda_memory()
        return EditArtifacts(
            sample_id=sample.sample_id,
            sample_output_dir=sample_output_dir,
            edited_image_path=edited_image_path,
            source_reconstruction_path=source_reconstruction_path,
            overview_path=overview_path,
            roi_soft_path=roi_soft_path,
            roi_hard_path=roi_hard_path,
            debug_json_path=debug_json_path,
        )


def run_experiment(config: ExperimentConfig, samples: list[StandardSample]) -> Path:
    run_dir = make_run_dir(config.output_root, prefix=config.method.name)
    write_json(run_dir / "experiment_config.json", config.to_serializable_dict())
    editor = SourceAnchorEditor(config)
    artifacts = []
    for sample in samples:
        artifacts.append(editor.run_sample(sample, run_dir))
    write_json(run_dir / "run_manifest.json", build_run_manifest_payload(config, artifacts))
    return run_dir
