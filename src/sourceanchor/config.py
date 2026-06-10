from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MethodConfig:
    name: str = "source_anchor"
    no_target_hints: bool = True
    source_anchor_start: float = 0.25
    num_inversion_steps: int = 10
    num_edit_steps: int = 10
    guidance_scale: float = 7.5
    soft_roi_start_weight: float = 0.75
    soft_roi_end_weight: float = 0.10
    anchor_hardness_start: float = 0.35
    anchor_hardness_end: float = 1.0
    discrepancy_weight: float = 0.55
    attention_weight: float = 0.30
    latent_weight: float = 0.15
    temperature: float = 8.0
    threshold: float = 0.35
    min_value: float = 0.0
    max_value: float = 1.0
    smoothing_kernel: int = 5
    attention_locations: tuple[str, ...] = ("down", "mid", "up")


@dataclass
class RoiConfig:
    source: str = "live"
    cache_root: Path | None = None
    save_cache: bool = False
    threshold: float = 0.5
    num_maps_per_mask: int = 1
    mask_encode_strength: float = 0.5
    mask_thresholding_ratio: float = 3.0


@dataclass
class InversionConfig:
    source: str = "live"
    cache_root: Path | None = None
    save_cache: bool = False


@dataclass
class ModelConfig:
    sd_model: str | None = None
    clip_model: str | None = None
    dino_weights: str | None = None


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 1
    local_files_only: bool = True
    attention_slicing: bool = True
    vae_slicing: bool = True
    channels_last: bool = True
    enable_tf32: bool = True
    enable_cpu_offload: bool = False
    enable_xformers: bool = False


@dataclass
class SaveConfig:
    roi_cache: bool = False
    inversion_tensors: bool = False
    debug_json: bool = True
    step_visualizations: bool = False
    step_interval: int = 1
    step_indices: tuple[int, ...] = ()
    step_include_last: bool = True
    step_save_trace: bool = True
    step_save_diagnostics: bool = True
    step_save_dynamic_mask: bool = True
    step_save_effective_mask: bool = True
    step_save_anchor_mask: bool = True
    step_save_attention: bool = True
    step_save_discrepancy: bool = True
    step_save_latent_drift: bool = True
    step_save_roi: bool = False
    step_save_latent_preview: bool = False
    overview: bool = True


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    enable_metrics: bool = True
    enable_lpips: bool = True
    enable_psnr: bool = True
    enable_mse: bool = True
    enable_ssim: bool = True
    enable_clip_score: bool = True
    enable_structure_distance: bool = False
    enable_locality_ratio: bool = True
    lpips_net: str = "squeeze"
    clip_model_id: str = "openai/clip-vit-large-patch14"
    clip_local_files_only: bool = True
    dino_model_name: str = "dino_vits8"
    dino_global_patch_size: int = 224
    dino_repo_or_dir: Path | None = None
    dino_cache_dir: Path | None = None
    dino_weights_path: Path | None = None
    dino_local_files_only: bool = True


@dataclass
class ExperimentConfig:
    input_manifest: Path
    output_root: Path
    method: MethodConfig = field(default_factory=MethodConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_serializable_dict(self) -> dict[str, Any]:
        return _to_jsonable(self.to_dict())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _load_mapping_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    suffix = resolved.suffix.lower()
    raw_text = resolved.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported config format: {resolved}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {resolved}")
    return payload


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_optional_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidate_from_base = (base_dir / path).resolve()
    if candidate_from_base.exists():
        return candidate_from_base

    candidate_from_cwd = path.resolve()
    if candidate_from_cwd.exists():
        return candidate_from_cwd

    return candidate_from_base


def _load_referenced_block(
    parent_mapping: dict[str, Any],
    *,
    key: str,
    base_dir: Path,
) -> tuple[dict[str, Any], Path]:
    block = dict(parent_mapping.get(key) or {})
    reference = block.pop("config", None)
    if reference is None:
        return block, base_dir
    ref_path = _resolve_optional_path(reference, base_dir=base_dir)
    assert ref_path is not None
    referenced = _load_mapping_file(ref_path)
    merged = _merge_dicts(referenced.get(key) or {}, block)
    return merged, ref_path.parent


def load_method_config(mapping: dict[str, Any]) -> MethodConfig:
    config = MethodConfig(
        name=str(mapping.get("name", "source_anchor")),
        no_target_hints=bool(mapping.get("no_target_hints", True)),
        source_anchor_start=float(mapping.get("source_anchor_start", 0.25)),
        num_inversion_steps=int(mapping.get("num_inversion_steps", 10)),
        num_edit_steps=int(mapping.get("num_edit_steps", 10)),
        guidance_scale=float(mapping.get("guidance_scale", 7.5)),
        soft_roi_start_weight=float(mapping.get("soft_roi_start_weight", 0.75)),
        soft_roi_end_weight=float(mapping.get("soft_roi_end_weight", 0.10)),
        anchor_hardness_start=float(mapping.get("anchor_hardness_start", 0.35)),
        anchor_hardness_end=float(mapping.get("anchor_hardness_end", 1.0)),
        discrepancy_weight=float(mapping.get("discrepancy_weight", 0.55)),
        attention_weight=float(mapping.get("attention_weight", 0.30)),
        latent_weight=float(mapping.get("latent_weight", 0.15)),
        temperature=float(mapping.get("temperature", 8.0)),
        threshold=float(mapping.get("threshold", 0.35)),
        min_value=float(mapping.get("min_value", 0.0)),
        max_value=float(mapping.get("max_value", 1.0)),
        smoothing_kernel=int(mapping.get("smoothing_kernel", 5)),
        attention_locations=tuple(str(value) for value in mapping.get("attention_locations", ("down", "mid", "up"))),
    )
    if not config.no_target_hints:
        raise ValueError("The final release fixes no_target_hints=true and does not support enabling hints.")
    if not 0.0 <= config.source_anchor_start <= 1.0:
        raise ValueError(f"method.source_anchor_start must be in [0, 1], got {config.source_anchor_start}")
    if config.num_inversion_steps <= 0:
        raise ValueError("method.num_inversion_steps must be > 0")
    if config.num_edit_steps <= 0:
        raise ValueError("method.num_edit_steps must be > 0")
    return config


def load_roi_config(mapping: dict[str, Any], *, base_dir: Path) -> RoiConfig:
    source = str(mapping.get("source", "live")).strip().lower()
    if source not in {"live", "cache"}:
        raise ValueError(f"roi.source must be 'live' or 'cache', got: {source}")
    cache_root = _resolve_optional_path(mapping.get("cache_root"), base_dir=base_dir)
    config = RoiConfig(
        source=source,
        cache_root=cache_root,
        save_cache=bool(mapping.get("save_cache", False)),
        threshold=float(mapping.get("threshold", 0.5)),
        num_maps_per_mask=int(mapping.get("num_maps_per_mask", 1)),
        mask_encode_strength=float(mapping.get("mask_encode_strength", 0.5)),
        mask_thresholding_ratio=float(mapping.get("mask_thresholding_ratio", 3.0)),
    )
    if config.source == "cache" and config.cache_root is None:
        raise ValueError("roi.cache_root must be configured when roi.source=cache")
    if config.num_maps_per_mask <= 0:
        raise ValueError("roi.num_maps_per_mask must be > 0")
    return config


def load_inversion_config(mapping: dict[str, Any], *, base_dir: Path) -> InversionConfig:
    source = str(mapping.get("source", "live")).strip().lower()
    if source not in {"live", "cache"}:
        raise ValueError(f"inversion.source must be 'live' or 'cache', got: {source}")
    cache_root = _resolve_optional_path(mapping.get("cache_root"), base_dir=base_dir)
    config = InversionConfig(
        source=source,
        cache_root=cache_root,
        save_cache=bool(mapping.get("save_cache", False)),
    )
    if config.source == "cache" and config.cache_root is None:
        raise ValueError("inversion.cache_root must be configured when inversion.source=cache")
    return config


def load_model_config(mapping: dict[str, Any], *, base_dir: Path) -> ModelConfig:
    sd_model = mapping.get("sd_model")
    clip_model = mapping.get("clip_model")
    dino_weights = mapping.get("dino_weights")

    def _resolve_model_like(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate.resolve())
        candidate_from_base = (base_dir / candidate).resolve()
        if candidate_from_base.exists():
            return str(candidate_from_base)
        return raw

    return ModelConfig(
        sd_model=_resolve_model_like(sd_model),
        clip_model=_resolve_model_like(clip_model),
        dino_weights=_resolve_model_like(dino_weights),
    )


def load_runtime_config(mapping: dict[str, Any]) -> RuntimeConfig:
    config = RuntimeConfig(
        device=str(mapping.get("device", "cuda")),
        dtype=str(mapping.get("dtype", "float16")),
        batch_size=int(mapping.get("batch_size", 1)),
        local_files_only=bool(mapping.get("local_files_only", True)),
        attention_slicing=bool(mapping.get("attention_slicing", True)),
        vae_slicing=bool(mapping.get("vae_slicing", True)),
        channels_last=bool(mapping.get("channels_last", True)),
        enable_tf32=bool(mapping.get("enable_tf32", True)),
        enable_cpu_offload=bool(mapping.get("enable_cpu_offload", False)),
        enable_xformers=bool(mapping.get("enable_xformers", False)),
    )
    if config.batch_size <= 0:
        raise ValueError("runtime.batch_size must be > 0")
    return config


def load_save_config(mapping: dict[str, Any]) -> SaveConfig:
    step_indices_raw = mapping.get("step_indices", ())
    if step_indices_raw is None:
        step_indices = ()
    elif isinstance(step_indices_raw, (list, tuple)):
        step_indices = tuple(sorted({int(value) for value in step_indices_raw}))
    else:
        step_indices = (int(step_indices_raw),)
    config = SaveConfig(
        roi_cache=bool(mapping.get("roi_cache", False)),
        inversion_tensors=bool(mapping.get("inversion_tensors", False)),
        debug_json=bool(mapping.get("debug_json", True)),
        step_visualizations=bool(mapping.get("step_visualizations", False)),
        step_interval=int(mapping.get("step_interval", 1)),
        step_indices=step_indices,
        step_include_last=bool(mapping.get("step_include_last", True)),
        step_save_trace=bool(mapping.get("step_save_trace", True)),
        step_save_diagnostics=bool(mapping.get("step_save_diagnostics", True)),
        step_save_dynamic_mask=bool(mapping.get("step_save_dynamic_mask", True)),
        step_save_effective_mask=bool(mapping.get("step_save_effective_mask", True)),
        step_save_anchor_mask=bool(mapping.get("step_save_anchor_mask", True)),
        step_save_attention=bool(mapping.get("step_save_attention", True)),
        step_save_discrepancy=bool(mapping.get("step_save_discrepancy", True)),
        step_save_latent_drift=bool(mapping.get("step_save_latent_drift", True)),
        step_save_roi=bool(mapping.get("step_save_roi", False)),
        step_save_latent_preview=bool(mapping.get("step_save_latent_preview", False)),
        overview=bool(mapping.get("overview", True)),
    )
    if config.step_interval <= 0:
        raise ValueError("save.step_interval must be > 0")
    if any(index < 0 for index in config.step_indices):
        raise ValueError("save.step_indices must contain non-negative step indices")
    return config


def load_metrics_config(mapping: dict[str, Any], *, base_dir: Path) -> MetricsConfig:
    """Load metrics configuration from YAML mapping."""
    return MetricsConfig(
        enable_metrics=bool(mapping.get("enable_metrics", False)),
        enable_lpips=bool(mapping.get("enable_lpips", True)),
        enable_psnr=bool(mapping.get("enable_psnr", True)),
        enable_mse=bool(mapping.get("enable_mse", True)),
        enable_ssim=bool(mapping.get("enable_ssim", True)),
        enable_clip_score=bool(mapping.get("enable_clip_score", True)),
        enable_structure_distance=bool(mapping.get("enable_structure_distance", False)),
        enable_locality_ratio=bool(mapping.get("enable_locality_ratio", True)),
        lpips_net=str(mapping.get("lpips_net", "squeeze")),
        clip_model_id=str(mapping.get("clip_model_id", "openai/clip-vit-large-patch14")),
        clip_local_files_only=bool(mapping.get("clip_local_files_only", True)),
        dino_model_name=str(mapping.get("dino_model_name", "dino_vits8")),
        dino_global_patch_size=int(mapping.get("dino_global_patch_size", 224)),
        dino_repo_or_dir=_resolve_optional_path(mapping.get("dino_repo_or_dir"), base_dir=base_dir),
        dino_cache_dir=_resolve_optional_path(mapping.get("dino_cache_dir"), base_dir=base_dir),
        dino_weights_path=_resolve_optional_path(mapping.get("dino_weights_path"), base_dir=base_dir),
        dino_local_files_only=bool(mapping.get("dino_local_files_only", True)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _load_mapping_file(config_path)
    base_dir = config_path.parent

    experiment_block = dict(payload.get("experiment") or {})
    if "input_manifest" not in experiment_block:
        raise ValueError("experiment.input_manifest is required")
    if "output_root" not in experiment_block:
        raise ValueError("experiment.output_root is required")

    method_mapping, _method_base_dir = _load_referenced_block(payload, key="method", base_dir=base_dir)
    model_mapping, model_base_dir = _load_referenced_block(payload, key="models", base_dir=base_dir)

    return ExperimentConfig(
        input_manifest=_resolve_optional_path(experiment_block["input_manifest"], base_dir=base_dir) or Path(),
        output_root=_resolve_optional_path(experiment_block["output_root"], base_dir=base_dir) or Path(),
        method=load_method_config(method_mapping),
        roi=load_roi_config(dict(payload.get("roi") or {}), base_dir=base_dir),
        inversion=load_inversion_config(dict(payload.get("inversion") or {}), base_dir=base_dir),
        models=load_model_config(model_mapping, base_dir=model_base_dir),
        runtime=load_runtime_config(dict(payload.get("runtime") or {})),
        save=load_save_config(dict(payload.get("save") or {})),
        metrics=load_metrics_config(dict(payload.get("metrics") or {}), base_dir=base_dir),
    )
