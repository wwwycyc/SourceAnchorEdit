from __future__ import annotations

from pathlib import Path

from sourceanchor.config import ExperimentConfig, load_experiment_config
from sourceanchor.input.loader import load_samples_from_path
from sourceanchor.inversion import save_inversion_output
from sourceanchor.method.editor import SourceAnchorEditor, run_experiment
from sourceanchor.outputs import inversion_cache_dir
from sourceanchor.outputs.writer import write_json
from sourceanchor.roi.builder import build_roi_cache_for_samples
from sourceanchor.utils.image import load_rgb_image


def load_config_and_samples(config_path: str | Path) -> tuple[ExperimentConfig, list]:
    """Load one experiment config and resolve it to a list of standard samples."""
    config = load_experiment_config(config_path)
    samples = load_samples_from_path(config.input_manifest)
    return config, samples


def print_preflight(
    config_path: str | Path,
    config: ExperimentConfig,
    samples: list,
    *,
    cache_root: Path | None = None,
    cache_label: str = "cache_root",
) -> None:
    print(f"[preflight] config={config_path}")
    print(f"[preflight] output_root={config.output_root}")
    if cache_root is not None:
        print(f"[preflight] {cache_label}={cache_root}")
    print(f"[preflight] samples={len(samples)}")


def run_from_config(config_path: str | Path, *, dry_run: bool = False) -> Path | None:
    """Main entry used by single-run and batch-run scripts."""
    config, samples = load_config_and_samples(config_path)
    print_preflight(config_path, config, samples)
    if dry_run:
        return None
    run_dir = run_experiment(config, samples)
    print(f"[done] run_dir={run_dir}")
    return run_dir


def build_roi_cache_manifest_payload(config: ExperimentConfig, results: list, cache_root: Path) -> dict:
    return {
        "roi_source": "live",
        "roi_cache_root": str(cache_root),
        "sample_count": len(results),
        "samples": [
            {
                "sample_id": item.sample_id,
                "cache_dir": str(item.cache_dir),
                "source": item.source,
            }
            for item in results
        ],
        "config_snapshot_path": str(cache_root / "source_anchor_cache_config.json"),
    }


def build_roi_cache_from_config(
    config_path: str | Path,
    *,
    cache_root_override: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Build ROI cache from a live-ROI config and export a cache manifest."""
    config, samples = load_config_and_samples(config_path)
    cache_root = Path(cache_root_override).expanduser().resolve() if cache_root_override else config.roi.cache_root
    if cache_root is None:
        cache_root = (config.output_root / "roi_cache").resolve()
    config.roi.cache_root = cache_root
    config.roi.source = "live"

    print_preflight(config_path, config, samples, cache_root=cache_root, cache_label="roi_cache_root")
    if dry_run:
        return None

    editor = SourceAnchorEditor(config)
    results = build_roi_cache_for_samples(editor, config, samples)
    write_json(cache_root / "source_anchor_cache_config.json", config.to_serializable_dict())
    manifest_path = write_json(cache_root / "cache_manifest.json", build_roi_cache_manifest_payload(config, results, cache_root))
    print(f"[done] roi_cache_root={cache_root}")
    print(f"[done] roi_cache_manifest={manifest_path}")
    return manifest_path


def build_inversion_cache_manifest_payload(config: ExperimentConfig, results: list[dict], cache_root: Path) -> dict:
    return {
        "inversion_source": "live",
        "inversion_cache_root": str(cache_root),
        "sample_count": len(results),
        "samples": results,
        "config_snapshot_path": str(cache_root / "source_anchor_inversion_cache_config.json"),
    }


def build_inversion_cache_from_config(
    config_path: str | Path,
    *,
    cache_root_override: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Build DDIM inversion cache and export a cache manifest."""
    config, samples = load_config_and_samples(config_path)
    cache_root = Path(cache_root_override).expanduser().resolve() if cache_root_override else config.inversion.cache_root
    if cache_root is None:
        cache_root = (config.output_root / "inversion_cache").resolve()
    config.inversion.cache_root = cache_root
    config.inversion.source = "live"

    print_preflight(config_path, config, samples, cache_root=cache_root, cache_label="inversion_cache_root")
    if dry_run:
        return None

    editor = SourceAnchorEditor(config)
    results: list[dict] = []
    for sample in samples:
        source_image = load_rgb_image(sample.source_image_path)
        inversion = editor.inversion_backend.invert(source_image, source_prompt=sample.source_prompt)
        cache_dir = inversion_cache_dir(cache_root, sample.sample_id)
        save_inversion_output(cache_dir, inversion)
        results.append(
            {
                "sample_id": sample.sample_id,
                "cache_dir": str(cache_dir.resolve()),
                "backend": inversion.metadata.get("backend"),
            }
        )

    write_json(cache_root / "source_anchor_inversion_cache_config.json", config.to_serializable_dict())
    manifest_path = write_json(
        cache_root / "inversion_cache_manifest.json",
        build_inversion_cache_manifest_payload(config, results, cache_root),
    )
    print(f"[done] inversion_cache_root={cache_root}")
    print(f"[done] inversion_cache_manifest={manifest_path}")
    return manifest_path
