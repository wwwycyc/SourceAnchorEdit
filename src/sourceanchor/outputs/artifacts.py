from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(output_root: str | Path, prefix: str = "source_anchor") -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sample_dir(run_dir: str | Path, sample_id: str) -> Path:
    path = Path(run_dir).expanduser().resolve() / "samples" / sample_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def roi_cache_dir(cache_root: str | Path, sample_id: str) -> Path:
    path = Path(cache_root).expanduser().resolve() / sample_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def inversion_cache_dir(cache_root: str | Path, sample_id: str) -> Path:
    path = Path(cache_root).expanduser().resolve() / sample_id
    path.mkdir(parents=True, exist_ok=True)
    return path
