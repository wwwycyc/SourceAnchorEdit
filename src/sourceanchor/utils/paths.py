from __future__ import annotations

from pathlib import Path


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
