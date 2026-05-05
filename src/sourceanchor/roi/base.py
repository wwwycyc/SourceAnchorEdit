from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RoiPayload:
    sample_id: str
    mask: np.ndarray
    source: str
    cache_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoiBuildResult:
    sample_id: str
    cache_dir: Path
    source: str
