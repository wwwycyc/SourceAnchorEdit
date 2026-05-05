from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sourceanchor.config import load_experiment_config
from sourceanchor.input.loader import load_samples_from_path


def main() -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "source_anchor.demo.example.yaml")
    samples = load_samples_from_path(config.input_manifest)
    print("smoke-ok")
    print(f"samples={len(samples)}")


if __name__ == "__main__":
    main()
