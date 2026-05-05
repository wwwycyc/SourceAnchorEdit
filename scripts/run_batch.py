from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sourceanchor.runner import run_from_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the standalone source_anchor method on a batch manifest.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and sample loading.")
    args = parser.parse_args()

    run_from_config(args.config, dry_run=args.dry_run)
