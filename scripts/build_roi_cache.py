from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sourceanchor.runner import build_roi_cache_from_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ROI cache for the standalone source-anchor release.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--cache-root", default=None, help="Optional override for roi.cache_root.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config, cache root, and sample loading.")
    args = parser.parse_args()

    build_roi_cache_from_config(
        args.config,
        cache_root_override=args.cache_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
