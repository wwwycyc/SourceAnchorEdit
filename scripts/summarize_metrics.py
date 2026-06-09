from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sourceanchor.metrics_summary import write_metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-sample metrics from a source_anchor run directory.")
    parser.add_argument("run_dir", help="Run directory containing samples/*/debug.json.")
    args = parser.parse_args()

    json_path, csv_path = write_metrics_summary(args.run_dir)
    print(f"[done] metrics_summary_json={json_path}")
    print(f"[done] metrics_summary_csv={csv_path}")


if __name__ == "__main__":
    main()
