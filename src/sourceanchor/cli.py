from __future__ import annotations

import argparse

from .runner import run_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone source-anchor release.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and inputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_from_config(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
