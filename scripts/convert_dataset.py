from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sourceanchor.datasets.piebench import PIEBenchAdapter
from sourceanchor.outputs.writer import ensure_directory, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a dataset into the standard sourceanchor sample format.")
    parser.add_argument("--dataset", required=True, choices=["piebench", "manifest"], help="Dataset adapter name.")
    parser.add_argument("--input", required=True, help="Dataset source path.")
    parser.add_argument("--output", required=True, help="Output directory or manifest path.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if args.dataset == "manifest":
        samples: list[str] = []
        if input_path.is_dir():
            for sample_json in sorted(input_path.rglob("sample.json")):
                samples.append(str(sample_json.resolve()))
        elif input_path.is_file():
            samples.append(str(input_path.resolve()))
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        if not samples:
            raise ValueError(f"No sample.json files found under: {input_path}")
        ensure_directory(output_path.parent)
        write_json(output_path, {"samples": samples})
        print(f"[done] manifest={output_path}")
        return

    ensure_directory(output_path)
    adapter = PIEBenchAdapter(input_path)
    samples = adapter.export(output_path)
    print(f"[done] dataset={args.dataset}")
    print(f"[done] samples={len(samples)}")
    print(f"[done] manifest={output_path / 'manifest.json'}")


if __name__ == "__main__":
    main()
