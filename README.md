# Source Anchor

[English](README.md) | [中文](README_zh.md)

This repository is being refactored toward a standalone, reproducible release centered on one final method line:

- source anchor
- no temporal accumulation
- no weak prompt hints
- ROI always enabled
- ROI source can be `live` or `cache`

The new implementation lives under [src/sourceanchor](src/sourceanchor).
The clean release files in this folder are intended to be uploaded as a separate GitHub repository.

## Main Entry Points

- single-run CLI: [scripts/run_single.py](scripts/run_single.py)
- batch-run CLI: [scripts/run_batch.py](scripts/run_batch.py)
- ROI cache builder: [scripts/build_roi_cache.py](scripts/build_roi_cache.py)
- dataset conversion: [scripts/convert_dataset.py](scripts/convert_dataset.py)
- web demo launcher: [scripts/launch_web_demo.py](scripts/launch_web_demo.py)

## Core Docs

- input format: [docs/input_format.md](docs/input_format.md)
- configuration layout: [docs/config.md](docs/config.md)
- method overview: [docs/method.md](docs/method.md)
- reproducibility notes: [docs/reproducibility.md](docs/reproducibility.md)

## Quick Start

Run the standalone example:

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.demo.example.yaml
```

Build ROI cache:

```powershell
python scripts\build_roi_cache.py --config configs\experiments\source_anchor.build_cache.example.yaml
```

Run with cached ROI:

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.use_cache.example.yaml
```

Launch the local visualization demo:

```powershell
python scripts\launch_web_demo.py
```

## Included in This Release

- layered config loading
- standard sample format loading
- standalone single-sample execution
- batch execution entry point
- ROI live/cache workflow
- ROI cache builder
- PIE-Bench dataset conversion
- local web demo backend

## Before Uploading

- keep `configs/models/*.example.yaml` and `configs/web/*.example.yaml`
- do not commit any `*.local.yaml` or `*.local.json`
- put your real model paths in local config files
- put your real API key only in `configs/web/source_anchor_web.local.yaml`
- choose and add an explicit open-source license
