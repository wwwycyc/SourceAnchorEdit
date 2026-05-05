# Reproducibility

## Environment

Install PyTorch first for your CUDA / CPU environment, then install project dependencies.

Example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Or install the new package in editable mode:

```powershell
pip install -e .
```

## Required External Assets

These must be configured through YAML files:

- Stable Diffusion model
- CLIP model
- NTIP2P root
- optional DINO weights

Do not hard-code such paths inside the method implementation.

For local-only secrets and machine-specific paths, use untracked `*.local.yaml` files.

## Minimal Single-Case Run

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.demo.example.yaml
```

## ROI Cache Workflow

### 1. Build cache

```powershell
python scripts\build_roi_cache.py --config configs\experiments\source_anchor.build_cache.example.yaml
```

### 2. Run with cache

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.use_cache.example.yaml
```

## Web Demo Config

The web demo should use two config files:

- tracked template:
  - `configs/web/source_anchor_web.example.yaml`
- local private file:
  - `configs/web/source_anchor_web.local.yaml`

Keep the local file out of Git and store your API key there.

## Standard Input

The method only accepts standard `sample.json` inputs.

See:

- [docs/input_format.md](input_format.md)

## Expected Output Structure

```text
runs/
  source_anchor_demo/
    source_anchor_<timestamp>/
      run_manifest.json
      samples/
        <sample_id>/
          source.png
          source_reconstruction.png
          edited.png
          roi_soft.png
          roi_hard.png
          debug.json
          overview.png
```

## Dataset Scope

For this release effort, the official dataset conversion target is:

- `PIE-Bench`

## Current Validation Status

The following configs have already been exercised in the current local environment:

- `source_anchor.demo.example.yaml`
- `source_anchor.build_cache.example.yaml`
- `source_anchor.use_cache.example.yaml`
