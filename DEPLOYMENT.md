# SourceAnchor Deployment Guide

This guide describes the files and model assets needed to run the current
`sourceanchor` release on a new machine.

## Environment

- Python 3.10 or newer
- CUDA-capable PyTorch if running on GPU
- Stable Diffusion v1.5 local cache or network access to Hugging Face
- Optional metric assets:
  - CLIP ViT-L/14 for CLIP score
  - DINO torch hub repo and checkpoint for structure distance

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

In the local conda setup used for validation:

```powershell
conda activate imgedit
```

## Required Model

The editing pipeline requires Stable Diffusion v1.5:

```yaml
models:
  sd_model: runwayml/stable-diffusion-v1-5
```

For offline use, point `sd_model` to the local snapshot directory, for example:

```yaml
models:
  sd_model: C:/Users/11939/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/<hash>
```

The SD v1.5 snapshot must include `tokenizer`, `text_encoder`, `vae`, `unet`,
`scheduler`, and `model_index.json`.

## Optional Metrics Assets

CLIP score uses:

```yaml
metrics:
  clip_model_id: openai/clip-vit-large-patch14
  clip_local_files_only: true
```

DINO structure distance supports local-first loading:

```yaml
metrics:
  enable_structure_distance: true
  dino_model_name: dino_vits8
  dino_repo_or_dir: .cache/torch/hub/facebookresearch_dino_main
  dino_cache_dir: .cache/torch/hub
  dino_weights_path: .cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth
  dino_local_files_only: true
```

If `dino_weights_path` is null, the code automatically looks under
`dino_cache_dir/checkpoints` for the expected checkpoint filename.

## Run Examples

Single sample:

```powershell
conda activate imgedit
python scripts\run_batch.py --config configs\experiments\source_anchor.demo.example.yaml
```

Build ROI and inversion cache:

```powershell
python scripts\run_batch.py --config configs\experiments\source_anchor.build_cache.example.yaml
```

Use cached ROI and inversion:

```powershell
python scripts\run_batch.py --config configs\experiments\source_anchor.use_cache.example.yaml
```

True batch denoising smoke test:

```powershell
python scripts\run_batch.py --config configs\experiments\source_anchor.batch2.example.yaml
```

DINO structure distance smoke test:

```powershell
python scripts\run_batch.py --config configs\experiments\test_metrics_dino.yaml
```

Runtime outputs are written under `runs/`, which is ignored by Git.
