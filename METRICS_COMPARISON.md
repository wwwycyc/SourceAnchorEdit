# Metrics Comparison

The current `sourceanchor` release now covers the practical metric set used by
the legacy DyMask line while keeping the expensive parts optional.

## Implemented Metrics

- Source reconstruction quality: `source_recon_psnr`, `source_recon_lpips`
- Source vs edited fidelity: `psnr`, `mse`, `ssim`, `lpips`
- Prompt alignment: `clip_similarity`, `clip_score`
- Edit-region prompt alignment: `clip_similarity_edit_part`, `clip_score_edit_part`
- Outside/edit locality: `outside_mse`, `outside_psnr`, `outside_ssim`, `outside_lpips`
- Perceptual locality: `locality_ratio`
- DINO structure distance: `structure_distance`, `structure_distance_unedit_part`

## Legacy DyMask Parity

Legacy DyMask included DINO-based structure distance. The release version now
implements it with local-first loading:

```yaml
metrics:
  enable_structure_distance: true
  dino_model_name: dino_vits8
  dino_repo_or_dir: .cache/torch/hub/facebookresearch_dino_main
  dino_cache_dir: .cache/torch/hub
  dino_weights_path: .cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth
  dino_local_files_only: true
```

If the local repo or checkpoint is missing, local-only mode records a clear
metric error in `debug.json` instead of silently downloading.

## Defaults

DINO structure distance is disabled by default because it requires extra model
assets and is slower than PSNR/SSIM/MSE. Enable it only for paper-grade
evaluation or when comparing against the old DyMask metric table.
