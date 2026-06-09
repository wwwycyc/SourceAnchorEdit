# Metrics And Batch Completion Notes

This file records the current completion state for the refactored
`sourceanchor` line.

## Completed

- Native DDIM inversion is integrated.
- ROI live generation and ROI cache read/write are integrated.
- Inversion cache read/write is integrated.
- Source/target two-branch denoising is integrated.
- DyMask, effective mask, anchor mask, and source anchoring are integrated.
- True batch denoising is integrated for compatible samples in the same chunk.
- Metrics are integrated, including LPIPS, PSNR, MSE, SSIM, CLIP, locality, and DINO structure distance.

## Verified Commands

```powershell
conda activate imgedit
python -m compileall src scripts
python scripts\run_batch.py --config configs\experiments\source_anchor.batch2.example.yaml
python scripts\run_batch.py --config configs\experiments\test_metrics_dino.yaml
```

Observed batch verification:

- `batch_case_0001`: `batch_size=2`, `batch_index=0`, `steps=10`
- `batch_case_0002`: `batch_size=2`, `batch_index=1`, `steps=10`

Observed DINO verification:

- `structure_distance`: approximately `0.01855`
- `structure_distance_unedit_part`: approximately `0.01172`

## Notes

The denoising loop is batched. Inversion and live ROI preparation remain
per-sample so that cache handling, image size differences, and DiffEdit mask
generation stay predictable. If samples in a chunk have incompatible latent or
mask shapes, the runner falls back to per-sample denoising for that chunk.
