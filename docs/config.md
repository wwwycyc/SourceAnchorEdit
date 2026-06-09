# Configuration Layout

The standalone `source_anchor` release uses layered configuration instead of one oversized config file.

Goals:

- every external model path is configurable
- method logic, ROI source, runtime, and saving policy stay separated
- dataset conversion stays decoupled from the core method

## Recommended Layout

```text
configs/
  methods/
    source_anchor.example.yaml
  models/
    local_models.example.yaml
  experiments/
    source_anchor.demo.example.yaml
    source_anchor.build_cache.example.yaml
    source_anchor.use_cache.example.yaml
```

## Method Config

`configs/methods/source_anchor.example.yaml`

```yaml
method:
  name: source_anchor
  no_target_hints: true
  source_anchor_start: 0.25
  num_inversion_steps: 10
  num_edit_steps: 10
  guidance_scale: 7.5
```

Notes:

- `no_target_hints` is fixed to `true`
- it is not a feature toggle
- it simply records the final experimental definition

## Model Config

`configs/models/local_models.example.yaml`

```yaml
models:
  sd_model: runwayml/stable-diffusion-v1-5
  clip_model: openai/clip-vit-large-patch14
  dino_weights: null
```

Notes:

- all external model locations must be injected through config
- `sd_model` and `clip_model` may be either local paths or repo ids
- model sources must not be hard-coded inside method code

For local use, create an untracked file such as:

```text
configs/models/local_models.local.yaml
```

and keep it out of Git.

## Experiment Config

`configs/experiments/source_anchor.demo.example.yaml`

```yaml
experiment:
  input_manifest: ../../examples/single_case/sample.json
  output_root: ../../runs/source_anchor_demo

method:
  config: ../methods/source_anchor.example.yaml

models:
  config: ../models/local_models.example.yaml

roi:
  source: live
  cache_root: null
  save_cache: false
  threshold: 0.5
  num_maps_per_mask: 1
  mask_encode_strength: 0.5
  mask_thresholding_ratio: 3.0

runtime:
  device: cuda
  dtype: float16
  batch_size: 1
  local_files_only: true
  attention_slicing: true
  vae_slicing: true
  channels_last: true
  enable_tf32: true
  enable_cpu_offload: false
  enable_xformers: false

save:
  roi_cache: false
  inversion_tensors: false
  debug_json: true
  step_visualizations: false
  overview: true
```

## ROI Semantics

ROI is always enabled. Only its **source** changes:

- `roi.source: live`
  - generate ROI during the current run
  - do not read historical cache

- `roi.source: cache`
  - read ROI from `roi.cache_root`
  - use this when you want fixed ROI conditions across runs

Recommended workflow:

1. build cache with `source_anchor.build_cache.example.yaml`
2. run experiments with `source_anchor.use_cache.example.yaml`

## Save Policy

The `save` section controls only whether intermediate artifacts are written to disk:

- `roi_cache`
- `inversion_tensors`
- `debug_json`
- `step_visualizations`
- `overview`

This keeps “method uses something internally” separate from “artifact is written out”.

## Dataset Scope

For the current release effort, the officially supported dataset adapter target is:

- `PIE-Bench`

Other datasets can be added later by exporting the same standard `sample.json` format.
