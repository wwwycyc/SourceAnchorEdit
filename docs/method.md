# Source Anchor Method

## Final Scope

This standalone release keeps one final method line only:

- source anchor
- no temporal accumulation
- no weak prompt hints
- ROI always enabled
- ROI source supports `live` and `cache`

## Prompt Handling Principle

“No target hints” is **not** treated as an independent module.

It simply means:

- the target branch uses the full `target_prompt`
- `target_token_hints` are ignored
- bracket markup is ignored
- no extra weak target-side information is introduced

The purpose is to keep the prompt handling logic minimal and fair across experiments.

## High-Level Pipeline

```text
source image
  -> DDIM inversion with source prompt
  -> source latent trajectory

target prompt
  -> prompt encoding
  -> target branch denoising

source branch + target branch
  -> discrepancy / attention / latent drift
  -> dynamic mask
  -> ROI-guided effective mask
  -> source-anchor blend
  -> edited image
```

## ROI Semantics

- `roi.source = live`
  - compute ROI during the current run
  - do not reuse cache

- `roi.source = cache`
  - read ROI from a specified cache root
  - useful when ROI conditions must stay fixed across experiments

ROI is always used. The only question is where it comes from.

## Current Implementation Status

The current `sourceanchor` package already supports:

- layered configuration
- standard sample loading
- single-case execution
- batch execution
- ROI live/cache workflow
- ROI cache building
- local web demo backend integration

Still being polished:

- broader automated tests
- cleaner release-oriented docs
- tighter packaging and naming cleanup
