# Standard Input Format

The standalone `source_anchor` pipeline accepts one dataset-agnostic sample format only.
All dataset-specific records must be converted into this format before they are passed to the method.

## Goals

- decouple the method from any specific dataset schema
- support both single-case and batch workflows
- keep the algorithm layer independent from dataset-specific fields
- make open-source reproduction easier

## Required Files

Each case requires at least:

- `sample.json`
- `source image`

Example layout:

```text
examples/
  single_case/
    sample.json
    source.png
```

## Required JSON Fields

```json
{
  "sample_id": "case_0001",
  "source_image_path": "source.png",
  "source_prompt": "a long-haired tabby cat sitting on a wooden chair indoors",
  "target_prompt": "a long-haired dog sitting on a wooden chair indoors",
  "metadata": {
    "dataset": "custom",
    "record_id": "custom_0001"
  }
}
```

Field meaning:

- `sample_id`
  - unique case identifier inside a run
- `source_image_path`
  - absolute path or path relative to `sample.json`
- `source_prompt`
  - English source-side prompt
- `target_prompt`
  - English target-side prompt used directly by the final method
- `metadata`
  - optional container for non-core information
  - the method does not depend on any field inside it

## Optional Fields

```json
{
  "sample_id": "case_0001",
  "source_image_path": "source.png",
  "source_prompt": "a long-haired tabby cat sitting on a wooden chair indoors",
  "target_prompt": "a long-haired dog sitting on a wooden chair indoors",
  "metadata": {
    "dataset": "piebench",
    "record_id": "row_1024",
    "edit_instruction": "change the cat to a dog, keep everything else unchanged"
  },
  "target_reference_path": null,
  "mask_path": null
}
```

These are optional:

- `target_reference_path`
  - only for optional evaluation or manual comparison
- `mask_path`
  - dataset-provided GT/edit mask used by masked metrics such as edit-region CLIP
  - method-generated ROI masks must not be used here

## Explicit Non-Fields

The final `source_anchor` method does **not** depend on these legacy fields:

- `target_token_hints`
- `blended_word`
- bracket markup
- any weak target-side hints

If such fields exist in historical records, they should be ignored during dataset conversion.

## Batch Manifest

For batch runs, use a manifest that points to multiple standard samples:

```json
{
  "samples": [
    "examples/single_case/sample.json",
    "examples/another_case/sample.json"
  ]
}
```

The runner only reads standard samples. It does not need to know which dataset they came from.

## Dataset Adapter Boundary

Dataset adapters are responsible for only two things:

1. reading raw dataset records
2. exporting standard `sample.json` files

The method layer, ROI layer, inversion layer, and visualization layer must only read the standard input format.
