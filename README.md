# Source Anchor

[English](README.md) | [中文](README_zh.md)

High-fidelity image editing method based on diffusion models.

## Features

- High-fidelity editing: preserve non-edited regions
- Source anchoring mechanism: ensure background structure stability
- Dynamic masking: precise editing region control
- Support ROI caching for acceleration

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

### 2. Configure Model Paths

Copy and edit the configuration file:

```bash
cp configs/models/local_models.example.yaml configs/models/local_models.local.yaml
```

Edit `local_models.local.yaml` to set model paths:

```yaml
models:
  sd_model: runwayml/stable-diffusion-v1-5  # or local path
  clip_model: openai/clip-vit-large-patch14  # or local path
  dino_weights: null  # optional
```

## Quick Start

### Single Sample Editing

```bash
python scripts/run_single.py --config configs/experiments/source_anchor.demo.example.yaml
```

### Batch Processing

```bash
python scripts/run_batch.py --config configs/experiments/source_anchor.use_cache.example.yaml
```

### Build ROI Cache (optional, speeds up repeated experiments)

```bash
python scripts/build_roi_cache.py --config configs/experiments/source_anchor.build_cache.example.yaml
```

### Launch Web Demo

```bash
python scripts/launch_web_demo.py
```

Then open the displayed URL in your browser.

## Input Format

Create a `sample.json`:

```json
{
  "sample_id": "example_001",
  "source_image_path": "path/to/source.png",
  "source_prompt": "a cat sitting on a chair",
  "target_prompt": "a dog sitting on a chair",
  "editing_region": "auto"
}
```

See [docs/input_format.md](docs/input_format.md) for details.

## Output

Results are saved in the `runs/` directory:

```
runs/
  source_anchor_<timestamp>/
    samples/
      <sample_id>/
        source.png              # original image
        edited.png              # editing result
        source_reconstruction.png  # reconstructed image
        roi_soft.png            # ROI visualization (soft mask)
        roi_hard.png            # ROI visualization (hard mask)
        overview.png            # complete comparison
        steps/                  # optional per-step traces, diagnostics, and maps
        debug.json              # debug information
```

## Configuration

Configuration files use a layered structure:

- `configs/models/` - Model path configuration
- `configs/methods/` - Method parameter configuration
- `configs/experiments/` - Experiment configuration

See [docs/config.md](docs/config.md) for details.

## Project Structure

```
source_anchor_release/
├── configs/              # Configuration files
├── docs/                 # Documentation
├── examples/             # Example data
├── scripts/              # Run scripts
├── src/sourceanchor/     # Core code
│   ├── inversion/        # Image inversion
│   ├── method/           # Core algorithms
│   ├── roi/              # ROI generation
│   └── runtime/          # Runtime components
└── tools/                # Utility tools
```

## FAQ

### 1. How to use local models?

Set local paths in `local_models.local.yaml`:

```yaml
models:
  sd_model: /path/to/stable-diffusion-v1-5
  clip_model: /path/to/clip-vit-large-patch14
```

### 2. How to adjust editing strength?

Adjust `guidance_scale` in method configuration:

```yaml
method:
  guidance_scale: 7.5  # default, increase for stronger edits
```

### 3. Out of GPU memory?

Enable memory optimization options:

```yaml
runtime:
  attention_slicing: true
  vae_slicing: true
  enable_cpu_offload: true  # most aggressive option
```

## License

See [LICENSE](LICENSE) file.

## Documentation

- [Input Format](docs/input_format.md)
- [Configuration Guide](docs/config.md)
- [Method Description](docs/method.md)
- [Reproducibility Guide](docs/reproducibility.md)
