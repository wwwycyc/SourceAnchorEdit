from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sourceanchor.datasets.piebench import PIEBenchAdapter, decode_piebench_rle
from sourceanchor.input.loader import load_samples_from_path


def test_decode_piebench_rle_matches_original_boundary_rule() -> None:
    mask = decode_piebench_rle([5, 2], height=4, width=4)

    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    assert mask[1, 1] == 1
    assert mask[1, 2] == 1
    assert np.all(mask[0, :] == 1)
    assert np.all(mask[-1, :] == 1)
    assert np.all(mask[:, 0] == 1)
    assert np.all(mask[:, -1] == 1)


def test_piebench_adapter_exports_gt_mask_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "piebench"
    image_dir = dataset_root / "annotation_images" / "0_random_140"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_dir / "000000000000.jpg")
    (dataset_root / "mapping_file.json").write_text(
        json.dumps(
            {
                "000000000000": {
                    "image_path": "0_random_140/000000000000.jpg",
                    "original_prompt": "a [round] cake",
                    "editing_prompt": "a [square] cake",
                    "editing_instruction": "change cake shape",
                    "editing_type_id": "0",
                    "blended_word": "cake cake",
                    "mask": [5, 2],
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "converted"
    samples = PIEBenchAdapter(dataset_root).export(output_dir)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.mask_path is not None
    assert sample.mask_path.name == "gt_mask.png"
    assert sample.mask_path.exists()
    assert sample.metadata.edit_instruction == "change cake shape"
    assert sample.metadata.extras["has_gt_mask"] is True

    exported = load_samples_from_path(output_dir / "manifest.json")[0]
    assert exported.mask_path is not None
    assert exported.mask_path.exists()
    exported_mask = np.asarray(Image.open(exported.mask_path).convert("L"))
    assert exported_mask.shape == (4, 4)
    assert exported_mask[1, 1] == 255
    assert exported_mask[1, 2] == 255
